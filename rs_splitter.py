"""纯规则回复切分核心。

设计要点（两个正交机制）：
  1. 保护区（protected intervals）—— 解决「不切坏」。
     代码块、行内代码、URL、@提及、括号对、颜文字先被标记为原子区间，
     任何切点都不允许落在区间内部。
  2. 候选边界优先级（cut priority）—— 解决「切点生硬」。
     每个可落刀位置按 段落 > 换行 > 句末标点 > 分号/冒号 > 逗号 > 空格 > 硬切
     打分，再在长度预算内挑「最靠后 + 优先级最高」的那一刀。

本模块不依赖 self.ctx、不做任何 I/O、不用随机数，因此可以脱机单测。
plugin.py 只负责把配置喂进来、把结果发出去。
"""

import re
from dataclasses import dataclass

__all__ = ["SplitRules", "split_reply", "protected_intervals", "cut_candidates"]

# ── 标点分类 ────────────────────────────────────────────────────────────────

# 句末：最高性价比的切点。
# 注意必须包含 ASCII 句点，否则纯英文回复没有任何「句末」切点可用。
# 小数 / 版本号 / 文件名里的点不会被误判为切点——
# 因为两侧字符都算「单词内部」时切点会被直接过滤掉。
SENTENCE_ENDERS = "。！？!?…‥."
# 次强：分号 / 冒号 / 波浪号（语气未完但语义可断）
SOFT_ENDERS = "；;：:~～"
# 再次：顿号 / 逗号
COMMA_ENDERS = "，,、"
# 收尾符：跟在句末标点后，切点应落到它后面而不是前面
CLOSERS = "」』】）》〉”’\"'）)]}"

# 不切开英文单词 / 数字 / 版本号 / 路径
_WORDISH = set("._-+/#@%")

# ── 颜文字字符类 ────────────────────────────────────────────────────────────
# 只收「装饰性符号」，刻意不含全角标点区（FF00-FF5F），
# 否则中文的 ，！？ 会被误判成颜文字而阻断分句。
_KAOMOJI_RANGES = (
    (0x2190, 0x21FF),  # 箭头
    (0x2200, 0x22FF),  # 数学运算符
    (0x2500, 0x257F),  # 制表符
    (0x2580, 0x259F),  # 方块元素
    (0x25A0, 0x25FF),  # 几何图形
    (0x2600, 0x27BF),  # 杂项符号 + 装饰符号
    (0x2E80, 0x2EFF),  # CJK 部首补充
    (0x30FB, 0x30FB),  # ・
    (0x30FC, 0x30FC),  # ー
    (0xFE30, 0xFE4F),  # CJK 兼容形式
    (0xFF61, 0xFF9F),  # 半角片假名（颜文字主力）
    (0x1F300, 0x1FAFF),  # emoji
)
# 混合字符类：单独出现不足以判定颜文字，但可参与构成
_KAOMOJI_WEAK_RANGES = ((0x00A0, 0x00FF),)

_SPLIT_PUNCT = set(SENTENCE_ENDERS + SOFT_ENDERS + COMMA_ENDERS + "\n\r\t ")


def _in_ranges(code: int, ranges) -> bool:
    return any(lo <= code <= hi for lo, hi in ranges)


def _is_kaomoji_strong(ch: str) -> bool:
    if ch in _SPLIT_PUNCT or ch.isascii():
        return False
    return _in_ranges(ord(ch), _KAOMOJI_RANGES)


def _is_kaomoji_weak(ch: str) -> bool:
    if ch in _SPLIT_PUNCT or ch.isascii():
        return False
    return _in_ranges(ord(ch), _KAOMOJI_WEAK_RANGES)


@dataclass(frozen=True)
class SplitRules:
    """切分规则。全部为纯参数，便于配置注入与单测。"""

    min_length: int = 60
    soft_max_length: int = 180
    hard_max_length: int = 480
    max_segments: int = 6
    # 短于该长度的段会被并回相邻段，避免出现「。」这种被甩出来的孤立尾巴
    min_segment_length: int = 12
    protect_code: bool = True
    protect_url: bool = True
    protect_brackets: bool = True
    protect_kaomoji: bool = True
    # 换行是模型表达「这句单独发一条」的唯一手段，默认无条件尊重它：
    # 否则「剩余内容装得下就收尾」这条优化会把换行吞掉，模型想分条也分不了。
    force_split_on_newline: bool = True
    # 目标段长的下限比例：低于该比例的切点只在没有更靠后选择时才用
    late_window_ratio: float = 0.55


# ── 保护区 ──────────────────────────────────────────────────────────────────

# 顺序有讲究：围栏代码块必须先于行内代码，否则 ```a``` 会被切成两个行内代码
_FENCE_RE = None
_INLINE_CODE_RE = None
_URL_RE = None
_MENTION_RE = None
_BRACKET_RE = None


def _lazy_regex():
    global _FENCE_RE, _INLINE_CODE_RE, _URL_RE, _MENTION_RE, _BRACKET_RE
    if _FENCE_RE is not None:
        return

    _FENCE_RE = re.compile(r"```[^\n]*\n?.*?(?:```|\Z)", re.S)
    _INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
    _URL_RE = re.compile(
        r"(?:https?://|ftp://|www\.)[^\s<>（）()\[\]【】「」《》\"']+",
        re.I,
    )
    _MENTION_RE = re.compile(r"[@#][A-Za-z0-9_\-\u4e00-\u9fff]{1,32}")
    _BRACKET_RE = re.compile(
        r"[（(\[【《「][^（()）\[\]【】《》「」\n]{0,40}[）)\]】》」]"
    )


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """合并重叠/相邻区间。"""
    if not spans:
        return []
    spans = sorted(spans)
    merged = [list(spans[0])]
    for start, end in spans[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def protected_intervals(text: str, rules: SplitRules) -> list[tuple[int, int]]:
    """算出所有禁止落刀的原子区间（已合并重叠）。"""
    _lazy_regex()
    spans: list[tuple[int, int]] = []

    if rules.protect_code:
        spans += [(m.start(), m.end()) for m in _FENCE_RE.finditer(text)]
        spans += [(m.start(), m.end()) for m in _INLINE_CODE_RE.finditer(text)]
    if rules.protect_url:
        spans += [(m.start(), m.end()) for m in _URL_RE.finditer(text)]
    if rules.protect_brackets:
        spans += [(m.start(), m.end()) for m in _BRACKET_RE.finditer(text)]
    if rules.protect_kaomoji:
        spans += _kaomoji_spans(text)

    return _merge_spans(spans)


def _kaomoji_spans(text: str) -> list[tuple[int, int]]:
    """把「含装饰符号的连续非空白段」整体标记为原子。

    判定门槛：该段里至少要有一个强装饰字符（制表符/几何/箭头/emoji…），
    只有弱字符（拉丁补充区）不算，避免把普通外文单词错判成颜文字。
    """
    spans: list[tuple[int, int]] = []
    start = None
    strong = 0
    for index, ch in enumerate(text):
        if ch.isspace() or ch in SENTENCE_ENDERS:
            if start is not None and strong > 0:
                spans.append((start, index))
            start, strong = None, 0
            continue
        if start is None:
            start = index
        if _is_kaomoji_strong(ch):
            strong += 1
    if start is not None and strong > 0:
        spans.append((start, len(text)))
    return spans


# ── 候选切点 ────────────────────────────────────────────────────────────────

PRIORITY_PARAGRAPH = 0
PRIORITY_NEWLINE = 1
PRIORITY_SENTENCE = 2
PRIORITY_SOFT = 3
PRIORITY_COMMA = 4
# 汉字之间的空格：中文排版里不该出现空格，所以它一定是回复模型刻意打出的停顿，
# 属于子句边界。比逗号弱（真有逗号时逗号才是语法边界），但明显强于普通空格。
PRIORITY_CJK_SPACE = 5
# 普通空格：拉丁词之间、中英之间——这类空格是正常排版，弱边界
PRIORITY_SPACE = 6
PRIORITY_HARD = 9

# 判定「汉字之间的空格」时用于跳过连续空白
_SPACE_RUN = " \t\u3000"


def _cut_priority(text: str, pos: int) -> int:
    """给「在 text[pos-1] 与 text[pos] 之间落刀」打分，越小越优先。"""
    prev = text[pos - 1]
    if prev == "\n":
        # 段落（空行）优先于单换行
        if pos >= 2 and text[pos - 2] == "\n":
            return PRIORITY_PARAGRAPH
        return PRIORITY_NEWLINE
    if prev in SENTENCE_ENDERS:
        return PRIORITY_SENTENCE
    if prev in CLOSERS and pos >= 2 and text[pos - 2] in SENTENCE_ENDERS:
        return PRIORITY_SENTENCE
    if prev in SOFT_ENDERS:
        return PRIORITY_SOFT
    if prev in COMMA_ENDERS:
        return PRIORITY_COMMA
    if prev.isspace():
        # 汉字之间的空格 → 子句边界；其余（拉丁词间、中英之间）→ 弱边界
        if _is_han(_nearest_in(text, pos - 2, -1)) and _is_han(_nearest_in(text, pos, 1)):
            return PRIORITY_CJK_SPACE
        return PRIORITY_SPACE
    return PRIORITY_HARD


def _is_han(ch: str) -> bool:
    """是否为汉字（含扩展 A 区与兼容区）。"""
    if not ch:
        return False
    code = ord(ch)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
    )


def _nearest_in(text: str, index: int, step: int) -> str:
    """从 index 出发沿 step 方向跳过连续空白，返回第一个非空白字符。"""
    cursor = index
    while 0 <= cursor < len(text) and text[cursor] in _SPACE_RUN:
        cursor += step
    if 0 <= cursor < len(text):
        return text[cursor]
    return ""


def _is_wordish(ch: str) -> bool:
    """判断字符是否属于「单词内部」。

    只认 ASCII 字母数字 + 少量连接符号。**中文字符不算**——中文没有词边界，
    硬切时允许在汉字之间落刀；若把汉字也算进来，整段无标点中文将完全没有切点。
    """
    if ch in _WORDISH:
        return True
    return ch.isascii() and ch.isalnum()


def cut_candidates(text: str, protected: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """产出所有合法切点 (pos, priority)，按 pos 升序。"""
    candidates: list[tuple[int, int]] = []
    span_index = 0
    for pos in range(1, len(text)):
        # 落在保护区内部（严格内部）的位置一律非法
        while span_index < len(protected) and protected[span_index][1] <= pos:
            span_index += 1
        if span_index < len(protected):
            start, end = protected[span_index]
            if start < pos < end:
                continue
        before, after = text[pos - 1], text[pos]
        # 不切开省略号（... / ……）
        if before in ".…" and after in ".…":
            continue
        # 不切开英文单词 / 数字 / 版本号 / 路径残段
        if _is_wordish(before) and _is_wordish(after):
            continue
        candidates.append((pos, _cut_priority(text, pos)))
    return candidates


# ── 贪心打包 ────────────────────────────────────────────────────────────────


def _best_cut(
    candidates: list[tuple[int, int]],
    positions: list[int],
    low: int,
    high: int,
) -> tuple[int, int] | None:
    """在 (low, high] 区间内挑最优切点：优先级最高，同优先级取最靠后。"""
    from bisect import bisect_right

    start_index = bisect_right(positions, low)
    best: tuple[int, int] | None = None
    for index in range(start_index, len(candidates)):
        pos, priority = candidates[index]
        if pos > high:
            break
        if best is None or priority < best[1] or (priority == best[1] and pos > best[0]):
            best = (pos, priority)
    return best


def _extend_past_span(protected: list[tuple[int, int]], cut: int) -> int:
    """若切点落在保护区内部，把切点推到该区间之后。

    代价是这一段会超过硬上限，但换来「链接/代码块绝不从中间断开」——
    这正是本插件要修的核心问题，宁可超长也不切坏。
    """
    for start, end in protected:
        if start < cut < end:
            return end
    return cut


# 列表项行首：这类换行属于「排版」，不是「分条意图」，不该被强制切开
_LIST_LINE_RE = re.compile(r"^\s*(?:[-*+•·]|\d+[.、)]|[（(]\d+[）)])\s*")


def _inside_span(spans: list[tuple[int, int]], pos: int) -> bool:
    return any(start < pos < end for start, end in spans)


def _forced_blocks(
    text: str, rules: SplitRules, protected: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """按「强制分条边界」把文本切成块，块内再按长度预算细分。

    强制边界 = 段落（连续两个以上换行）与单换行。
    换行是回复模型表达「这句单独发一条」的唯一手段，必须无条件尊重 ——
    否则「剩余内容装得下就收尾」那条优化会把换行吞掉，模型想分条也分不了。

    两条例外：
      - **保护区内部的换行放行**。代码块本身就是多行的，若在这里强制分条，
        代码块会被从中切开，保护区形同虚设。优先级：保护区 > 强制边界。
      - 列表项之间的换行只是排版，不等同于分条。
    """
    total = len(text)
    if not rules.force_split_on_newline:
        return [(0, total)] if text.strip() else []

    blocks: list[tuple[int, int]] = []
    block_start = 0
    index = 0
    while index < total:
        if text[index] != "\n":
            index += 1
            continue

        run_end = index
        while run_end < total and text[run_end] == "\n":
            run_end += 1

        if _inside_span(protected, index):
            index = run_end
            continue

        is_forced = run_end - index >= 2  # 段落一定分条
        if not is_forced:
            line_end = text.find("\n", run_end)
            next_line = text[run_end : line_end if line_end != -1 else total]
            is_forced = _LIST_LINE_RE.match(next_line) is None

        if is_forced:
            blocks.append((block_start, index))
            block_start = run_end
        index = run_end

    blocks.append((block_start, total))
    return [(start, end) for start, end in blocks if text[start:end].strip()]


def _pack(
    text: str,
    candidates: list[tuple[int, int]],
    protected: list[tuple[int, int]],
    start: int,
    end: int,
    soft_max: int,
    hard_max: int,
    late_window_ratio: float,
) -> list[str]:
    """把 text[start:end] 这一段按长度预算贪心打包。

    只在块内切分，绝不越出 [start, end) —— 强制分条边界由调用方先切好。

    选刀顺序（宁可短一点，也要切口干净）：
      1. 靠后窗口内有「句末标点级」切点 → 取之（段更饱满）
      2. 否则全窗口内的「句末标点级」切点 → 容忍略短，换取干净切口
      3. 都没有 → 退回靠后窗口的逗号/空格级切点
      4. 再没有 → 全窗口的逗号/空格级切点
      5. 只剩硬切级 → 放宽到硬上限取最优，最后才是硬切（并避让保护区）
    """
    positions = [pos for pos, _ in candidates]
    segments: list[str] = []
    cursor = start

    while cursor < end:
        # 剩余内容本就装得下 → 直接收尾，不要再为了「找个切点」而硬切
        if end - cursor <= soft_max:
            segments.append(text[cursor:end].strip())
            break

        soft_limit = min(end, cursor + soft_max)
        late_low = cursor + max(1, int(soft_max * late_window_ratio))
        full = _best_cut(candidates, positions, cursor, soft_limit)
        late = _best_cut(candidates, positions, late_low, soft_limit) if late_low < soft_limit else None

        late_clean = late if late is not None and late[1] <= PRIORITY_SENTENCE else None
        full_clean = full if full is not None and full[1] <= PRIORITY_SENTENCE else None

        if late_clean is not None:
            cut = late_clean[0]
        elif full_clean is not None:
            cut = full_clean[0]
        elif late is not None and late[1] <= PRIORITY_SPACE:
            cut = late[0]
        elif full is not None and full[1] <= PRIORITY_SPACE:
            cut = full[0]
        elif late is not None:
            cut = late[0]
        else:
            hard_limit = min(end, cursor + hard_max)
            relaxed = _best_cut(candidates, positions, cursor, hard_limit)
            cut = relaxed[0] if relaxed is not None else hard_limit

        if cut <= cursor:
            cut = min(end, cursor + hard_max)
        cut = _extend_past_span(protected, cut)
        if cut >= end:
            segments.append(text[cursor:end].strip())
            break
        segments.append(text[cursor:cut].strip())
        cursor = cut

    return [segment for segment in segments if segment]


def _absorb_short_segments(segments: list[str], min_segment_length: int) -> list[str]:
    """把过短的段并回相邻段（纯拼接，不会破坏保护区）。"""
    if len(segments) <= 1 or min_segment_length <= 0:
        return segments
    merged: list[str] = []
    for segment in segments:
        if merged and len(segment) < min_segment_length:
            merged[-1] = merged[-1] + segment
        else:
            merged.append(segment)
    # 首段过短时没有「上一段」可并，改为并入下一段
    if len(merged) >= 2 and len(merged[0]) < min_segment_length:
        merged[1] = merged[0] + merged[1]
        merged.pop(0)
    return merged


def _merge_to_limit(segments: list[str], max_segments: int) -> list[str]:
    """段数超限时，按序「均匀并入相邻段」合并到上限（O(段数)）。

    原实现是「反复合并最短的相邻对」：每轮都要 O(段数) 扫描找最短对，整体 O(n²)。
    实测把 12000 段合并到 5 段要 **5.5 秒**；`soft=hard=1` 这类病态配置下
    48000 字要 **88 秒** —— 而本函数跑在 BLOCKING hook 里**同步执行**，
    会把事件循环整个卡住，bot 期间收不到任何消息。故改为按序均匀分组：
    复杂度 O(段数)，且各组长度更接近，结果更可预测。
    """
    count = len(segments)
    if max_segments <= 0 or count <= max_segments:
        return segments
    merged: list[str] = []
    cursor = 0
    for group in range(max_segments):
        # 余数摊给靠前的组，保证各组段数至多差 1
        take = count // max_segments + (1 if group < count % max_segments else 0)
        merged.append("".join(segments[cursor : cursor + take]))
        cursor += take
    return merged


def _pack_all(
    text: str,
    blocks: list[tuple[int, int]],
    candidates: list[tuple[int, int]],
    protected: list[tuple[int, int]],
    soft_max: int,
    hard_max: int,
    min_segment_length: int,
    late_window_ratio: float,
) -> list[str]:
    """逐块打包。

    短段吸收只在块内进行 —— 绝不跨越强制分条边界，
    否则「好\\n嗯」这种模型明确要求分两条的回复又会被合并回一条。
    """
    segments: list[str] = []
    for start, end in blocks:
        block_segments = _pack(
            text, candidates, protected, start, end, soft_max, hard_max, late_window_ratio
        )
        segments.extend(_absorb_short_segments(block_segments, min_segment_length))
    return segments


def split_reply(text: str, rules: SplitRules) -> list[str]:
    """把一段回复切成多条。切不动时返回原文单段，绝不返回空列表。"""
    if not text or not text.strip():
        return []
    stripped = text.strip()

    protected = protected_intervals(stripped, rules)
    candidates = cut_candidates(stripped, protected)

    # 强制分条边界（换行）不受长度预算约束，必须先切出来
    blocks = _forced_blocks(stripped, rules, protected)
    if len(blocks) <= 1:
        # 没有强制边界时，才轮到「太短就不切」这条保护
        if len(stripped) < rules.min_length:
            return [stripped]
        if not candidates:
            return [stripped]

    soft_max = max(1, rules.soft_max_length)
    hard_max = max(soft_max, rules.hard_max_length)
    max_segments = max(1, rules.max_segments)

    segments = _pack_all(
        stripped, blocks, candidates, protected, soft_max, hard_max,
        rules.min_segment_length, rules.late_window_ratio,
    )

    # 段数超限：先试着放宽目标长度重打包（切点更自然），仍超限再合并
    attempts = 0
    while len(segments) > max_segments and soft_max < hard_max and attempts < 8:
        soft_max = min(hard_max, max(soft_max + 1, -(-len(stripped) // max_segments)))
        segments = _pack_all(
            stripped, blocks, candidates, protected, soft_max, hard_max,
            rules.min_segment_length, rules.late_window_ratio,
        )
        attempts += 1

    # 最后一道：段数上限是显式配置，超限时允许跨强制边界合并
    if len(segments) > max_segments:
        segments = _merge_to_limit(segments, max_segments)

    return segments or [stripped]
