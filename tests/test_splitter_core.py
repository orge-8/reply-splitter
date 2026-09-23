"""splitter_core 规则引擎的行为测试。

核心不变量（每条用例都要成立）：
  1. 零内容丢失 —— 拼接回去的文本与原文字词完全一致（忽略空白差异）
  2. 保护区完好 —— 代码块 / 行内代码 / URL / 颜文字 / 括号对绝不会被从中间切断
  3. 段数不超上限
  4. 不产生空段

运行: python -m pytest tests/ -v
"""

from __future__ import annotations

import pathlib
import sys

import pytest

PLUGIN_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from rs_splitter import (  # noqa: E402
    PRIORITY_CJK_SPACE,
    PRIORITY_SPACE,
    SplitRules,
    cut_candidates,
    protected_intervals,
    split_reply,
)

RULES = SplitRules()

LONG_URL = "https://example.com/" + "x" * 500

CORPUS = {
    "长中文": (
        "我今天去了那个新开的咖啡店，环境还不错。点了一杯拿铁，味道一般般吧，"
        "没有之前那家好喝。对了你上次推荐的那本书我看完了，超好看！我打算周末再去一趟，"
        "你要不要一起？顺便把那本书带给你。对了还有件事，下周的会议时间改到了周三下午三点，"
        "记得把材料提前发我，我好提前过一遍内容。"
    ),
    "含URL": (
        "这个插件我看了下文档，参考实现在这里 "
        "https://github.com/saberlights/smart_segmentation_plugin "
        "你可以先看看它的架构，特别是它挂 hook 的方式。它用的是 LLM 语义分段，"
        "而且必须先关掉宿主的 response_splitter，否则会双重切分。这一点很重要，别踩坑了。"
        "另外还要注意，它的 planner hook 载荷格式变过，如果你照抄旧版本会静默失效。"
    ),
    "含代码块": (
        "给你个例子，大概长这样：```python\n"
        "async def handle(self, **kwargs):\n"
        '    return {"action": "continue"}\n'
        "``` 注意装饰器必须紧贴 def，中间不能插别的方法，否则会静默注册错人。"
        "这个坑我踩过，本地全绿但真机直接 unexpectedly 报错。"
        "而且最难的地方在于它完全静默，你只能靠真机日志反推，非常费时间。"
    ),
    "含行内代码": (
        "把 `response_splitter.enable` 改成 `false` 就行了，注意是 `config/bot_config.toml` "
        "这个文件。改完之后要完整重启 MaiBot，热重载对 manifest 和配置变更都不生效。"
        "这一点在文档里写得很清楚，但很多人还是会踩。"
    ),
    "含颜文字": (
        "哈哈哈哈草（╯°□°）╯︵ ┻━┻ 你这也太离谱了吧，我整个人都无语了，真的绷不住了，"
        "谁懂啊家人们。这个 bug 我调了一整个下午最后发现是配置文件带了 BOM，导致 TOML 解析直接报错。"
        "真的服了，下次一定要用不带 BOM 的方式写文件(╯‵□′)╯︵┻━┻"
    ),
    "英文长文": (
        "The plugin system uses msgpack encoded RPC protocol between the host and runner processes. "
        "Each plugin runs in its own subprocess, so a crash in one plugin will not take down the whole bot. "
        "That is a very important design decision for stability. You should keep it in mind when writing plugins, "
        "because it changes how you handle errors and shared state."
    ),
    "无标点中文": "这是一段完全没有标点符号的超长文本" * 20,
    "段落结构": (
        "先说结论：这个方案可行。\n\n"
        "具体来说，宿主的分割器只是配置节，我们关掉它即可，不用改主程序源码。"
        "这一点很关键，因为它决定了插件的边界。\n\n"
        "然后是实现，需要挂两个 hook，一个改参一个补发，中间用重入保护串起来，"
        "否则补发的内容会被二次切分，形成无限循环。\n\n"
        "最后是验证，必须真机跑一遍看日志。"
    ),
    "版本号与路径": (
        "请把依赖升级到 maibot-plugin-sdk 2.8.1 版本，然后参考 docs.mai-mai.org/plugin/hooks "
        "这个页面的说明去改。配置文件在 config/bot_config.toml，注意 response_splitter 那一节"
        "要先把 enable 改成 false，否则两边会同时切分导致消息被切得乱七八糟。改完记得完整重启。"
    ),
    "超长URL": f"看这个链接 {LONG_URL} 后面还有一些说明文字需要跟在后面不然会很难看，所以这里再补一点内容凑长度。",
}


def _normalized(text: str) -> str:
    return "".join(text.split())


def _assert_no_content_lost(original: str, segments: list[str]) -> None:
    assert _normalized("".join(segments)) == _normalized(original), "切分过程丢字或加字了"


def _assert_atoms_intact(original: str, segments: list[str], rules: SplitRules) -> None:
    stripped = original.strip()
    for start, end in protected_intervals(stripped, rules):
        atom = stripped[start:end]
        assert any(atom in segment for segment in segments), f"保护区被切断: {atom[:40]!r}"


def _assert_well_formed(segments: list[str], rules: SplitRules) -> None:
    assert segments, "不允许返回空列表"
    for segment in segments:
        assert segment.strip() == segment, "段首尾不应有空白"
        assert segment, "不允许产生空段"
    assert len(segments) <= rules.max_segments, f"段数 {len(segments)} 超过上限 {rules.max_segments}"


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_corpus_invariants(name: str) -> None:
    """全语料必须同时满足四个不变量。"""
    text = CORPUS[name]
    segments = split_reply(text, RULES)
    _assert_no_content_lost(text, segments)
    _assert_atoms_intact(text, segments, RULES)
    _assert_well_formed(segments, RULES)


def test_short_text_not_split() -> None:
    text = "好的，收到了，我这就去办。"
    assert split_reply(text, RULES) == [text]


def test_empty_input() -> None:
    assert split_reply("", RULES) == []
    assert split_reply("   \n  ", RULES) == []


def test_text_just_below_min_length_not_split() -> None:
    rules = SplitRules(min_length=20)
    text = "这是一句不够长的话。"
    assert len(text) < rules.min_length
    assert split_reply(text, rules) == [text]


def test_长文本确实被切分() -> None:
    """防止「永远只返回一段」这类静默退化。"""
    text = "这是一句需要被切开的测试内容。" * 20
    segments = split_reply(text, RULES)
    assert len(segments) > 1, "长文本没有被切分，规则可能已失效"
    _assert_no_content_lost(text, segments)


def test_ascii句点被当作句末切点() -> None:
    """纯英文回复必须能找到「句末标点级」切点，而不是退而求其次切逗号。"""
    text = (
        "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu. "
        "Nu xi omicron pi rho sigma tau upsilon phi chi psi omega, then more words follow here. "
        "Another sentence appears at the very end of this paragraph to push the length over."
    )
    rules = SplitRules(soft_max_length=120, hard_max_length=300, max_segments=4, min_length=10)
    segments = split_reply(text, rules)
    assert len(segments) > 1
    # 首段应切在句末标点后，而不是逗号后
    assert segments[0].endswith("."), f"首段未切在句末标点: {segments[0]!r}"


def test_省略号不被切开() -> None:
    """省略号必须整对落在同一段里，不能被从中间分开。"""
    text = "我还在想这件事……" + "然后呢然后呢然后呢" * 12 + "最后还是算了。"
    rules = SplitRules(soft_max_length=40, hard_max_length=120, max_segments=8, min_length=10)
    segments = split_reply(text, rules)
    assert any("……" in segment for segment in segments), "省略号被切成了两半"


def test_超长URL完整保留() -> None:
    """单个原子比硬上限还长时，宁可让这一段超长，也不把链接切断。"""
    text = f"看这个链接 {LONG_URL} 后面跟着说明。"
    segments = split_reply(text, RULES)
    assert any(LONG_URL in segment for segment in segments), "超长 URL 被切断了"
    _assert_no_content_lost(text, segments)


def test_代码块完整保留() -> None:
    text = "前置说明文字。" * 6 + "```\n" + "code line\n" * 20 + "```" + "后置说明文字。" * 6
    rules = SplitRules(soft_max_length=100, hard_max_length=260, max_segments=8, min_length=10)
    segments = split_reply(text, rules)
    for start, end in protected_intervals(text.strip(), rules):
        atom = text.strip()[start:end]
        assert any(atom in segment for segment in segments), "代码块被切断"


def test_关掉保护时会允许切开() -> None:
    """保护开关必须真的起作用，否则配置项是摆设。"""
    text = f"看这个链接 {LONG_URL} 后面跟着说明。"
    rules = SplitRules(protect_url=False)
    unprotected = protected_intervals(text, rules)
    assert not unprotected, "关闭 protect_url 后仍产生了 URL 保护区"


def test_段数上限被遵守() -> None:
    text = "这是一句测试内容。" * 200
    rules = SplitRules(max_segments=3)
    segments = split_reply(text, rules)
    assert len(segments) == 3, f"期望恰好 3 段，实际 {len(segments)}"
    _assert_no_content_lost(text, segments)


def test_极短上限不会死循环() -> None:
    """回归：soft/hard 配置被设成极小值时不能挂死。"""
    text = "这是一句测试内容，用来验证极小上限下不会死循环。" * 10
    rules = SplitRules(
        min_length=1,
        soft_max_length=1,
        hard_max_length=1,
        max_segments=100,
        min_segment_length=0,
    )
    segments = split_reply(text, rules)
    _assert_no_content_lost(text, segments)
    assert len(segments) > 1


def test_无标点中文仍可切分() -> None:
    """回归：中文不能被当成「单词内部」字符，否则零切点。"""
    text = "这是一段完全没有标点符号的超长文本" * 20
    stripped = text
    assert cut_candidates(stripped, protected_intervals(stripped, RULES)), "无标点中文没有任何切点"
    assert len(split_reply(text, RULES)) > 1


def test_短尾段被并回上一段() -> None:
    """回归：不允许甩出「。」这种孤立的尾巴。"""
    text = "有一句内容。再来一句内容。最后再补上一句内容。" * 8
    rules = SplitRules(soft_max_length=90, hard_max_length=200, max_segments=8, min_length=10)
    segments = split_reply(text, rules)
    for segment in segments:
        assert len(segment) >= rules.min_segment_length or len(segments) == 1, (
            f"出现过短的孤立段: {segment!r}"
        )


# ── 强制分条边界（换行 = 模型表达「这条单独发」）──────────────────────────

NL = "\n"


def test_换行被无条件尊重_不受长度预算约束() -> None:
    """回归：换行是模型表达分条意图的唯一手段，不能被「装得下就收尾」吞掉。

    短文本（远小于 soft_max）也必须按换行分开。
    """
    text = "今天主要整理一下曲库，顺便看看有没有新歌可以放进轮播里" + NL + "晚点可能试播一小段"
    assert len(text) < SplitRules().soft_max_length, "样本应短于目标长度，才能验证不被吞掉"
    segments = split_reply(text, SplitRules())
    assert segments == [
        "今天主要整理一下曲库，顺便看看有没有新歌可以放进轮播里",
        "晚点可能试播一小段",
    ], f"换行未被尊重: {segments}"


def test_极短两行也分条_min_length不拦() -> None:
    """模型只发两行短句时，「太短就不切」这条保护不应生效。"""
    segments = split_reply("整理曲库" + NL + "晚点试播？", SplitRules(min_length=60))
    assert segments == ["整理曲库", "晚点试播？"], f"极短两行未分条: {segments}"


def test_代码块内的换行不强制分条() -> None:
    """回归（真实踩到）：代码块本身就是多行的。

    若把块内换行当成分条边界，代码块会被从中切开 —— 保护区形同虚设。
    优先级必须是：保护区 > 强制分条边界。
    """
    text = "示例：```python" + NL + "a = 1" + NL + "b = 2" + NL + "``` 就这样。"
    rules = SplitRules(min_length=1, max_segments=10)
    segments = split_reply(text, rules)
    assert len(segments) == 1, f"代码块被换行强制切开了: {segments}"
    assert "a = 1" + NL + "b = 2" in segments[0], "代码块内容不完整"


def test_列表项换行不强制分条() -> None:
    """列表项之间的换行属于排版，不等于分条意图。"""
    text = "几个要点：" + NL + "- 第一条" + NL + "- 第二条" + NL + "- 第三条"
    segments = split_reply(text, SplitRules(min_length=1, max_segments=10))
    assert len(segments) == 1, f"列表被按行拆碎了: {segments}"


def test_段落空行强制分条() -> None:
    segments = split_reply("先说结论。" + NL + NL + "具体来说是这样。", SplitRules(min_length=1))
    assert len(segments) == 2, f"段落未被分开: {segments}"


def test_关掉强制换行则回到旧行为() -> None:
    text = "上一句" + NL + "下一句"
    rules = SplitRules(force_split_on_newline=False)
    assert split_reply(text, rules) == [text], "未配置时应保持原样单段"


def test_压缩目标长度会额外切碎_因此不该靠它做极短() -> None:
    """守住一个设计结论：极短效果应靠模型换行，不靠把 soft_max 压小。

    soft_max 压到 20 会在逗号处多切一刀，那正是内置分割器「机械按长度切」
    的老毛病。这条用例把差异固定下来，防止以后有人用「调小 soft_max」来实现极短。
    """
    text = "今天主要整理一下曲库，顺便看看有没有新歌可以放进轮播里" + NL + "晚点可能试播一小段"
    narrow = SplitRules(
        min_length=1, soft_max_length=20, hard_max_length=60,
        max_segments=10, min_segment_length=0,
    )
    wide = SplitRules(min_length=1, soft_max_length=90, max_segments=10, min_segment_length=0)
    assert len(split_reply(text, narrow)) == 3, "压缩目标长度应多切一刀（这是要避免的行为）"
    assert len(split_reply(text, wide)) == 2, "推荐档应在换行处切成两段"
    _assert_no_content_lost(text, split_reply(text, wide))
    _assert_atoms_intact(text, split_reply(text, wide), wide)


# ── 空格分级：汉字之间的空格是强边界 ────────────────────────────────────────

# 真机实测的回复原文（181 字，无任何标点，仅空格分隔）。
# 用作回归锚点：任何改动都不应改变它的切分结果。
REAL_REPLY = (
    "好 那再测一次 今天下午有点犯困 泡了杯茶继续整理曲库 翻到几首很适合秋天的歌 "
    "前奏一响就舍不得跳过了 电台的轮播列表也快排好了 等会儿试播的时候你们可以听听看 "
    "昨天有人说想听温柔一点的 我就多放了这几首慢歌 傍晚的风从窗户吹进来 和旋律缠在一起 "
    "那种感觉挺好的 有时候觉得歌和人一样 都有自己的频率 对上了就会在心里留很久 "
    "好啦这段应该够长了 分割器工作正常吗"
)


def test_空格分级_汉字间为强边界其余为弱边界() -> None:
    """中文排版里汉字之间不该有空格 —— 出现了就说明是模型刻意打的停顿。"""
    text = "汉字 汉字 abc def 汉字"
    priorities = dict(cut_candidates(text, protected_intervals(text, SplitRules())))
    assert priorities[3] == PRIORITY_CJK_SPACE, "汉字之间的空格应为强边界"
    assert priorities[6] == PRIORITY_SPACE, "汉字与拉丁之间的空格应为弱边界"
    assert priorities[14] == PRIORITY_SPACE, "拉丁与汉字之间的空格应为弱边界"


def test_连续空格也识别为汉字间强边界() -> None:
    text = "前面  后面"
    priorities = dict(cut_candidates(text, protected_intervals(text, SplitRules())))
    assert any(value == PRIORITY_CJK_SPACE for value in priorities.values()), (
        f"连续空格未被识别为强边界: {priorities}"
    )


def test_同一窗口内优先汉字间空格而非英文间空格() -> None:
    """窗口内两类空格竞争时，应选汉字间的那个 —— 英文短语就不会被拆开。"""
    text = (
        "我今天去了那家新开的咖啡店 the coffee shop nearby is quite nice "
        "环境还不错 味道也还行 价格也不贵"
    )
    rules = SplitRules(min_length=1, soft_max_length=62, max_segments=9, min_segment_length=0)
    segments = split_reply(text, rules)
    assert len(segments) > 1
    # 切在「环境还不错」之后的汉字间空格，而不是更靠前的英文间空格
    assert segments[1].startswith("味道也还行"), f"未优先汉字间空格: {segments}"


def test_真机回复原文切分结果锁死() -> None:
    """回归锚点：真机软上限 90 下的 [80, 81, 18] 必须逐字保持。"""
    segments = split_reply(
        REAL_REPLY,
        SplitRules(min_length=1, soft_max_length=90, max_segments=10, min_segment_length=0),
    )
    assert [len(segment) for segment in segments] == [80, 81, 18], (
        f"切分结果发生偏移: {[len(s) for s in segments]}"
    )
    assert segments[0].endswith("你们可以听听看"), "首段末尾变了"
    assert segments[2] == "好啦这段应该够长了 分割器工作正常吗", "尾段内容变了"
    _assert_no_content_lost(REAL_REPLY, segments)
