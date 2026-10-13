"""采集埋点：把切分前原文、切分结果与成败事件以 JSONL 落盘，供离线分析。

设计红线（与插件主链路的关系）：
  - 采集是**旁路**：任何写盘异常都被吞掉只计数，绝不影响切分与发送；
  - 默认关闭（[collect] enabled = false），只在调试周期手动开启；
  - 数据是 bot 自己产出的回复，原文落盘不出本机。

文件形态：<落盘目录>/YYYY-MM-DD.jsonl，按天分文件，不自动清理——
攒够样本后手动处理（决策记录 D5/D8，docs/decisions-reply-collector.md）。

除落盘外，本模块提供 locate_segments 纯函数：从原文反推每段切点的
边界类型。rs_splitter.split_reply 只返回段列表、不带元数据（保持其
纯函数接口不被分析逻辑污染），切点归类由这一层补齐。
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .rs_splitter import COMMA_ENDERS, SENTENCE_ENDERS, SOFT_ENDERS

__all__ = ["Collector", "locate_segments"]

logger = logging.getLogger("reply-splitter.collect")

_IS_HAN_RANGES = ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF))


def _is_han(ch: str) -> bool:
    if not ch:
        return False
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _IS_HAN_RANGES)


def _skip_ws(text: str, cursor: int) -> int:
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    return cursor


def _classify_boundary(preceding: str, skipped: str, next_ch: str) -> str:
    """按「紧邻前字符 + 起点前空白 run」给段起点归类。

    preceding: 原文里该段起点之前、跳过空白后的最后一个字符；
    skipped:   起点前被 strip 掉的空白 run；
    next_ch:   该段第一个字符。
    """
    if not preceding and not skipped:
        return "start"
    if "\n" in skipped:
        # 连续两个以上换行 = 段落；单换行 = 模型主动分条
        return "paragraph" if "\n\n" in skipped else "newline"
    if preceding in SENTENCE_ENDERS:
        return "sentence"
    if preceding in SOFT_ENDERS:
        return "soft"
    if preceding in COMMA_ENDERS:
        return "comma"
    if skipped:
        if _is_han(preceding) and _is_han(next_ch):
            return "cjk_space"
        return "space"
    # 落刀处既无标点也无空白 → 硬切（所有干净边界都不可用时的兜底刀）
    return "hard"


def _match_segment(text: str, pos: int, segment: str) -> tuple[int, str, int] | None:
    """在 text[pos:] 处匹配 segment，返回 (段起点, 起点前空白run, 段内并入的换行数)。

    匹配规则（与 rs_splitter 的产出方式一一对应）：
      - 段内容是 text 的子串（两端 strip 过），起点前可能有被 strip 的空白；
      - 合并段可能带有合成空格分隔符（_join_separator）或直接拼接
        （_absorb_short_segments），对应原文位置本就有空白——匹配时跳过；
      - 段内空白（含代码块换行）必须与原文逐字对应，绝不跳过。
    匹配失败返回 None（调用方标记 unknown 并尝试重新同步）。
    """
    start = _skip_ws(text, pos)
    skipped = text[pos:start]
    i, j = start, 0
    merged_newlines = 0
    while j < len(segment):
        if i < len(text) and text[i] == segment[j]:
            i += 1
            j += 1
            continue
        if i < len(text) and text[i].isspace():
            # 段边界处被 strip 掉的空白（短段并回时不补分隔符，直接跳过原文空白）
            run_start = i
            i = _skip_ws(text, i)
            merged_newlines += text[run_start:i].count("\n")
            continue
        if segment[j] == " ":
            # 合成分隔符：对应原文的空白（被 strip 掉后由 _join_separator 还原成空格）
            if i < len(text) and text[i].isspace():
                run_start = i
                i = _skip_ws(text, i)
                merged_newlines += text[run_start:i].count("\n")
            j += 1
            continue
        return None
    return start, skipped, merged_newlines


def _char_before(text: str, start: int) -> str:
    """段起点前、跳过空白后的最后一个字符。"""
    cursor = start - 1
    while cursor >= 0 and text[cursor].isspace():
        cursor -= 1
    return text[cursor] if cursor >= 0 else ""


def _find_segment_start(text: str, pos: int, segment: str) -> int:
    """直接匹配失败时，从 pos 向后搜索段首 3 个字符的落点；找不到返回 -1。"""
    probe = segment.lstrip()[:3]
    if not probe:
        return -1
    return text.find(probe, pos)


def locate_segments(text: str, segments: list[str]) -> list[dict[str, Any]]:
    """从原文反推每段的切点元数据（纯函数，供 split 事件与离线分析共用）。

    返回与 segments 等长的列表，每项：
      len             段长度（字符数）
      boundary        切点边界类型：start/paragraph/newline/sentence/
                      soft/comma/cjk_space/space/hard/unknown
      hard_cut        是否硬切（boundary == "hard"）
      merged_newlines 段内并入的换行强制边界数（>0 = 触发了「合并相邻行」）
    """
    results: list[dict[str, Any]] = []
    pos = 0
    lost = False  # 一旦彻底失配，后续段只能标 unknown

    for index, segment in enumerate(segments):
        entry: dict[str, Any] = {
            "len": len(segment),
            "boundary": "unknown",
            "hard_cut": False,
            "merged_newlines": 0,
        }
        if not lost:
            matched = _match_segment(text, pos, segment)
            if matched is None:
                probe = _find_segment_start(text, pos, segment)
                if probe > pos:
                    matched = _match_segment(text, probe, segment)
            if matched is None:
                lost = True
            else:
                start, skipped, merged_newlines = matched
                if index == 0:
                    boundary = "start"
                else:
                    boundary = _classify_boundary(
                        _char_before(text, start), skipped, segment[:1]
                    )
                entry["boundary"] = boundary
                entry["hard_cut"] = boundary == "hard"
                entry["merged_newlines"] = merged_newlines
                pos = start + len(segment)
        results.append(entry)
    return results


class Collector:
    """JSONL 旁路落盘器。任何异常都不外抛——采集绝不影响切分主链路。"""

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.failed_writes = 0

    def write(self, event: dict[str, Any]) -> None:
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            path = self.base_dir / (datetime.now().strftime("%Y-%m-%d") + ".jsonl")
            record = {"ts": round(time.time(), 3), **event}
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001 —— 采集是旁路，失败只计数不外抛
            self.failed_writes += 1
            if self.failed_writes == 1 or self.failed_writes % 100 == 0:
                logger.warning(
                    "采集落盘失败（已累计 %s 次）dir=%s", self.failed_writes, self.base_dir
                )
