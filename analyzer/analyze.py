#!/usr/bin/env python3
"""reply-splitter 采集数据离线分析器。

读插件 [collect] 埋点落盘的 JSONL（<插件目录>/collect/YYYY-MM-DD.jsonl），
产出 markdown 报告：整体统计 + 坏案例清单（硬信号 + 启发式举牌 + 随机对照组）。
程序只「举牌」不下结论——判「这条切得好不好」由人来读（决策记录 D11）。

纯标准库，零第三方依赖：

    python analyzer/analyze.py <collect 目录或 jsonl 文件...> [-o report.md]
                              [--control 10] [--seed 42] [--max-text 100]

报告结构：
  1. 概览 —— 事件计数、切分率、时间范围
  2. 切分统计 —— 段数分布、段长分布、切点边界分布、配置快照
  3. 硬信号（确定性坏案例）—— 实发校验失败 / 补发失败 / 异常 / 放弃切分
  4. 启发式举牌 —— 硬切、合并相邻行、段数打满、段超硬上限、极值样本
  5. 随机对照组 —— 普通样本抽样，防止只见特例不见常态
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = ["load_events", "build_report"]

# ── 显示辅助 ────────────────────────────────────────────────────────────────

BOUNDARY_LABELS = {
    "start": "开头",
    "paragraph": "段落(空行)",
    "newline": "换行",
    "sentence": "句末标点",
    "soft": "分号/冒号",
    "comma": "逗号/顿号",
    "cjk_space": "汉字间空格",
    "space": "普通空格",
    "hard": "硬切",
    "unknown": "未知",
}

_SKIP_LABELS = {
    "resending": "补发中重入保护",
    "command_protection": "命令回执保护",
    "pending_exists": "上条分段未补发完",
    "not_armed": "未标记回复来源",
    "no_text": "载荷无正文",
}

_HEURISTIC_CAP = 40  # 举牌清单最多渲染多少条，防止报告失控


def _label_boundary(name: str) -> str:
    return BOUNDARY_LABELS.get(name, name)


def _excerpt(text: str, limit: int) -> str:
    """压缩成单行摘录：换行显式化、反引号替换掉（防破坏 markdown 行内码）。"""
    flat = str(text or "").replace("`", "´").replace("\r", "")
    flat = "\n".join(flat.split("\n"))  # 保留换行为字面 ⏎ 标记
    flat = flat.replace("\n", "⏎")
    if len(flat) > limit:
        return flat[:limit] + f"…(+{len(flat) - limit}字)"
    return flat


def _fmt_ts(ts: float) -> str:
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%m-%d %H:%M:%S")
    except (TypeError, ValueError, OSError, OverflowError):
        return "?"


def _pctl(values: list[int], ratio: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(ratio * (len(ordered) - 1))))
    return ordered[index]


# ── 事件加载 ────────────────────────────────────────────────────────────────


def load_events(paths: list[str]) -> tuple[list[dict[str, Any]], list[str], int]:
    """读入全部 JSONL 事件。返回 (事件列表, 来源文件列表, 坏行数)。"""
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.glob("*.jsonl")))
        elif path.is_file():
            files.append(path)
        else:
            print(f"警告：路径不存在，跳过 {raw}", file=sys.stderr)
    events: list[dict[str, Any]] = []
    bad_lines = 0
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                bad_lines += 1
    return events, [str(path) for path in files], bad_lines


def _by_rid(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        rid = str(event.get("rid") or "")
        if rid:
            grouped.setdefault(rid, []).append(event)
    return grouped


# ── 样本渲染 ────────────────────────────────────────────────────────────────


def split_flags(event: dict[str, Any]) -> list[str]:
    """启发式举牌（D11）：程序只给客观信号，不下好坏结论。"""
    flags: list[str] = []
    located = event.get("located") or []
    segments = event.get("segments") or []
    rules = event.get("rules") or {}

    cuts = [item for item in located if item.get("boundary") not in ("start", "unknown")]
    hard = [item for item in cuts if item.get("hard_cut")]
    if hard:
        flags.append(f"含硬切切点×{len(hard)}")
    merged = [item for item in located if int(item.get("merged_newlines") or 0) > 0]
    if merged:
        total = sum(int(item["merged_newlines"]) for item in merged)
        flags.append(f"合并相邻行×{total}")
    max_segments = int(rules.get("max_segments") or 0)
    if max_segments and len(segments) >= max_segments:
        flags.append(f"段数打满上限({max_segments})")
    hard_max = int(rules.get("hard_max_length") or 0)
    if hard_max:
        over = [len(seg) for seg in segments if len(seg) > hard_max]
        if over:
            flags.append(f"段超硬上限{max(over)}字(>{hard_max})")
    return flags


def _render_sample(event: dict[str, Any], flags: list[str], max_text: int) -> list[str]:
    rid = str(event.get("rid") or "?")
    segments = event.get("segments") or []
    located = event.get("located") or []
    boundaries = " → ".join(_label_boundary(str(item.get("boundary"))) for item in located)
    lines = [
        f"#### rid={rid}  {_fmt_ts(event.get('ts', 0))}  stream={event.get('stream', '?')}",
        "",
        f"- 原文（{len(event.get('text') or '')}字）：`{_excerpt(event.get('text') or '', max_text)}`",
        f"- 切成 {len(segments)} 段 "
        f"[{', '.join(str(len(seg)) for seg in segments)}]，边界：{boundaries or '未知'}",
        f"- 分段预览：`{_excerpt(' ｜ '.join(segments), max_text)}`",
    ]
    if flags:
        lines.append(f"- 触发：{'；'.join(flags)}")
    lines.append("")
    return lines


def _no_split_reason(event: dict[str, Any]) -> str:
    rules = event.get("rules") or {}
    text = str(event.get("text") or "")
    min_length = int(rules.get("min_length") or 0)
    if min_length and len(text.strip()) < min_length:
        return f"短于 min_length({min_length})"
    return "单段输出 / 无可用切点"


# ── 报告生成 ────────────────────────────────────────────────────────────────


def build_report(
    events: list[dict[str, Any]],
    source_files: list[str],
    bad_lines: int,
    control_n: int,
    seed: int,
    max_text: int,
) -> str:
    out: list[str] = []
    say = out.append

    splits = [event for event in events if event.get("event") == "split"]
    no_splits = [event for event in events if event.get("event") == "no_split"]
    verifies = [event for event in events if event.get("event") == "verify"]
    resends = [event for event in events if event.get("event") == "resend"]
    expires = [event for event in events if event.get("event") == "expire"]
    aborted = [event for event in events if event.get("event") == "aborted"]
    exceptions = [event for event in events if event.get("event") == "exception"]
    skips = [event for event in events if event.get("event") == "skip"]

    rid_index = _by_rid(events)

    say("# reply-splitter 采集数据分析报告")
    say("")
    if source_files:
        say(f"- 数据来源：{len(source_files)} 个文件（{', '.join(source_files[:3])}"
            f"{'…' if len(source_files) > 3 else ''}）")
    timestamps = [float(event["ts"]) for event in events if event.get("ts")]
    if timestamps:
        say(f"- 时间范围：{_fmt_ts(min(timestamps))} ~ {_fmt_ts(max(timestamps))}")
    if bad_lines:
        say(f"- ⚠ 无法解析的行：{bad_lines}（可能跨版本 schema）")
    say("")

    # ── 1. 概览 ──
    say("## 1. 概览")
    say("")
    say("| 事件 | 条数 | 说明 |")
    say("|---|---:|---|")
    say(f"| split | {len(splits)} | 切分成功（首段改写 + 其余待补发） |")
    say(f"| no_split | {len(no_splits)} | 候选回复但未切分 |")
    say(f"| verify | {len(verifies)} | 实发校验结果 |")
    say(f"| resend | {len(resends)} | 补发任务结果 |")
    say(f"| expire | {len(expires)} | 待补发登记过期 |")
    say(f"| aborted | {len(aborted)} | 放弃切分（载荷定位失败） |")
    say(f"| exception | {len(exceptions)} | 切分规则引擎异常 |")
    say(f"| skip | {len(skips)} | 结构性跳过 |")
    say("")
    candidates = len(splits) + len(no_splits)
    if candidates:
        rate = len(splits) / candidates
        say(f"切分率：**{len(splits)}/{candidates} = {rate:.0%}**（候选 = split + no_split）")
        say("")
    if skips:
        reasons = Counter(str(event.get("reason")) for event in skips)
        say("跳过原因分布：" + "，".join(
            f"{_SKIP_LABELS.get(reason, reason)}×{count}" for reason, count in reasons.most_common()
        ))
        say("")

    # ── 2. 切分统计 ──
    say("## 2. 切分统计")
    say("")
    if splits:
        seg_count = Counter(len(event.get("segments") or []) for event in splits)
        say(f"- 段数分布：{'，'.join(f'{count}段×{num}' for count, num in sorted(seg_count.items()))}")
        lengths = [len(seg) for event in splits for seg in (event.get("segments") or [])]
        if lengths:
            say(
                f"- 段长分布：min={min(lengths)} p50={_pctl(lengths, 0.5)} "
                f"p90={_pctl(lengths, 0.9)} max={max(lengths)}（共 {len(lengths)} 段）"
            )
        boundaries = Counter(
            str(item.get("boundary"))
            for event in splits
            for item in (event.get("located") or [])
            if item.get("boundary") not in ("start",)
        )
        say("- 切点边界分布：" + "，".join(
            f"{_label_boundary(name)}×{count}" for name, count in boundaries.most_common()
        ))
        merged_n = sum(1 for event in splits if any(
            int(item.get("merged_newlines") or 0) > 0 for item in (event.get("located") or [])
        ))
        hard_n = sum(1 for event in splits if any(
            item.get("hard_cut") for item in (event.get("located") or [])
        ))
        say(f"- 合并相邻行：{merged_n}/{len(splits)}；含硬切：{hard_n}/{len(splits)}")
        snapshots = Counter(
            "min={} soft={} hard={} max_seg={}".format(
                (event.get("rules") or {}).get("min_length"),
                (event.get("rules") or {}).get("soft_max_length"),
                (event.get("rules") or {}).get("hard_max_length"),
                (event.get("rules") or {}).get("max_segments"),
            )
            for event in splits
        )
        say("- 配置快照：" + "；".join(f"`{name}`×{count}" for name, count in snapshots.most_common()))
        if verifies:
            verdicts = Counter(str(event.get("verdict")) for event in verifies)
            say("- 实发校验：" + "，".join(f"{verdict}×{count}" for verdict, count in verdicts.most_common()))
        if resends:
            resend_ok = sum(1 for event in resends if not event.get("failed") and not event.get("cancelled"))
            resend_bad = len(resends) - resend_ok
            say(f"- 补发结果：成功×{resend_ok}，异常×{resend_bad}")
        if no_splits:
            reason_count = Counter(_no_split_reason(event) for event in no_splits)
            say("- 未切分原因：" + "，".join(
                f"{reason}×{count}" for reason, count in reason_count.most_common()
            ))
        say("")
    else:
        say("（无 split 样本）")
        say("")

    # ── 3. 硬信号 ──
    say("## 3. 硬信号（确定性坏案例）")
    say("")
    hard_cases: list[str] = []
    for event in verifies:
        verdict = str(event.get("verdict"))
        if verdict != "ok":
            origin = (rid_index.get(str(event.get("rid"))) or [{}])[0]
            text = origin.get("text") if isinstance(origin, dict) else ""
            hard_cases.append(
                f"- **实发校验 {verdict}** rid={event.get('rid')} stream={event.get('stream')} "
                f"{_fmt_ts(event.get('ts', 0))}｜原文摘录：`{_excerpt(text or '', max_text)}`"
            )
    for event in resends:
        failed = int(event.get("failed") or 0)
        if failed or event.get("cancelled") or event.get("error"):
            detail = f"失败 {failed}/{event.get('total')} 段"
            if event.get("cancelled"):
                detail = "被取消"
            if event.get("error"):
                detail += f"（{event.get('error')}）"
            hard_cases.append(
                f"- **补发异常（{detail}）** rid={event.get('rid')} stream={event.get('stream')} "
                f"{_fmt_ts(event.get('ts', 0))}"
            )
    for event in exceptions:
        hard_cases.append(
            f"- **规则引擎异常** rid={event.get('rid')} stream={event.get('stream')} "
            f"{_fmt_ts(event.get('ts', 0))}｜{event.get('error')}｜原文：`{_excerpt(event.get('text') or '', max_text)}`"
        )
    for event in aborted:
        hard_cases.append(
            f"- **放弃切分（{event.get('reason')}）** rid={event.get('rid')} stream={event.get('stream')} "
            f"{_fmt_ts(event.get('ts', 0))}｜原文：`{_excerpt(event.get('text') or '', max_text)}`"
        )
    for event in expires:
        hard_cases.append(
            f"- **待补发过期** rid={event.get('rid')} stream={event.get('stream')} "
            f"{_fmt_ts(event.get('ts', 0))}"
        )
    if hard_cases:
        out.extend(hard_cases)
    else:
        say("（无——校验、补发、规则引擎全部正常）")
    say("")

    # ── 4. 启发式举牌 ──
    say("## 4. 启发式举牌（观感问题高发区，待人工判读）")
    say("")
    flagged: list[tuple[dict[str, Any], list[str]]] = []
    for event in splits:
        flags = split_flags(event)
        if flags:
            flagged.append((event, flags))
    # 极值样本：最长的 3 条原文（无论是否已被举牌）
    if splits:
        ranked = sorted(splits, key=lambda event: len(str(event.get("text") or "")), reverse=True)
        for rank, event in enumerate(ranked[:3], start=1):
            flags = next((flags for target, flags in flagged if target is event), [])
            flags = flags + [f"样本极长(第{rank}名,{len(event.get('text') or '')}字)"]
            flagged = [(target, flags if target is event else old) for target, old in flagged]
            if not any(target is event for target, _ in flagged):
                flagged.append((event, flags))
    if flagged:
        say(f"共 {len(flagged)} 条被举牌" + (
            f"，仅渲染前 {_HEURISTIC_CAP} 条" if len(flagged) > _HEURISTIC_CAP else ""
        ) + "。")
        say("")
        for event, flags in flagged[:_HEURISTIC_CAP]:
            out.extend(_render_sample(event, flags, max_text))
    else:
        say("（无）")
    say("")

    # ── 5. 随机对照组 ──
    say("## 5. 随机对照组（普通样本）")
    say("")
    flagged_ids = {str(event.get("rid")) for event, _ in flagged}
    normal = [event for event in splits if str(event.get("rid")) not in flagged_ids]
    if normal and control_n > 0:
        sample = random.Random(seed).sample(normal, min(control_n, len(normal)))
        say(f"从 {len(normal)} 条未举牌样本中随机抽 {len(sample)} 条（seed={seed}，可复现）。")
        say("")
        for event in sample:
            out.extend(_render_sample(event, [], max_text))
    else:
        say("（无普通样本可抽）")
    say("")

    say("---")
    say(f"生成时间 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        f"；分析器只举牌不下结论，判读结果请回填 docs/decisions-reply-collector.md")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="分析 reply-splitter 采集数据，产出 markdown 报告"
    )
    parser.add_argument("paths", nargs="+", help="collect 目录或 .jsonl 文件（可多个）")
    parser.add_argument("-o", "--output", default="", help="输出 markdown 文件路径（缺省打印到终端）")
    parser.add_argument("--control", type=int, default=10, help="随机对照组样本数（默认 10）")
    parser.add_argument("--seed", type=int, default=42, help="对照组抽样种子（默认 42，可复现）")
    parser.add_argument("--max-text", type=int, default=100, help="原文摘录最大长度（默认 100 字）")
    args = parser.parse_args(argv)

    events, files, bad_lines = load_events(args.paths)
    if not events:
        print("没有读到任何事件。确认路径指向 [collect] 落盘的 collect/ 目录。", file=sys.stderr)
        return 1
    report = build_report(events, files, bad_lines, args.control, args.seed, args.max_text)
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"报告已写入 {args.output}")
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
