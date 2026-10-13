"""分析器测试：报告结构、坏案例清单、启发式举牌、对照组可复现性。

运行: python -m pytest tests/test_analyzer.py -v
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

PLUGIN_DIR = pathlib.Path(__file__).resolve().parent.parent
TESTS_DIR = pathlib.Path(__file__).resolve().parent
ANALYZER_DIR = PLUGIN_DIR / "analyzer"
for _path in (str(ANALYZER_DIR),):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import analyze  # noqa: E402

SPLIT_EVENT = {
    "ts": 1791900000.0,
    "event": "split",
    "rid": "aaaa1111",
    "stream": "s-1",
    "text": "今天主要整理一下曲库，晚点试播？\n先把这些记下来。",
    "segments": ["今天主要整理一下曲库，晚点试播？", "先把这些记下来。"],
    "located": [
        {"len": 15, "boundary": "start", "hard_cut": False, "merged_newlines": 0},
        {"len": 8, "boundary": "newline", "hard_cut": False, "merged_newlines": 0},
    ],
    "rules": {"min_length": 25, "soft_max_length": 90, "hard_max_length": 480,
              "max_segments": 10, "min_segment_length": 8},
    "inplace_ok": True,
}

VERIFY_FAILED = {
    "ts": 1791900001.0,
    "event": "verify",
    "rid": "bbbb2222",
    "stream": "s-2",
    "verdict": "failed",
    "sample_len": 360,
}

HARD_CUT_SPLIT = {
    "ts": 1791900002.0,
    "event": "split",
    "rid": "cccc3333",
    "stream": "s-3",
    "text": "今天天气不错我们出去玩吧去公园走走看看花",
    "segments": ["今天天气不错我们出去玩", "吧去公园走走看看花"],
    "located": [
        {"len": 12, "boundary": "start", "hard_cut": False, "merged_newlines": 0},
        {"len": 9, "boundary": "hard", "hard_cut": True, "merged_newlines": 0},
    ],
    "rules": {"min_length": 1, "soft_max_length": 8, "hard_max_length": 12,
              "max_segments": 6, "min_segment_length": 0},
    "inplace_ok": True,
}


def _write_jsonl(tmp_path: pathlib.Path, events: list[dict]) -> pathlib.Path:
    collect = tmp_path / "collect"
    collect.mkdir()
    with (collect / "2026-10-13.jsonl").open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    return collect


def test_build_report_contains_all_sections(tmp_path):
    collect = _write_jsonl(tmp_path, [SPLIT_EVENT, VERIFY_FAILED, HARD_CUT_SPLIT])
    events, files, bad_lines = analyze.load_events([str(collect)])
    assert not bad_lines and len(files) == 1
    report = analyze.build_report(events, files, bad_lines, control_n=5, seed=42, max_text=100)

    for heading in ("## 1. 概览", "## 2. 切分统计", "## 3. 硬信号",
                    "## 4. 启发式举牌", "## 5. 随机对照组"):
        assert heading in report, f"报告缺章节: {heading}"
    assert "rid=bbbb2222" in report and "实发校验 failed" in report, "硬信号缺实发校验失败"
    assert "rid=cccc3333" in report and "含硬切切点" in report, "启发式未举牌硬切"


def test_load_events_skips_bad_lines(tmp_path):
    collect = _write_jsonl(tmp_path, [SPLIT_EVENT])
    with (collect / "2026-10-13.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("{broken json\n")
    events, _files, bad_lines = analyze.load_events([str(collect)])
    assert bad_lines == 1 and len(events) == 1


def test_cli_writes_report_file(tmp_path):
    collect = _write_jsonl(tmp_path, [SPLIT_EVENT, HARD_CUT_SPLIT])
    out = tmp_path / "report.md"
    result = subprocess.run(
        [sys.executable, str(ANALYZER_DIR / "analyze.py"), str(collect), "-o", str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    report = out.read_text(encoding="utf-8")
    assert report.startswith("# reply-splitter 采集数据分析报告")
    assert "段数分布" in report and "切点边界分布" in report


def test_control_group_reproducible(tmp_path):
    collect = _write_jsonl(
        tmp_path,
        [SPLIT_EVENT] + [
            {**SPLIT_EVENT, "rid": f"dup{i:04d}", "stream": f"s-{i}"}
            for i in range(20)
        ],
    )
    events, files, bad = analyze.load_events([str(collect)])
    report_a = analyze.build_report(events, files, bad, control_n=5, seed=42, max_text=100)
    report_b = analyze.build_report(events, files, bad, control_n=5, seed=42, max_text=100)
    report_c = analyze.build_report(events, files, bad, control_n=5, seed=7, max_text=100)
    assert report_a == report_b, "同 seed 结果应可复现"
    assert report_a != report_c, "不同 seed 应抽出不同对照组"
