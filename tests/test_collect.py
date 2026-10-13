"""采集埋点测试：落盘格式、开关零写入、写失败不外抛、locate_segments 反推正确性。

对应决策记录 D3（采集范围 A+B+C）/ D5（插件目录 collect/，按天分文件）/
D6（原文落盘）/ D8（默认关闭），docs/decisions-reply-collector.md。

运行: python -m pytest tests/test_collect.py -v
"""

from __future__ import annotations

import asyncio
import importlib
import json
import pathlib
import sys
import time

import pytest

PLUGIN_DIR = pathlib.Path(__file__).resolve().parent.parent
TESTS_DIR = pathlib.Path(__file__).resolve().parent
for _path in (str(PLUGIN_DIR), str(TESTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from fakehost import (  # noqa: E402
    FakeHost,
    bind_context,
    build_context,
    get_default_config,
    load_plugin_module,
)

LONG_TEXT = "这是一句需要被切开的测试内容，用来验证补发链路是否正常工作。" * 12


@pytest.fixture(scope="module")
def rs_collect():
    """按真机加载方式（包式）导入 rs_collect 模块。"""
    module = load_plugin_module(PLUGIN_DIR)
    return importlib.import_module(module.__name__ + ".rs_collect")


# ── locate_segments：切点边界反推 ───────────────────────────────────────────


def test_locate_newline_boundary(rs_collect):
    located = rs_collect.locate_segments("整理曲库\n晚点试播？", ["整理曲库", "晚点试播？"])
    assert [item["boundary"] for item in located] == ["start", "newline"]
    assert all(item["merged_newlines"] == 0 for item in located)


def test_locate_paragraph_boundary(rs_collect):
    located = rs_collect.locate_segments("第一段\n\n第二段", ["第一段", "第二段"])
    assert [item["boundary"] for item in located] == ["start", "paragraph"]


def test_locate_sentence_boundary(rs_collect):
    located = rs_collect.locate_segments("今天很好。明天更好。", ["今天很好。", "更好。"][:1] + ["明天更好。"])
    assert [item["boundary"] for item in located] == ["start", "sentence"]


def test_locate_comma_and_soft_boundary(rs_collect):
    located = rs_collect.locate_segments("先说这个，再说那个", ["先说这个，", "再说那个"])
    assert located[1]["boundary"] == "comma"
    located = rs_collect.locate_segments("准备好了：开始吧", ["准备好了：", "开始吧"])
    assert located[1]["boundary"] == "soft"


def test_locate_cjk_space_vs_plain_space(rs_collect):
    located = rs_collect.locate_segments("好 那再来一段", ["好", "那再来一段"])
    assert located[1]["boundary"] == "cjk_space"
    located = rs_collect.locate_segments("use the force", ["use the", "force"])
    assert located[1]["boundary"] == "space"


def test_merged_newlines_counted(rs_collect):
    # _merge_to_limit 跨强制边界合并：原文的换行被打包吞掉、分隔符还原成空格
    located = rs_collect.locate_segments(
        "好\n那再来一段今天把电台设备擦了一遍",
        ["好 那再来一段今天把电台设备擦了一遍"],
    )
    assert located[0]["merged_newlines"] == 1


def test_hard_cut_detected(rs_collect):
    # 无任何干净边界的纯汉字长句，按硬上限硬切
    located = rs_collect.locate_segments("今天天气不错我们出去玩吧", ["今天天气不错我", "们出去玩吧"])
    assert located[1]["hard_cut"] is True
    assert located[1]["boundary"] == "hard"


def test_unknown_on_unmatchable_segment(rs_collect):
    located = rs_collect.locate_segments("abc", ["完全不在原文里的一段话"])
    assert located[0]["boundary"] == "unknown"


def test_locate_matches_real_splitter_output(rs_collect):
    """端到端：对 split_reply 的真实产出反推，四不变量之外还要「每段都定位成功」。"""
    from rs_splitter import SplitRules, split_reply

    rules = SplitRules(min_length=1, soft_max_length=40, max_segments=10, min_segment_length=0)
    text = "今天主要整理一下曲库，晚点试播？\n先把这些记下来。回头再对一遍清单，别漏了电烙铁和面包板。"
    segments = split_reply(text, rules)
    assert len(segments) >= 2
    located = rs_collect.locate_segments(text, segments)
    assert all(item["boundary"] != "unknown" for item in located), f"真实产出出现失配: {located}"
    assert sum(item["len"] for item in located) >= len(text) - 20  # strip 掉的空白允许少量出入


# ── Collector：落盘 ─────────────────────────────────────────────────────────


def test_collector_writes_jsonl_per_day(tmp_path, rs_collect):
    base = tmp_path / "collect"
    collector = rs_collect.Collector(base)
    collector.write({"event": "split", "rid": "abc123", "text": "原文"})
    collector.write({"event": "no_split", "rid": "def456"})
    files = list(base.glob("*.jsonl"))
    assert len(files) == 1, "同一天的两次写入应落在同一文件"
    records = [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert records[0]["event"] == "split" and records[0]["rid"] == "abc123"
    assert records[0]["text"] == "原文"
    # 每条记录都带时间戳
    assert all(abs(record["ts"] - time.time()) < 60 for record in records)


def test_collector_failure_is_silent(tmp_path, rs_collect):
    """落盘失败绝不外抛——采集是旁路，最坏情况就是丢样本。"""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir", encoding="utf-8")
    collector = rs_collect.Collector(blocker)  # base 是文件，mkdir 必败
    collector.write({"event": "split"})  # 不应抛异常
    assert collector.failed_writes == 1


# ── 集成：fakehost 全链路 ───────────────────────────────────────────────────


def _build_plugin(collect_dir: pathlib.Path | None):
    module = load_plugin_module(PLUGIN_DIR)
    plugin = module.create_plugin()
    host = FakeHost(plugin_id="org.mai-mai.reply-splitter")
    ctx = build_context("org.mai-mai.reply-splitter", rpc_call=host.rpc_call)
    config = get_default_config(getattr(type(plugin), "config_model", None))
    if collect_dir is not None:
        config["collect"]["enabled"] = True
        config["collect"]["dir"] = str(collect_dir)
    bind_context(plugin, ctx, config)
    return module, plugin, host


def _read_events(collect_dir: pathlib.Path) -> list[dict]:
    events: list[dict] = []
    for path in sorted(collect_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
    return events


def test_default_config_collect_disabled(tmp_path):
    """默认关闭（D8）：跑完整切分流程后，插件目录不得出现 collect/。"""
    module, plugin, _host = _build_plugin(collect_dir=None)
    assert plugin.config.collect.enabled is False

    async def run():
        await plugin.on_load()
        # 未标记来源的出站（skip 路径）与正常路径都不应触发任何写盘
        await plugin.split_outgoing_message(
            message={"session_id": "s-x", "processed_plain_text": LONG_TEXT},
            processed_plain_text=LONG_TEXT,
            stream_id="s-x",
        )
        await plugin.on_unload()

    asyncio.run(run())
    assert not (PLUGIN_DIR / "collect").exists(), "采集默认关闭却产生了落盘"


def test_split_flow_writes_split_and_resend_events(tmp_path):
    collect_dir = tmp_path / "collect"
    module, plugin, host = _build_plugin(collect_dir=collect_dir)

    async def run():
        await plugin.on_load()
        stream = "s-collect"
        await plugin.arm_replyer_reply(session_id=stream)
        message = {"session_id": stream, "processed_plain_text": LONG_TEXT}
        result = await plugin.split_outgoing_message(
            message=message, processed_plain_text=LONG_TEXT, stream_id=stream
        )
        assert "modified_kwargs" in result
        # 复用同一个 message（共享对象已被就地改写成首段）→ 实发校验应判 ok
        await plugin.dispatch_pending_segments(message=message, sent=True, stream_id=stream)
        await asyncio.sleep(0.15)  # 等后台补发任务跑完
        await plugin.on_unload()

    asyncio.run(run())
    events = _read_events(collect_dir)
    kinds = [event["event"] for event in events]
    assert "split" in kinds, f"缺 split 事件: {kinds}"
    assert "resend" in kinds, f"缺 resend 事件: {kinds}"

    split = next(event for event in events if event["event"] == "split")
    assert split["text"] == LONG_TEXT, "原文未落盘（D6）"
    assert len(split["segments"]) >= 2
    assert split["located"][0]["boundary"] == "start"
    assert split["rules"]["soft_max_length"] == 180, "配置快照缺失"
    assert split["inplace_ok"] is True
    rid = split["rid"]
    assert rid, "split 事件缺 rid"
    resend = next(event for event in events if event["event"] == "resend")
    assert resend["rid"] == rid, "rid 未贯穿 split → resend"
    assert resend["total"] == len(split["segments"]) - 1
    assert resend["failed"] == 0 and resend["cancelled"] is False


def test_verify_failure_recorded(tmp_path):
    """宿主未采纳改写 → 实发校验判 failed，必须留下 verify 事件（C 类硬信号）。"""
    collect_dir = tmp_path / "collect"
    module, plugin, _host = _build_plugin(collect_dir=collect_dir)

    async def run():
        await plugin.on_load()
        stream = "s-opaque"
        await plugin.arm_replyer_reply(session_id=stream)
        # message 不含正文 → 只改到副本；宿主照旧发全文
        opaque_message = {"session_id": stream}
        result = await plugin.split_outgoing_message(
            message=opaque_message, processed_plain_text=LONG_TEXT, stream_id=stream
        )
        assert "modified_kwargs" in result
        await plugin.dispatch_pending_segments(
            message={"session_id": stream, "processed_plain_text": LONG_TEXT},
            sent=True,
            stream_id=stream,
        )
        await plugin.on_unload()

    asyncio.run(run())
    events = _read_events(collect_dir)
    verify = [event for event in events if event["event"] == "verify"]
    assert verify, "实发校验失败未留事件"
    assert verify[0]["verdict"] == "failed"
    assert verify[0]["rid"], "verify 事件缺 rid"


def test_skip_event_recorded(tmp_path):
    collect_dir = tmp_path / "collect"
    module, plugin, _host = _build_plugin(collect_dir=collect_dir)

    async def run():
        await plugin.on_load()
        stream = "s-skip"
        plugin._begin_resend(stream)
        echo_text = "补发中的分段文本"
        plugin._mark_sending_text(stream, plugin._normalize_for_compare(echo_text))
        try:
            await plugin.split_outgoing_message(
                message={"session_id": stream, "processed_plain_text": echo_text},
                processed_plain_text=echo_text,
                stream_id=stream,
            )
        finally:
            plugin._unmark_sending_text(stream, plugin._normalize_for_compare(echo_text))
            plugin._end_resend(stream)
        await plugin.on_unload()

    asyncio.run(run())
    events = _read_events(collect_dir)
    skips = [event for event in events if event["event"] == "skip"]
    assert skips and skips[0]["reason"] == "resend_echo"


def test_new_reply_during_resend_still_splits(tmp_path):
    """文本级重入判据（v1.2.0）：补发期间同会话另一轮回复照常切分，
    只有回环补发文本被拦。"""
    collect_dir = tmp_path / "collect"
    module, plugin, _host = _build_plugin(collect_dir=collect_dir)

    async def run():
        await plugin.on_load()
        stream = "s-concurrent"
        echo_text = "补发中的分段文本"
        plugin._begin_resend(stream)
        plugin._mark_sending_text(stream, plugin._normalize_for_compare(echo_text))
        try:
            # 回环文本 → 拦
            echo_result = await plugin.split_outgoing_message(
                message={"session_id": stream, "processed_plain_text": echo_text},
                processed_plain_text=echo_text,
                stream_id=stream,
            )
            assert "modified_kwargs" not in echo_result
            # 新一轮回复 → 照常切分
            await plugin.arm_replyer_reply(session_id=stream)
            fresh = await plugin.split_outgoing_message(
                message={"session_id": stream, "processed_plain_text": LONG_TEXT},
                processed_plain_text=LONG_TEXT,
                stream_id=stream,
            )
            assert "modified_kwargs" in fresh, "并发轮次被会话级一票否决误拦"
        finally:
            plugin._unmark_sending_text(stream, plugin._normalize_for_compare(echo_text))
            plugin._end_resend(stream)
        plugin._pending.pop(stream, None)
        await plugin.on_unload()

    asyncio.run(run())
    events = _read_events(collect_dir)
    kinds = [event["event"] for event in events]
    assert "split" in kinds
    skips = [event for event in events if event["event"] == "skip"]
    assert skips and skips[0]["reason"] == "resend_echo"


def test_planner_wait_hook_releases_without_pending(tmp_path):
    """planner 等待钩子：无待补发时立即放行。"""
    module, plugin, _host = _build_plugin(collect_dir=None)

    async def run():
        await plugin.on_load()
        result = await plugin.wait_for_pending_segments_planner(
            session_id="s-planner", stream_id="s-planner"
        )
        assert result.get("action") == "continue"
        await plugin.on_unload()

    asyncio.run(run())


def test_no_split_event_recorded(tmp_path):
    collect_dir = tmp_path / "collect"
    module, plugin, _host = _build_plugin(collect_dir=collect_dir)

    async def run():
        await plugin.on_load()
        stream = "s-short"
        await plugin.arm_replyer_reply(session_id=stream)
        short_text = "好的，收到。这条回复足够短不会被切开，但还是记录一下。" * 2
        result = await plugin.split_outgoing_message(
            message={"session_id": stream, "processed_plain_text": short_text},
            processed_plain_text=short_text,
            stream_id=stream,
        )
        assert "modified_kwargs" not in result
        await plugin.on_unload()

    asyncio.run(run())
    events = _read_events(collect_dir)
    no_split = [event for event in events if event["event"] == "no_split"]
    assert no_split, "no_split 事件缺失"
    assert no_split[0]["text"].startswith("好的，收到。")
