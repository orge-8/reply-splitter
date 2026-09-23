"""上线前审计用例（代码审查 / 安全审计 / QA）。

与 `test_splitter_core.py` 的分工：
  - core 测「规则本身对不对」
  - 本文件测「退化输入、性能上界、状态机守卫、文档一致性」——
    这些是静态审查看不出来、只有跑起来才能证明的项。

每条用例都对应审计清单里的一个具体风险，注释里写明它防的是什么。

运行: python -m pytest tests/ -v
"""

from __future__ import annotations

import asyncio
import pathlib
import re
import sys
import time

import pytest

PLUGIN_DIR = pathlib.Path(__file__).resolve().parent.parent
TESTS_DIR = pathlib.Path(__file__).resolve().parent
for _path in (str(PLUGIN_DIR), str(TESTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from rs_splitter import SplitRules, split_reply  # noqa: E402

# 审计用规则：刻意取较小长度，让切分路径被充分触发
AUDIT_RULES = SplitRules(
    min_length=1, soft_max_length=40, max_segments=10, min_segment_length=0
)

LONG_TEXT = "这是一句需要被切开的测试内容，用来验证补发链路是否正常工作。" * 12


def _normalized(text: str) -> str:
    return "".join(text.split())


# ── 1. 边界输入组 ───────────────────────────────────────────────────────────
# 审计要求：每类退化输入都要单独成组。这类缺陷不报错、不崩溃，
# 只是安静地产出错误行为，静态审查几乎发现不了。

EDGE_INPUTS = {
    "空字符串": "",
    "纯空白": "   \n\t  \u3000 ",
    "单字符": "啊",
    "单个标点": "。",
    "只有空格": " " * 2000,
    "只有换行": "\n" * 2000,
    "只有标点": "。！？，、" * 500,
    "只有占位符": "[图片][表情]" * 200,
    "只有emoji": "🌾" * 300,
    "颜文字重复": "（╯°□°）╯︵ ┻━┻" * 50,
    "未闭合代码块": "```python\n" + "x = 1\n" * 200,
    "未闭合行内代码": "`" * 500,
    "超长单原子URL": "https://e.com/" + "x" * 3000,
    "零宽字符": "\u200b" * 500,
    "混合退化": " \n [图片] 。 \t 啊 \u3000",
}


@pytest.mark.parametrize("name", sorted(EDGE_INPUTS))
def test_退化输入_不崩溃且零内容丢失(name: str) -> None:
    text = EDGE_INPUTS[name]
    segments = split_reply(text, AUDIT_RULES)
    assert isinstance(segments, list), "返回值类型必须是 list"
    for segment in segments:
        assert segment, "不允许出现空段"
        assert segment.strip() == segment, f"段首尾不应有空白: {segment[:40]!r}"
    assert _normalized("".join(segments)) == _normalized(text), f"退化输入丢了内容: {name}"


# ── 2. 确定性与幂等 ─────────────────────────────────────────────────────────
# 审计要求：同一输入必须得到同一结果，否则真机行为无法复现、日志无法比对。

AUDIT_CORPUS = {
    "纯空格短句": "好 那再测一次 今天下午有点犯困 泡了杯茶继续整理曲库 翻到几首很适合秋天的歌 前奏一响就舍不得跳过了",
    "带标点长句": "这个方案我觉得可以。第一件事是把宿主的分割器关掉。第二件事是挂上自己的 hook。第三件事是补发剩余段落。",
    "中英混排": "我今天去了那家新开的咖啡店 the coffee shop nearby is quite nice 环境还不错 味道也还行",
    "含代码块": '示例：```python\nreturn {"action": "continue"}\n``` 注意装饰器要紧贴 def 定义。',
    "含URL": "参考实现在这里 https://github.com/saberlights/smart_segmentation_plugin 你可以先看架构。",
    "含换行": "整理曲库\n晚点试播？",
}


@pytest.mark.parametrize("name", sorted(AUDIT_CORPUS))
def test_切分确定性_同输入同输出(name: str) -> None:
    text = AUDIT_CORPUS[name]
    results = [split_reply(text, AUDIT_RULES) for _ in range(3)]
    assert results[0] == results[1] == results[2], f"切分不确定: {results}"


# ── 3. 性能上界（审计第 13 项：同步阻塞事件循环）────────────────────────────
# split_reply 跑在 BLOCKING hook 里**同步执行**，一旦变慢就是整个事件循环被卡住，
# 期间 bot 收不到任何消息，而日志毫无异常。这是最难发现的一类缺陷。


def test_性能_现实规模输入不构成阻塞() -> None:
    text = "这是一句测试内容，用来验证性能与阻塞。" * 300  # 约 5700 字
    start = time.perf_counter()
    split_reply(text, AUDIT_RULES)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.15, f"5700 字耗时 {elapsed * 1000:.0f}ms，作为 BLOCKING hook 偏慢"


def test_性能_病态配置不触发二次复杂度() -> None:
    """回归：兜底合并曾是 O(n²)。

    修复前实测：`soft=hard=1, max_segments=5` 下 48000 字耗时 **88.5 秒**
    （12000 字 5.5 秒），而它同步跑在 BLOCKING hook 里 —— 足以让 bot 失去响应。
    改为「按序均匀分组」后应呈线性。本用例把「不许再出现二次增长」钉死。
    """
    text = "测试内容" * 6000  # 24000 字
    rules = SplitRules(
        min_length=1, soft_max_length=1, hard_max_length=1,
        max_segments=5, min_segment_length=0,
    )
    start = time.perf_counter()
    segments = split_reply(text, rules)
    elapsed = time.perf_counter() - start
    assert len(segments) <= 5, f"段数超限: {len(segments)}"
    assert elapsed < 1.0, f"病态配置 24000 字耗时 {elapsed:.2f}s —— 疑似二次复杂度回归"
    assert _normalized("".join(segments)) == _normalized(text), "合并过程丢了内容"


def test_性能_大量未闭合围栏不引发回溯爆炸() -> None:
    """代码块正则用了惰性量词 + `\\Z` 兜底，必须确认没有灾难性回溯。"""
    text = "```x " * 4000
    start = time.perf_counter()
    split_reply(text, AUDIT_RULES)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"未闭合围栏耗时 {elapsed:.2f}s —— 疑似正则回溯爆炸"


def test_性能_深嵌套保护区间不退化() -> None:
    text = "（嵌套（括号）内容）" * 800
    start = time.perf_counter()
    split_reply(text, AUDIT_RULES)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"括号嵌套耗时 {elapsed:.2f}s"


# ── 4. 插件级状态机守卫（用 FakeHost 驱动）──────────────────────────────────


def _build_plugin():
    from fakehost import (
        FakeHost,
        bind_context,
        build_context,
        get_default_config,
        load_plugin_module,
    )

    module = load_plugin_module(PLUGIN_DIR)
    plugin = module.create_plugin()
    host = FakeHost(plugin_id="org.mai-mai.reply-splitter")
    ctx = build_context("org.mai-mai.reply-splitter", rpc_call=host.rpc_call)
    bind_context(plugin, ctx, get_default_config(getattr(type(plugin), "config_model", None)))
    return module, plugin, host


def test_已有待补发时不覆盖旧登记() -> None:
    """审计：同一聊天流在上一条补发完成前又来一条，旧分段曾被静默丢弃。

    `self._pending[stream_id] = ...` 是覆盖写。若第一条回复的 after_send 还没到、
    第二条就已经构建完成，第一条剩余的分段会无声消失 —— 用户看到的是「回复少了几句」，
    而日志里什么都没有。
    """
    module, plugin, host = _build_plugin()
    module_name = "audit_pending_guard"

    async def run() -> None:
        await plugin.on_load()
        stream = "s-overwrite"
        await plugin.arm_replyer_reply(session_id=stream)

        first = "第一句内容足够长用来触发切分。" * 20
        result = await plugin.split_outgoing_message(
            message={"session_id": stream, "processed_plain_text": first},
            processed_plain_text=first,
            stream_id=stream,
        )
        assert "modified_kwargs" in result, "首次切分应成功"
        pending_before = list(plugin._pending[stream]["segments"])

        # 第二条同流消息到达（第一条的 after_send 尚未触发）
        await plugin.arm_replyer_reply(session_id=stream)
        second = "第二句完全不同的内容也足够长。" * 20
        result2 = await plugin.split_outgoing_message(
            message={"session_id": stream, "processed_plain_text": second},
            processed_plain_text=second,
            stream_id=stream,
        )
        assert "modified_kwargs" not in result2, "已有待补发时不应再次切分"
        assert plugin._pending[stream]["segments"] == pending_before, "旧登记被覆盖了"
        await plugin.on_unload()

    try:
        asyncio.run(run())
    finally:
        sys.modules.pop(module_name, None)


def test_过期登记不得补发() -> None:
    """审计：清理只在有出站消息时执行，迟到很久的 after_send 曾能捞出过期分段。"""
    module, plugin, host = _build_plugin()

    async def run() -> None:
        await plugin.on_load()
        stream = "s-expired"
        await plugin.arm_replyer_reply(session_id=stream)
        text = "过期登记的内容也要足够长才能触发切分。" * 20
        message = {"session_id": stream, "processed_plain_text": text}
        result = await plugin.split_outgoing_message(
            message=message, processed_plain_text=text, stream_id=stream
        )
        first = result["modified_kwargs"]["processed_plain_text"]

        # 手动把登记改成已过期
        plugin._pending[stream]["expires_at"] = time.monotonic() - 1.0

        sends_before = len(host.calls_of("send.text"))
        await plugin.dispatch_pending_segments(
            message={"session_id": stream, "processed_plain_text": first},
            sent=True,
            stream_id=stream,
        )
        await asyncio.sleep(0.05)
        assert len(host.calls_of("send.text")) == sends_before, "过期登记被补发了"
        assert stream not in plugin._pending, "过期登记未清理"
        await plugin.on_unload()

    asyncio.run(run())


def test_发送失败时不补发() -> None:
    """副作用守卫：host 报 sent=False 时绝不能补发。"""
    module, plugin, host = _build_plugin()

    async def run() -> None:
        await plugin.on_load()
        stream = "s-notsent"
        await plugin.arm_replyer_reply(session_id=stream)
        text = "发送失败场景下的内容也要足够长才能触发切分。" * 20
        message = {"session_id": stream, "processed_plain_text": text}
        result = await plugin.split_outgoing_message(
            message=message, processed_plain_text=text, stream_id=stream
        )
        assert "modified_kwargs" in result
        first = result["modified_kwargs"]["processed_plain_text"]

        sends_before = len(host.calls_of("send.text"))
        await plugin.dispatch_pending_segments(
            message={"session_id": stream, "processed_plain_text": first},
            sent=False,
            stream_id=stream,
        )
        await asyncio.sleep(0.05)
        assert len(host.calls_of("send.text")) == sends_before, "sent=False 时仍补发了"
        await plugin.on_unload()

    asyncio.run(run())


def test_生命周期清理无残留() -> None:
    """审计第 9 项：卸载后不得留下后台任务或状态残留。"""
    module, plugin, host = _build_plugin()

    async def run() -> None:
        await plugin.on_load()
        stream = "s-cleanup"
        await plugin.arm_replyer_reply(session_id=stream)
        text = "生命周期清理场景的内容也要足够长。" * 8
        await plugin.split_outgoing_message(
            message={"session_id": stream, "processed_plain_text": text},
            processed_plain_text=text,
            stream_id=stream,
        )
        await plugin.on_unload()
        assert not plugin._tasks, "卸载后仍有后台任务"
        assert not plugin._pending, "卸载后仍有待补发登记"
        assert not plugin._resend_guards, "卸载后仍有重入保护残留"
        assert not plugin._armed, "卸载后仍来源标记残留"

    asyncio.run(run())


# ── 5. 命令正则边界 ─────────────────────────────────────────────────────────


def test_命令正则边界() -> None:
    """审计：命令正则内嵌在装饰器里无法审计，故抽成模块常量并逐档校验。"""
    module, _, _ = _build_plugin()
    pattern = module.SPLIT_COMMAND_PATTERN
    regex = re.compile(pattern)

    should_match = [
        "/split",
        "/split on",
        "/split off",
        "/split status",
        "  /split off  ",
        "／split on",          # 全角斜杠
        "/split OFF",          # 大小写不敏感
        "/split Status",
    ]
    should_not_match = [
        "x/split",             # 不能匹配正文中间
        "/splitter",           # 不能匹配更长的词
        "/split maybe",        # 非法的动作
        "split",               # 缺少斜杠
        "/split on extra",     # 多余参数
        "",
    ]
    for text in should_match:
        assert regex.match(text), f"应匹配但没匹配: {text!r}"
    for text in should_not_match:
        assert not regex.match(text), f"不该匹配却匹配了: {text!r}"

    # 具名组必须能取出动作（宿主靠它传 matched_groups）
    match = regex.match("/split off")
    assert match is not None and match.group("action") == "off"


# ── 6. 文档一致性（审计第 11 项）────────────────────────────────────────────


def test_README承诺的本地命令都真实存在() -> None:
    """审计：README 引用不存在的脚本会让 gate 根本跑不起来。

    本项目实测踩到：README 写了 `python run_gates.py --plugin .`，
    但 run_gates.py 属于 devkit，并未随插件分发。
    """
    readme = (PLUGIN_DIR / "README.md").read_text(encoding="utf-8")
    for script in ("check_plugin.py", "tests/smoke_test.py"):
        assert script in readme, f"README 未提及 {script}"
        assert (PLUGIN_DIR / script).exists(), f"README 承诺 {script} 存在，但文件缺失"
    # run_gates.py 必须在 README 里被明确标注为「不在此目录」
    assert "run_gates.py" in readme, "README 未说明门禁脚本位置"
    offending = [
        line.strip()
        for line in readme.splitlines()
        if line.strip().startswith("python run_gates.py")
    ]
    assert not offending, f"README 仍把 run_gates.py 写成可直接执行: {offending}"


def test_未包含敏感信息() -> None:
    """审计第 1/7 项：源码不得含凭据、绝对路径、QQ 号。"""
    forbidden = [
        (r"(skey|p_skey|token|secret|password|api[_-]?key)\s*=\s*[\"'][^\"']{8,}[\"']", "疑似硬编码凭据"),
        (r"(C:\\|C:/Users|/home/|D:\\|E:\\)", "绝对路径"),
        (r"[\"'][1-9][0-9]{6,11}[\"']", "疑似 QQ 号"),
        (r"\b(eval|exec|pickle\.loads|os\.system|subprocess)\s*\(", "危险调用"),
    ]
    for filename in ("plugin.py", "rs_splitter.py"):
        source = (PLUGIN_DIR / filename).read_text(encoding="utf-8")
        for pattern, label in forbidden:
            match = re.search(pattern, source)
            assert match is None, f"{filename} 命中{label}: {match.group(0)[:40]!r}"


def test_插件无出站网络与文件写入() -> None:
    """安全审计：本插件不应具备外发能力（零攻击面来自它最安全）。"""
    source = (PLUGIN_DIR / "plugin.py").read_text(encoding="utf-8")
    source += (PLUGIN_DIR / "rs_splitter.py").read_text(encoding="utf-8")
    for pattern, label in [
        (r"\b(httpx|requests|aiohttp|urllib|socket)\b", "网络库"),
        (r"\bopen\s*\(", "文件写入"),
    ]:
        assert not re.search(pattern, source), f"源码出现{label}"
