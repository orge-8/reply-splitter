"""冒烟测试：不启动 MaiBot，用 FakeHost 跑完插件生命周期与切分主流程。

运行: python tests/smoke_test.py
"""

from __future__ import annotations

import asyncio
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

PLUGIN_DIR = pathlib.Path(__file__).resolve().parent.parent

LONG_TEXT = "这是一句需要被切开的测试内容，用来验证补发链路是否正常工作。" * 12

# 组件名 -> 必须紧跟该装饰器的函数名。
# 装饰器与 def 之间一旦插入别的方法，SDK 会静默注册到错误的函数上：
# 本地直调全绿，真机一调就报 unexpected keyword argument。
EXPECTED_COMPONENTS = {
    "split": "cmd_split",
    "split_outgoing_message": "split_outgoing_message",
    "dispatch_pending_segments": "dispatch_pending_segments",
    "arm_replyer_reply": "arm_replyer_reply",
    "wait_for_pending_segments": "wait_for_pending_segments",
    "mark_command_active": "mark_command_active",
    "mark_command_inactive": "mark_command_inactive",
}

_DECORATORS = ("@Command(", "@Tool(", "@HookHandler(", "@EventHandler(", "@API(")


def _collect_component_pairs(source: str) -> dict[str, str]:
    """扫描源码，取出「组件名 -> 紧随其后的 def 名」。"""
    lines = source.splitlines()
    pairs: dict[str, str] = {}
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not any(stripped.startswith(item) for item in _DECORATORS):
            index += 1
            continue

        block = [stripped]
        depth = stripped.count("(") - stripped.count(")")
        while depth > 0 and index + 1 < len(lines):
            index += 1
            nxt = lines[index].strip()
            block.append(nxt)
            depth += nxt.count("(") - nxt.count(")")

        text = "\n".join(block)
        probe = index + 1
        func_name = ""
        while probe < len(lines):
            candidate = lines[probe].strip()
            if candidate.startswith(("async def ", "def ")):
                func_name = candidate.split("(", 1)[0].split()[-1]
                break
            if candidate and not candidate.startswith(("#", "@", ")")):
                break
            probe += 1

        name_match = re.search(r'name\s*=\s*"([^"]+)"', text)
        if name_match:
            component_name = name_match.group(1)
        else:
            positional = re.search(r'\(\s*"([^"]+)"', text)
            component_name = positional.group(1) if positional else ""

        assert component_name, f"未取到组件名，装饰器载荷异常:\n{text}"
        assert func_name, f"组件 {component_name} 后面没有紧跟 def:\n{text}"
        assert component_name not in pairs, f"组件名重复: {component_name}"
        pairs[component_name] = func_name
        index += 1
    return pairs


def _assert_component_binding() -> None:
    source = (PLUGIN_DIR / "plugin.py").read_text(encoding="utf-8")
    pairs = _collect_component_pairs(source)
    assert pairs == EXPECTED_COMPONENTS, (
        f"组件绑定与预期不符\n实际: {pairs}\n预期: {EXPECTED_COMPONENTS}"
    )


async def _wait_until(predicate, timeout: float = 3.0) -> None:
    """轮询等待；谓词内部必须重新取值，不能捕获旧结果。"""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"等待超时：{getattr(predicate, '__name__', predicate)}")


def _normalized(text: str) -> str:
    return "".join(text.split())


def main() -> int:
    try:
        import maibot_sdk  # noqa: F401
    except Exception:
        print("SKIP: 未安装 maibot-plugin-sdk，跳过冒烟测试（这不代表通过）")
        return 0

    _assert_component_binding()

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

    async def run() -> None:
        await plugin.on_load()
        assert plugin.config.plugin.enabled is True, "默认配置未生效"
        assert plugin.config.splitter.scope == "replyer_only", "默认 scope 不符"
        assert plugin._splitter_ready() is True, "插件应为可用状态"

        # ── 1. 未标记来源时不切分（scope=replyer_only 的默认行为）
        plain = {"session_id": "s-plain", "processed_plain_text": LONG_TEXT}
        result = await plugin.split_outgoing_message(
            message=plain, processed_plain_text=LONG_TEXT, stream_id="s-plain"
        )
        assert result.get("action") == "continue"
        assert "modified_kwargs" not in result, "未标记来源却在切分"
        assert plugin._replyer_hook_seen is False

        # ── 2. 标记来源后按规则切分，首段走宿主链路、其余登记待补发
        stream = "s-main"
        await plugin.arm_replyer_reply(session_id=stream)
        assert plugin._replyer_hook_seen is True, "replyer 标记钩子未生效"
        message = {"session_id": stream, "processed_plain_text": LONG_TEXT}
        result = await plugin.split_outgoing_message(
            message=message, processed_plain_text=LONG_TEXT, stream_id=stream
        )
        modified = result.get("modified_kwargs")
        assert modified, "标记来源后仍未切分"
        first = modified["processed_plain_text"]
        assert first != LONG_TEXT, "首段未被改写"
        assert modified["message"]["processed_plain_text"] == first, "message 内正文未同步改写"

        pending = plugin._pending[stream]["segments"]
        assert pending, "未登记待补发分段"
        assert len(pending) + 1 <= plugin.config.splitter.max_segments, "超出段数上限"
        assert _normalized(first + "".join(pending)) == _normalized(LONG_TEXT), "切分过程丢字"

        # ── 3. after_send 触发后台补发，逐条按序发出
        await plugin.dispatch_pending_segments(message=message, sent=True, stream_id=stream)
        await _wait_until(lambda: host.calls_of("send.text"))
        assert host.sent_texts == pending, f"补发内容或顺序不符: {host.sent_texts}"

        send_args = host.calls_of("send.text")[0]
        assert send_args.get("sync_to_maisaka_history") is True, "补发未同步到 Maisaka 历史"
        assert "typing" in send_args, "补发未透传 typing 参数"
        assert stream not in plugin._pending, "待补发登记未清理"

        # ── 4. 重入保护：补发期间同一聊天流的出站消息不得被再次切分
        await plugin.arm_replyer_reply(session_id=stream)
        plugin._begin_resend(stream)
        try:
            again = await plugin.split_outgoing_message(
                message={"session_id": stream, "processed_plain_text": LONG_TEXT},
                processed_plain_text=LONG_TEXT,
                stream_id=stream,
            )
            assert "modified_kwargs" not in again, "重入保护失效，补发内容会被二次切分"
        finally:
            plugin._end_resend(stream)

        # ── 5. 命令回执保护窗口内的回复不切分
        cmd_stream = "s-command"
        await plugin.mark_command_active(stream_id=cmd_stream)
        await plugin.arm_replyer_reply(session_id=cmd_stream)
        guarded = await plugin.split_outgoing_message(
            message={"session_id": cmd_stream, "processed_plain_text": LONG_TEXT},
            processed_plain_text=LONG_TEXT,
            stream_id=cmd_stream,
        )
        assert "modified_kwargs" not in guarded, "命令回执保护窗口未生效"
        await plugin.mark_command_inactive(stream_id=cmd_stream)

        # ── 6. 短回复不切分
        short_stream = "s-short"
        await plugin.arm_replyer_reply(session_id=short_stream)
        short_text = "好的，收到。"
        short_result = await plugin.split_outgoing_message(
            message={"session_id": short_stream, "processed_plain_text": short_text},
            processed_plain_text=short_text,
            stream_id=short_stream,
        )
        assert "modified_kwargs" not in short_result, "短回复不应被切分"
        assert short_stream not in plugin._pending

        # ── 7. 正文定位失败的载荷必须原样放行（宁可不少讲一遍）
        ghost_stream = "s-ghost"
        await plugin.arm_replyer_reply(session_id=ghost_stream)
        ghost = await plugin.split_outgoing_message(
            message={"session_id": ghost_stream}, processed_plain_text="", stream_id=ghost_stream
        )
        assert "modified_kwargs" not in ghost, "无法回填正文时不应切分"

        # ── 8. 管理命令
        ok, resp, intercept = await plugin.cmd_split(
            text="/split status", stream_id="s-cmd", matched_groups={"action": "status"}
        )
        assert ok is True and intercept is True, f"命令返回异常: {(ok, resp, intercept)}"
        assert "回复切分" in (resp or ""), "命令回执内容异常"
        assert "send.text" in [cap for cap, _ in host.calls], "命令未显式发送回执"

        ok, _, _ = await plugin.cmd_split(
            text="/split off", stream_id="s-cmd", matched_groups={"action": "off"}
        )
        assert ok is True
        assert plugin._splitter_ready() is False, "运行时开关关闭后仍处于可用状态"
        ok, _, _ = await plugin.cmd_split(
            text="/split on", stream_id="s-cmd", matched_groups={"action": "on"}
        )
        assert ok is True and plugin._splitter_ready() is True, "运行时开关未能重新开启"

        # ── 8b. 宿主未提供具名组时应能从正文兜底解析
        ok, _, _ = await plugin.cmd_split(text="/split off", stream_id="s-cmd")
        assert ok is True and plugin._splitter_ready() is False, "正文兜底解析失效"
        await plugin.cmd_split(text="/split on", stream_id="s-cmd")

        # ── 9. 非管理员静默拒绝（配置了 admin_ids 后）
        # 注意：非空配置必须带 plugin.config_version，否则 SDK 直接拒收
        plugin.set_plugin_config(
            {
                "plugin": {
                    "enabled": True,
                    "config_version": getattr(module, "SUPPORTED_CONFIG_VERSION", "0.1.0"),
                },
                "admin": {"admin_ids": ["123456789"]},
            }
        )
        denied, denied_resp, _ = await plugin.cmd_split(
            text="/split off", stream_id="s-cmd", user_id="999999"
        )
        assert denied is False and denied_resp is None, "非管理员未被静默拒绝"
        assert plugin._runtime_enabled is True, "非管理员竟改动了运行时开关"
        allowed, _, _ = await plugin.cmd_split(
            text="/split off", stream_id="s-cmd", user_id="qq:123456789"
        )
        assert allowed is True, "带平台前缀的管理员未被识别"
        await plugin.cmd_split(text="/split on", stream_id="s-cmd", user_id="123456789")

        # ── 10. 宿主未采纳改写时，实发校验必须拦住补发（防「同内容讲两遍」）
        # 这是真机踩过的坑：只改到自己那份副本也能让替换「看起来成功」，
        # 于是待补发被登记，而宿主照旧发全文 → 内容重复。
        opaque_stream = "s-opaque"
        await plugin.arm_replyer_reply(session_id=opaque_stream)
        # message 里不含正文 → 只能改到副本，message_changed 应为 False
        opaque_message = {"session_id": opaque_stream}
        result = await plugin.split_outgoing_message(
            message=opaque_message, processed_plain_text=LONG_TEXT, stream_id=opaque_stream
        )
        assert "modified_kwargs" in result, "仍应产生改写结果供宿主重读"
        assert opaque_stream in plugin._pending, "应已登记待补发"

        sends_before = len(host.calls_of("send.text"))
        # 宿主实际发出的是「全文」→ 校验必须判 failed 并放弃补发
        await plugin.dispatch_pending_segments(
            message={"session_id": opaque_stream, "processed_plain_text": LONG_TEXT},
            sent=True,
            stream_id=opaque_stream,
        )
        await asyncio.sleep(0.05)
        extra = host.calls_of("send.text")[sends_before:]
        assert not extra, f"实发校验失效，内容被重复发送: {[c.get('text') for c in extra]}"
        assert opaque_stream not in plugin._pending, "待补发登记未清理"

        # ── 11. 实发校验分类函数（安全关键，逐档钉住）
        classify = plugin._classify_sent_text
        first, full = "abc def", "abc def ghi"
        assert classify("abc def", first, full) == "ok", "实发等于首段应判 ok"
        assert classify("abc def ghi", first, full) == "failed", "实发等于全文应判 failed"
        assert classify("abc", first, full) == "ok", "实发是全文前缀应判 ok（容忍后处理改动）"
        assert classify("xyz", first, full) == "unknown", "判不准应返回 unknown"
        assert classify("", first, full) == "unknown", "空文本应返回 unknown"
        assert classify("abcdefghij", first, full) == "failed", "实发比全文还长应判 failed"

        await plugin.on_unload()
        assert not plugin._tasks and not plugin._pending, "卸载后仍有残留任务或登记"

    asyncio.run(run())
    print("smoke: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
