"""reply-splitter —— 用可配置的规则切分替换宿主内置回复分割器。

宿主内置的 `[response_splitter]` 只有长度 / 句数这类机械阈值，不理解结构，
所以会把链接、代码块、颜文字从中间切断。本插件接管的正是这一层：

  1. 先把宿主的 `[response_splitter] enable` 改成 false（否则双重切分）
  2. 在 `send_service.after_build_message` 阶段按规则切分，
     首段仍走宿主原始发送链（保留引用与平台发送行为），其余段落登记待补发
  3. 在 `send_service.after_send` 阶段起后台任务，按顺序补发剩余段落
  4. 用重入保护避免补发内容被二次切分

所有切分逻辑在 rs_splitter 模块里，是不依赖 ctx 的纯函数，可脱机单测。

结构自检:  python check_plugin.py --plugin .
冒烟测试:  python tests/smoke_test.py
行为测试:  python -m pytest tests/ -v
交付门禁:  python run_gates.py --plugin .
"""

import asyncio
import time
from typing import Any, ClassVar

from maibot_sdk import Command, Field, HookHandler, MaiBotPlugin, PluginConfigBase
from maibot_sdk.types import (
    CONFIG_RELOAD_SCOPE_SELF,
    ErrorPolicy,
    HookMode,
    HookOrder,
)

from .rs_splitter import SplitRules, split_reply

SUPPORTED_CONFIG_VERSION = "1.0.4"

# 待补发登记的存活时间：超过则丢弃，避免失败发送把旧分段一直挂着
_PENDING_TTL_SECONDS = 60.0
# 「本次是回复模型产出的回复」标记的存活时间
_ARM_TTL_SECONDS = 120.0
# 命令回执保护窗口：只兜住命令 hook 与 send_service 之间的异步抖动，不能长
_COMMAND_GRACE_SECONDS = 1.0
# 失联命令标记的回收时间（没收到 after_execute 时的兜底）
_COMMAND_ACTIVE_TTL_SECONDS = 300.0
# 补发单段的最大重试次数
_SEND_MAX_ATTEMPTS = 2
# 补发单段的 RPC 超时（显式指定，不依赖 cap.call 的 30s 默认值）
_SEND_RPC_TIMEOUT_MS = 60_000

# 命令正则抽成模块常量：便于审计用例直接校验，不必从源码里正则抠。
# `(?i:...)` 让 on/off/status 大小写不敏感（/split ON 也该生效）。
# 前缀要求 `^\s*[/／]`，因此不会误匹配正文中间的 "xxx/split"。
SPLIT_COMMAND_PATTERN = r"^\s*[/／]\s*split(?:\s+(?P<action>(?i:on|off|status)))?\s*$"

# 单条目标长度的下界。过小的值（如 1）会让每条消息只有一个字，
# 既无意义，又会把段数放大到万字量级、拖慢切分。
MIN_SOFT_MAX_LENGTH = 8

# 出站消息正文可能出现的字段名（按可信度排序）
_TEXT_KEYS = ("processed_plain_text", "display_message", "raw_message", "content", "text")


def _replace_in_place(node: Any, original: str, replacement: str, depth: int = 0) -> bool:
    """递归**就地**替换所有等于 original 的字符串，返回是否发生过替换。

    必须就地改「宿主持有的那个对象」。改我们自己的 kwargs 副本是无效的 ——
    宿主不一定会重新读取 modified_kwargs，但一定会读到共享对象的改动。
    """
    if depth > 6:
        return False
    changed = False
    if isinstance(node, dict):
        for key in list(node.keys()):
            value = node[key]
            if isinstance(value, str):
                if value == original:
                    node[key] = replacement
                    changed = True
            elif isinstance(value, (dict, list)):
                changed = _replace_in_place(value, original, replacement, depth + 1) or changed
    elif isinstance(node, list):
        for index, value in enumerate(node):
            if isinstance(value, str):
                if value == original:
                    node[index] = replacement
                    changed = True
            elif isinstance(value, (dict, list)):
                changed = _replace_in_place(value, original, replacement, depth + 1) or changed
    return changed


def _collect_strings(node: Any, out: list[str], depth: int = 0) -> list[str]:
    """收集节点里所有的字符串（用于在实发消息中定位正文）。"""
    if depth > 6:
        return out
    if isinstance(node, str):
        if node.strip():
            out.append(node)
    elif isinstance(node, dict):
        for value in node.values():
            _collect_strings(value, out, depth + 1)
    elif isinstance(node, list):
        for value in node:
            _collect_strings(value, out, depth + 1)
    return out


def _describe_text_paths(
    node: Any,
    prefix: str,
    min_len: int = 20,
    depth: int = 0,
    out: list[str] | None = None,
) -> list[str]:
    """列出节点里所有「够长的字符串」的路径与长度。

    用途：定位宿主究竟从哪个字段取正文 —— 把这里的候选长度
    与实发校验打印的样本长度对照，即可确定真正生效的是哪一个。
    """
    if out is None:
        out = []
    if depth > 6:
        return out
    if isinstance(node, str):
        if len(node.strip()) >= min_len:
            out.append(f"{prefix}({len(node)}字)")
    elif isinstance(node, dict):
        for key, value in node.items():
            _describe_text_paths(value, f"{prefix}.{key}", min_len, depth + 1, out)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _describe_text_paths(value, f"{prefix}[{index}]", min_len, depth + 1, out)
    return out


class PluginSectionConfig(PluginConfigBase):
    __ui_label__ = "插件"
    __ui_icon__ = "scissors"
    __ui_order__ = 0

    enabled: bool = Field(default=True, description="是否启用插件")
    config_version: str = Field(
        default=SUPPORTED_CONFIG_VERSION,
        description="配置版本（与插件版本同步）",
        json_schema_extra={"hidden": True, "disabled": True},
    )


class SplitterSectionConfig(PluginConfigBase):
    __ui_label__ = "切分规则"
    __ui_order__ = 1

    enabled: bool = Field(default=True, description="是否启用规则切分")
    scope: str = Field(
        default="replyer_only",
        description=(
            "切分范围。replyer_only 只切回复模型产出的回复（推荐，与宿主内置分割器一致）；"
            "all 切分所有出站文本"
        ),
    )
    min_length: int = Field(
        default=60, description="短于该长度的回复不切分（保留原样一条发出）"
    )
    soft_max_length: int = Field(
        default=180,
        ge=MIN_SOFT_MAX_LENGTH,
        description="单条目标长度，切点会挑在这个长度附近",
    )
    hard_max_length: int = Field(
        default=480,
        ge=MIN_SOFT_MAX_LENGTH,
        description="单条长度上限。仅在单段内含不可分割的链接或代码块时才允许超出",
    )
    max_segments: int = Field(default=6, description="一次回复最多切成几条")
    min_segment_length: int = Field(
        default=12, description="短于该长度的段会并回相邻段，避免出现孤立尾巴"
    )
    force_split_on_newline: bool = Field(
        default=True,
        description=(
            "把换行当作强制分条边界（推荐开启）。回复模型用换行表达「这句单独发一条」，"
            "关掉后换行只会被当作普通切点、可能被长度预算吞掉；"
            "代码块内部的换行永远不受影响"
        ),
    )
    protect_code: bool = Field(default=True, description="保护 markdown 代码块与行内代码不被切断")
    protect_url: bool = Field(default=True, description="保护链接不被切断")
    protect_brackets: bool = Field(default=True, description="保护括号对内的内容不被切断")
    protect_kaomoji: bool = Field(default=True, description="保护颜文字不被切断")
    typing: bool = Field(
        default=True, description="补发分段时沿用宿主的模拟打字等待（由 typing_speed 控制速度）"
    )
    wait_for_pending_ms: int = Field(
        default=8000,
        description="新消息到达时，最多等待同一聊天流的补发完成多少毫秒，避免回复交错",
    )
    diagnose_payload: bool = Field(
        default=False,
        description=(
            "诊断开关：打印出站载荷的字段结构，以及所有含长文本的字段路径与长度。"
            "把该长度与「实发校验」记录的样本长度对照，即可确定宿主究竟从哪个字段取正文"
        ),
    )


class AdminSectionConfig(PluginConfigBase):
    __ui_label__ = "管理"
    __ui_order__ = 2

    admin_ids: list[str] = Field(
        default_factory=list,
        description="允许操作运行时开关的管理员 ID 列表，兼容 123456789 与 qq:123456789 两种写法；留空表示所有人可用",
    )


class ReplySplitterConfig(PluginConfigBase):
    __ui_label__ = "回复切分"
    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig)
    splitter: SplitterSectionConfig = Field(default_factory=SplitterSectionConfig)
    admin: AdminSectionConfig = Field(default_factory=AdminSectionConfig)


class ReplySplitter(MaiBotPlugin):
    """插件主类。

    按 SDK 要求：辅助方法全部排在组件区之前，装饰器必须紧贴其后的 def，
    中间不插任何方法，否则会静默注册到错误的函数上。
    """

    config_model: ClassVar[type[PluginConfigBase] | None] = ReplySplitterConfig

    def __init__(self) -> None:
        super().__init__()
        self._runtime_enabled: bool = True
        # 重入保护：正在补发内容的聊天流，其出站消息不再被切分
        self._resend_guards: dict[str, int] = {}
        # 待补发：stream_id -> {"segments": [...], "expires_at": float}
        self._pending: dict[str, dict[str, Any]] = {}
        # 「本条出站源自回复模型」标记：stream_id -> 过期时间
        self._armed: dict[str, float] = {}
        # 是否观测到过 replyer 钩子（用于区分「钩子不存在」与「bot 还没回过话」）
        self._replyer_hook_seen: bool = False
        self._arm_warning_emitted: bool = False
        # 命令回执保护
        self._command_active: dict[str, int] = {}
        self._command_active_expiry: dict[str, float] = {}
        self._command_grace_expiry: dict[str, float] = {}
        # 后台补发任务（强引用防 GC）
        self._tasks: set[asyncio.Task[Any]] = set()

    # ── 生命周期 ────────────────────────────────────────────────────────────

    async def on_load(self) -> None:
        self._reset_state()
        self.ctx.logger.info(
            "reply-splitter 已加载 enabled=%s scope=%s 规则[min=%s soft=%s hard=%s 最多%s段]",
            self.config.plugin.enabled,
            self.config.splitter.scope,
            self.config.splitter.min_length,
            self.config.splitter.soft_max_length,
            self.config.splitter.hard_max_length,
            self.config.splitter.max_segments,
        )
        self.ctx.logger.info(
            "请确认宿主 config/bot_config.toml 中 [response_splitter] enable 已设为 false，"
            "否则会与本插件双重切分"
        )

    async def on_unload(self) -> None:
        self._cancel_tasks()
        self._reset_state()
        self.ctx.logger.info("reply-splitter 已卸载")

    async def on_config_update(self, scope: str, config_data: dict[str, Any], version: str) -> None:
        if scope == CONFIG_RELOAD_SCOPE_SELF:
            self.ctx.logger.info("reply-splitter 配置已热更新 version=%s", version)

    # ── 状态与工具方法（必须全部排在组件区之前）────────────────────────────

    def _reset_state(self) -> None:
        self._resend_guards.clear()
        self._pending.clear()
        self._armed.clear()
        self._command_active.clear()
        self._command_active_expiry.clear()
        self._command_grace_expiry.clear()
        self._replyer_hook_seen = False
        self._arm_warning_emitted = False

    def _cancel_tasks(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        self._tasks.clear()

    @staticmethod
    def _normalize_stream_id(value: Any) -> str:
        return str(value or "").strip()

    @classmethod
    def _resolve_stream_id(cls, kwargs: dict[str, Any], message: Any) -> str:
        """定位聊天流 ID：顶层字段优先，其次取 message 里的 session_id / stream_id。"""
        for key in ("stream_id", "session_id"):
            stream_id = cls._normalize_stream_id(kwargs.get(key))
            if stream_id:
                return stream_id
        if isinstance(message, dict):
            for key in ("session_id", "stream_id", "chat_id"):
                stream_id = cls._normalize_stream_id(message.get(key))
                if stream_id:
                    return stream_id
        return ""

    @staticmethod
    def _resolve_user_id(kwargs: dict[str, Any]) -> str:
        """取触发者 ID：优先顶层 user_id，兜底 message.message_info.user_info.user_id。"""
        user_id = str(kwargs.get("user_id") or "").strip()
        if user_id:
            return user_id
        message = kwargs.get("message")
        if isinstance(message, dict):
            message_info = message.get("message_info")
            if isinstance(message_info, dict):
                user_info = message_info.get("user_info")
                if isinstance(user_info, dict):
                    return str(user_info.get("user_id") or "").strip()
        return ""

    @staticmethod
    def _strip_platform_prefix(value: Any) -> str:
        """`qq:123456789` 与 `123456789` 两种写法都归一成末段小写。"""
        return str(value or "").split(":")[-1].strip().lower()

    def _is_admin(self, kwargs: dict[str, Any]) -> bool:
        """管理命令鉴权：自管 admin_ids，留空时全部允许（fail-open），本地操作者天然放行。"""
        if kwargs.get("is_local_operator"):
            return True
        configured = {self._strip_platform_prefix(item) for item in (self.config.admin.admin_ids or [])}
        configured.discard("")
        if not configured:
            return True
        return self._strip_platform_prefix(self._resolve_user_id(kwargs)) in configured

    def _build_rules(self) -> SplitRules:
        section = self.config.splitter
        return SplitRules(
            min_length=max(0, int(section.min_length)),
            # 运行时再兜一层下界：配置校验被绕过（如直接注入 dict）时也不能退化
            soft_max_length=max(MIN_SOFT_MAX_LENGTH, int(section.soft_max_length)),
            hard_max_length=max(MIN_SOFT_MAX_LENGTH, int(section.hard_max_length)),
            max_segments=max(1, int(section.max_segments)),
            min_segment_length=max(0, int(section.min_segment_length)),
            protect_code=bool(section.protect_code),
            protect_url=bool(section.protect_url),
            protect_brackets=bool(section.protect_brackets),
            protect_kaomoji=bool(section.protect_kaomoji),
            force_split_on_newline=bool(section.force_split_on_newline),
        )

    def _splitter_ready(self) -> bool:
        return bool(self.config.plugin.enabled) and bool(self.config.splitter.enabled) and self._runtime_enabled

    # 重入保护

    def _is_resending(self, stream_id: str) -> bool:
        return self._resend_guards.get(stream_id, 0) > 0

    def _begin_resend(self, stream_id: str) -> None:
        self._resend_guards[stream_id] = self._resend_guards.get(stream_id, 0) + 1

    def _end_resend(self, stream_id: str) -> None:
        remaining = self._resend_guards.get(stream_id, 0) - 1
        if remaining > 0:
            self._resend_guards[stream_id] = remaining
        else:
            self._resend_guards.pop(stream_id, None)

    # 命令回执保护

    def _mark_command_active(self, stream_id: str) -> None:
        now = time.monotonic()
        self._prune_expired(now)
        self._command_active[stream_id] = self._command_active.get(stream_id, 0) + 1
        self._command_active_expiry[stream_id] = now + _COMMAND_ACTIVE_TTL_SECONDS
        self._command_grace_expiry[stream_id] = now + _COMMAND_GRACE_SECONDS

    def _mark_command_inactive(self, stream_id: str) -> None:
        # 刻意不续期 grace 窗口：1 秒足够覆盖回执的同步发送
        remaining = self._command_active.get(stream_id, 0) - 1
        if remaining > 0:
            self._command_active[stream_id] = remaining
        else:
            self._command_active.pop(stream_id, None)
            self._command_active_expiry.pop(stream_id, None)

    def _in_command_protection(self, stream_id: str) -> bool:
        now = time.monotonic()
        self._prune_expired(now)
        if self._command_active.get(stream_id, 0) > 0:
            return True
        return self._command_grace_expiry.get(stream_id, 0.0) > now

    def _prune_expired(self, now: float) -> None:
        for stream_id, expiry in list(self._command_active_expiry.items()):
            if expiry <= now:
                self._command_active.pop(stream_id, None)
                self._command_active_expiry.pop(stream_id, None)
        for stream_id, expiry in list(self._pending.items()):
            if float(expiry.get("expires_at", 0.0)) <= now:
                self._pending.pop(stream_id, None)
        for stream_id, expiry in list(self._armed.items()):
            if expiry <= now:
                self._armed.pop(stream_id, None)

    # 回复来源标记

    def _arm_stream(self, stream_id: str) -> None:
        if stream_id:
            self._armed[stream_id] = time.monotonic() + _ARM_TTL_SECONDS

    def _is_armed(self, stream_id: str) -> bool:
        expiry = self._armed.get(stream_id, 0.0)
        return expiry > time.monotonic()

    def _consume_arm(self, stream_id: str) -> None:
        self._armed.pop(stream_id, None)

    def _warn_missing_replyer_hook(self) -> None:
        if self._arm_warning_emitted:
            return
        self._arm_warning_emitted = True
        self.ctx.logger.warning(
            "尚未观测到 maisaka.replyer.after_response，scope=replyer_only 下不会切分任何消息。"
            "若确认日志里始终没有该钩子，请把 splitter.scope 改成 all"
        )

    # 正文定位与回填

    @classmethod
    def _locate_text(cls, kwargs: dict[str, Any], message: Any) -> str:
        for key in _TEXT_KEYS:
            value = kwargs.get(key)
            if isinstance(value, str) and value.strip():
                return value
        if isinstance(message, dict):
            for key in _TEXT_KEYS:
                value = message.get(key)
                if isinstance(value, str) and value.strip():
                    return value
        return ""

    @classmethod
    def _apply_replacement(
        cls, payload: dict[str, Any], message: Any, original: str, replacement: str
    ) -> tuple[dict[str, Any] | None, bool]:
        """把首段写回出站载荷。返回 (modified_kwargs, message_changed)。

        两个动作要分开看：
          - **就地改 message**（宿主持有的共享对象）→ 只有这一步可能真正影响发送内容
          - 再把副本里的标量交回 modified_kwargs → 宿主若重读也一并生效

        ⚠ `message_changed` 只表示「我们动到了共享对象」，**不等于宿主一定会照它发**。
        实测踩过：仅改自己那份 payload 副本也能让替换"看起来成功"（因为
        processed_plain_text 是我们自己从具名形参并进去的），于是待补发被登记，
        而宿主照旧发全文 → 内容被讲了两遍。
        因此调用方不能仅凭返回值决定补发，必须在 send_service.after_send
        比对宿主实发的正文（见 _classify_sent_text）。
        """
        modified = dict(payload)
        message_changed = _replace_in_place(message, original, replacement)
        _replace_in_place(modified, original, replacement)
        if not message_changed and modified == payload:
            return None, False
        if isinstance(message, dict):
            modified["message"] = message
        return modified, message_changed

    # 实发内容校验（防「内容讲两遍」）

    @staticmethod
    def _normalize_for_compare(value: Any) -> str:
        return "".join(str(value or "").split())

    @classmethod
    def _classify_sent_text(cls, sent_text: Any, expected_first: str, full_text: str) -> str:
        """判断宿主实际发出的是「首段」还是「全文」。

        返回 ok（是首段，可以补发）/ failed（是全文，绝不能补发）/ unknown（判不准，不补发）。
        """
        sent = cls._normalize_for_compare(sent_text)
        if not sent:
            return "unknown"
        first = cls._normalize_for_compare(expected_first)
        full = cls._normalize_for_compare(full_text)
        if sent == first:
            return "ok"
        if sent == full:
            return "failed"
        # 错别字等后处理可能改动首段，用长度和前缀关系兜底
        if len(sent) < len(full) and full.startswith(sent):
            return "ok"
        if len(sent) >= len(full):
            return "failed"
        return "unknown"

    @classmethod
    def _verify_sent(cls, message: Any, payload_keys: dict[str, Any],
                     expected_first: str, full_text: str) -> tuple[str, str]:
        """在实发载荷里找出正文，判定它是首段还是全文。返回 (verdict, 实发样本)。"""
        candidates = _collect_strings(message, [])
        candidates.extend(_collect_strings(payload_keys, []))
        sample = max(candidates, key=len) if candidates else ""
        verdict = "unknown"
        for candidate in candidates:
            current = cls._classify_sent_text(candidate, expected_first, full_text)
            if current == "ok":
                return "ok", sample
            if current == "failed":
                verdict = "failed"
        return verdict, sample

    # 补发

    @staticmethod
    def _send_succeeded(result: Any) -> bool:
        if isinstance(result, bool):
            return result
        if isinstance(result, dict):
            return bool(result.get("sent", False))
        return bool(result)

    async def _send_one(self, segment: str, stream_id: str, typing: bool) -> bool:
        for attempt in range(_SEND_MAX_ATTEMPTS):
            try:
                result = await self.ctx.send.text(
                    segment,
                    stream_id,
                    return_details=False,
                    typing=typing and attempt == 0,
                    sync_to_maisaka_history=True,
                    timeout_ms=_SEND_RPC_TIMEOUT_MS,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - 单段失败不应中断后续分段
                self.ctx.logger.warning(
                    "补发分段失败（第 %s 次）stream=%s: %s", attempt + 1, stream_id, exc
                )
                result = False
            if self._send_succeeded(result):
                return True
        return False

    async def _drain_pending(self, stream_id: str, segments: list[str]) -> None:
        typing = bool(self.config.splitter.typing)
        self._begin_resend(stream_id)
        try:
            # 让出一次事件循环，先让宿主完成首段的发送与历史同步
            await asyncio.sleep(0)
            failed = 0
            for segment in segments:
                if not await self._send_one(segment, stream_id, typing):
                    failed += 1
            if failed:
                self.ctx.logger.error(
                    "补发分段未全部成功 stream=%s 失败 %s/%s 段", stream_id, failed, len(segments)
                )
            else:
                self.ctx.logger.info("补发分段完成 stream=%s 共 %s 段", stream_id, len(segments))
        except asyncio.CancelledError:
            self.ctx.logger.warning("补发任务被取消 stream=%s", stream_id)
            raise
        except Exception as exc:  # noqa: BLE001
            self.ctx.logger.error("补发任务异常 stream=%s: %s", stream_id, exc)
        finally:
            self._end_resend(stream_id)

    def _has_pending_work(self, stream_id: str) -> bool:
        return bool(self._pending.get(stream_id)) or self._is_resending(stream_id)

    async def _wait_pending(self, stream_id: str, timeout_ms: int) -> None:
        """等待同一聊天流的补发结束，避免新回复插到旧回复的分段中间。"""
        if timeout_ms <= 0 or not self._has_pending_work(stream_id):
            return
        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            if not self._has_pending_work(stream_id):
                return
            await asyncio.sleep(0.05)
        self.ctx.logger.warning("等待补发超时 stream=%s，放行新消息", stream_id)

    # ── 组件区 ──────────────────────────────────────────────────────────────

    @Command(
        "split",
        description="查看或切换回复切分的运行时开关",
        pattern=SPLIT_COMMAND_PATTERN,
    )
    async def cmd_split(self, **kwargs: Any) -> tuple[bool, str | None, bool]:
        stream_id = self._resolve_stream_id(kwargs, kwargs.get("message"))
        if not self._is_admin(kwargs):
            # 拒绝时静默：不发任何消息，只留日志
            self.ctx.logger.warning(
                "拒绝非管理员操作切分开关 user_id=%s", self._resolve_user_id(kwargs)
            )
            return False, None, True

        action = ""
        matched = kwargs.get("matched_groups")
        if isinstance(matched, dict):
            action = str(matched.get("action") or "").strip().lower()
        if not action:
            # 兜底：宿主未提供具名组时直接从正文解析，避免管理命令静默失效
            raw = str(kwargs.get("text") or kwargs.get("raw_message") or "").replace("／", "/")
            for token in raw.split()[1:]:
                if token.lower() in ("on", "off", "status"):
                    action = token.lower()
                    break
        if action not in ("", "on", "off", "status"):
            return False, None, True
        if action == "on":
            self._runtime_enabled = True
        elif action == "off":
            self._runtime_enabled = False

        reply = "回复切分：{}\n配置项开关：{}，切分范围：{}，单条目标 {} 字 / 最多 {} 段".format(
            "已开启" if self._runtime_enabled else "已关闭",
            "已开启" if (self.config.plugin.enabled and self.config.splitter.enabled) else "已关闭",
            self.config.splitter.scope,
            self.config.splitter.soft_max_length,
            self.config.splitter.max_segments,
        )
        await self.ctx.send.text(reply, stream_id)
        return True, reply, True

    @HookHandler(
        "send_service.after_build_message",
        name="split_outgoing_message",
        description="出站消息构建完成后按规则切分：首段留下，其余登记待补发",
        mode=HookMode.BLOCKING,
        order=HookOrder.LATE,
        timeout_ms=3000,
        error_policy=ErrorPolicy.SKIP,
    )
    async def split_outgoing_message(
        self,
        message: dict[str, Any] | None = None,
        processed_plain_text: str = "",
        display_message: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not self._splitter_ready():
            return {"action": "continue"}

        stream_id = self._resolve_stream_id(kwargs, message)
        if not stream_id:
            return {"action": "continue"}
        if self._is_resending(stream_id):
            return {"action": "continue"}
        if self._in_command_protection(stream_id):
            return {"action": "continue"}
        if stream_id in self._pending:
            # 上一条回复的分段还没补发完。此刻覆盖登记会把旧分段静默丢掉，
            # 所以宁可这一条不切分，也不能让已经承诺发出去的分段凭空消失。
            self.ctx.logger.warning(
                "该聊天流已有待补发分段，本次跳过切分以免覆盖 stream=%s", stream_id
            )
            return {"action": "continue"}

        # 具名形参不会出现在 **kwargs 里，必须显式并回载荷，
        # 否则回填时定位不到正文，切分会整条静默失效
        payload: dict[str, Any] = dict(kwargs)
        if isinstance(processed_plain_text, str) and processed_plain_text.strip():
            payload["processed_plain_text"] = processed_plain_text
        if isinstance(display_message, str) and display_message.strip():
            payload["display_message"] = display_message

        if self.config.splitter.diagnose_payload:
            self.ctx.logger.info(
                "[诊断] 出站载荷 stream=%s kwargs键=%s message键=%s",
                stream_id,
                sorted(kwargs.keys())[:16],
                sorted(message.keys())[:16] if isinstance(message, dict) else None,
            )
            self.ctx.logger.info(
                "[诊断] 含长文本的字段（对照实发样本长度即可确定宿主真正读取的是哪个）：%s",
                ", ".join(
                    _describe_text_paths(message, "message")
                    + _describe_text_paths(payload, "kwargs")
                )
                or "（无）",
            )

        scope = str(self.config.splitter.scope or "replyer_only").strip().lower()
        if scope != "all" and not self._is_armed(stream_id):
            if not self._replyer_hook_seen:
                self._warn_missing_replyer_hook()
            return {"action": "continue"}

        text = self._locate_text(payload, message)
        if not text:
            return {"action": "continue"}

        try:
            segments = split_reply(text, self._build_rules())
        except Exception as exc:  # noqa: BLE001 - 切分失败必须原样放行
            self.ctx.logger.error("切分异常，原样放行 stream=%s: %s", stream_id, exc)
            return {"action": "continue"}
        if len(segments) <= 1:
            return {"action": "continue"}

        modified, message_changed = self._apply_replacement(payload, message, text, segments[0])
        if modified is None:
            self.ctx.logger.warning(
                "未能在出站载荷里定位正文，放弃切分 stream=%s keys=%s",
                stream_id,
                sorted(kwargs.keys())[:12],
            )
            return {"action": "continue"}
        if not message_changed:
            self.ctx.logger.warning(
                "未能就地改写宿主共享对象（只改到了自己的副本），宿主可能仍发全文；"
                "补发前会做一次实发校验 stream=%s message类型=%s",
                stream_id,
                type(message).__name__,
            )

        self._pending[stream_id] = {
            "segments": segments[1:],
            # 校验依据：after_send 用它判断宿主实发的到底是首段还是全文
            "expected_first": segments[0],
            "full_text": text,
            "expires_at": time.monotonic() + _PENDING_TTL_SECONDS,
        }
        self._consume_arm(stream_id)
        self.ctx.logger.info(
            "切分为 %s 段 stream=%s 首段 %s 字，其余 %s 段待补发",
            len(segments),
            stream_id,
            len(segments[0]),
            len(segments) - 1,
        )
        return {"action": "continue", "modified_kwargs": modified}

    @HookHandler(
        "send_service.after_send",
        name="dispatch_pending_segments",
        description="首段发送完成后，起后台任务按顺序补发剩余分段",
        mode=HookMode.OBSERVE,
    )
    async def dispatch_pending_segments(
        self,
        message: dict[str, Any] | None = None,
        sent: bool = False,
        **kwargs: Any,
    ) -> None:
        if not sent or not self._splitter_ready():
            return
        stream_id = self._resolve_stream_id(kwargs, message)
        if not stream_id:
            return
        entry = self._pending.pop(stream_id, None)
        if not entry:
            return
        if float(entry.get("expires_at", 0.0)) <= time.monotonic():
            # 清理只在有出站消息时才跑，所以这里必须自己再判一次：
            # 迟到的 after_send 不能把早已过期的分段补发出去。
            self.ctx.logger.warning("待补发登记已过期，放弃补发 stream=%s", stream_id)
            return
        segments = list(entry.get("segments") or [])
        if not segments:
            return

        # 实发校验：宿主发出来的到底是首段还是全文？
        # 判不准时一律不补发 —— 内容被讲两遍比「没切成」糟糕得多。
        verdict, sample = self._verify_sent(
            message,
            kwargs,
            str(entry.get("expected_first") or ""),
            str(entry.get("full_text") or ""),
        )
        if verdict != "ok":
            self.ctx.logger.warning(
                "实发校验未通过，已放弃补发以避免内容重复。verdict=%s stream=%s 实发样本 %s 字=%r",
                verdict,
                stream_id,
                len(sample),
                sample[:60],
            )
            return

        task = asyncio.create_task(self._drain_pending(stream_id, segments))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    @HookHandler(
        "maisaka.replyer.after_response",
        name="arm_replyer_reply",
        description="标记该聊天流接下来会有一条回复模型产出的回复（纯标记，不调模型）",
        mode=HookMode.BLOCKING,
        error_policy=ErrorPolicy.SKIP,
    )
    async def arm_replyer_reply(self, **kwargs: Any) -> dict[str, Any]:
        self._replyer_hook_seen = True
        stream_id = self._resolve_stream_id(kwargs, None)
        if stream_id:
            self._arm_stream(stream_id)
        return {"action": "continue"}

    @HookHandler(
        "chat.receive.before_process",
        name="wait_for_pending_segments",
        description="新消息到达时先等同一聊天流的补发结束，避免回复交错",
        mode=HookMode.BLOCKING,
        timeout_ms=12000,
        error_policy=ErrorPolicy.SKIP,
    )
    async def wait_for_pending_segments(
        self, message: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        if not self._splitter_ready():
            return {"action": "continue"}
        stream_id = self._resolve_stream_id(kwargs, message)
        if stream_id:
            await self._wait_pending(stream_id, int(self.config.splitter.wait_for_pending_ms))
        return {"action": "continue"}

    @HookHandler(
        "chat.command.before_execute",
        name="mark_command_active",
        description="进入命令回执保护窗口",
        mode=HookMode.BLOCKING,
        error_policy=ErrorPolicy.SKIP,
    )
    async def mark_command_active(
        self, message: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        stream_id = self._resolve_stream_id(kwargs, message)
        if stream_id:
            self._mark_command_active(stream_id)
        return {"action": "continue"}

    @HookHandler(
        "chat.command.after_execute",
        name="mark_command_inactive",
        description="退出命令回执保护窗口（保留 1 秒抖动兜底）",
        mode=HookMode.BLOCKING,
        error_policy=ErrorPolicy.SKIP,
    )
    async def mark_command_inactive(
        self, message: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        stream_id = self._resolve_stream_id(kwargs, message)
        if stream_id:
            self._mark_command_inactive(stream_id)
        return {"action": "continue"}


def create_plugin() -> ReplySplitter:
    """Runner 加载入口。"""
    return ReplySplitter()
