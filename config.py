# encoding:utf-8
"""
MyAgent 配置系统模块
=====================
职责：
  1. 定义所有合法配置项及其默认值（available_setting 白名单）
  2. 从 config.json 加载用户配置
  3. 支持环境变量覆盖（方便云端部署时不暴露密钥）
  4. 日志输出时自动脱敏敏感信息（api_key / secret）
  5. 提供全局单例 config，供所有模块使用

设计原则：
  - 所有配置项必须在 available_setting 中注册，否则读取时返回 None
  - 环境变量可以覆盖 JSON 配置（格式: 全小写，如 openai_api_key=sk-xxx）
  - 脱敏逻辑内建于日志输出，不需各模块自己处理
"""

import json
import os
import logging

from common.logging import configure_logging

# 获取本模块的 logger，后续所有日志都用它输出
logger = logging.getLogger(__name__)


# ── 配置白名单 ─────────────────────────────────────────────────
# available_setting 同时承担三种职责：
#   ① 文档：每个键的注释说明用途和可选值
#   ② 白名单：env var 只能覆盖此字典中存在的键（安全考量）
#   ③ 默认值：config.json 中缺少的键从此处取默认值
#
# 新增配置项时，只需要在这里加一行即可，load_config 无需改动。
# ────────────────────────────────────────────────────────────────

available_setting = {
    # ===== 语言 =====
    # 全局 UI 语言，影响 CLI 输出、错误提示、Agent 系统提示词
    # 可选值: "auto"(跟随系统) / "zh"(中文) / "en"(英文)
    "language": "auto",

    # ===== OpenAI 兼容 API 配置（主模型） =====
    # 大部分模型服务商都提供 OpenAI 兼容格式，这一套配置能覆盖 OpenAI / DeepSeek / 硅基流动等
    "openai_api_key": "",             # OpenAI 兼容 API 的密钥
    "openai_api_base": "https://api.openai.com/v1",  # API 地址，换服务商时改这里即可
    "openai_model": "gpt-4o",         # 模型名，如 gpt-4o / deepseek-chat / gpt-4-turbo

    # ===== Claude API 配置 =====
    "claude_api_key": "",              # Anthropic Claude API 密钥
    "claude_api_base": "https://api.anthropic.com/v1",
    "claude_model": "claude-sonnet-4-6",

    # ===== DeepSeek API 配置 =====
    # DeepSeek 也支持 OpenAI 兼容格式，但单独配置可以独立切换模型
    "deepseek_api_key": "",
    "deepseek_api_base": "https://api.deepseek.com/v1",
    "deepseek_model": "deepseek",
    "deepseek_thinking_enabled": True,  # 完整模式是否开启 DeepSeek 深度思考
    "deepseek_reasoning_effort": "high",  # 深度思考强度: high / max

    # ===== 当前使用的模型 =====
    # 必填，告诉系统现在用哪个模型，可选值: "openai" / "claude" / "deepseek"
    "model": "openai",

    # ===== HTTP 代理 =====
    # 需要科学上网时使用，如 "http://127.0.0.1:7890"
    "proxy": "",

    # ===== Agent 核心参数 =====
    "agent_enabled": True,             # 是否启用 Agent 模式（False 时退化为纯聊天模式，不调用工具）
    "agent_workspace": "~/myagent",    # Agent 工作区路径，存放技能、记忆、临时文件
    "agent_max_steps": 20,             # 每次运行最大工具调用步数，防止死循环
    "agent_max_context_tokens": 50000, # 上下文最大 token 数，超出会自动压缩
    "agent_max_context_turns": 20,     # 保留最近 N 轮对话（每轮 = 用户问题 + Agent 全部步骤）
    "agent_temperature": 0.7,          # LLM 温度参数，0=确定，1=创意
    "agent_top_p": 0.9,                # LLM top_p 采样参数
    "agent_lightweight_mode_enabled": True,  # 自动模式是否启用轻量问答
    "agent_lightweight_query_max_chars": 180,  # 自动识别的最大问题长度

    # ===== Skill 系统 =====
    "skills_enabled": True,             # 是否启用显式 /<name> Skill 调用
    "skills_builtin_dir": "",          # 空值表示使用项目根目录下的 skills/

    # ===== 记忆系统 =====
    "memory_enabled": True,            # 是否启用长期记忆
    "memory_max_entries": 500,         # 最多存储多少条记忆
    "memory_auto_refine_enabled": False,  # 是否在每天 18:00 后自动执行 refine
    "embedding_provider": "ollama",    # 向量嵌入服务: "ollama" / "openai"
    "embedding_model": "bge-m3",       # 默认本地 embedding 模型
    "embedding_api_base": "http://localhost:11434/api/embeddings",  # Ollama embedding API
    "embedding_openai_model": "text-embedding-3-small",  # 云端 OpenAI embedding 备用模型

    # ===== Web Console 配置 =====
    "web_enabled": True,               # 是否启动 Web 控制台
    "web_host": "127.0.0.1",           # 仅允许 localhost 或回环地址
    "web_port": 9899,                  # 监听端口
    "web_password": "",                # 预留配置，当前本机模式不启用密码登录
    "web_show_thinking": False,        # Web 页面默认是否展开模型思考

    # ===== 微信渠道配置 =====
    "wechat_enabled": False,           # 是否启用微信渠道
    "wechat_token": "",                # 微信登录 token（留空则启动时扫码登录）

    # ===== 调试 =====
    "debug": False,                    # 开启后输出 DEBUG 级别日志
    "request_timeout": 120,            # LLM 请求超时秒数
}


# ── 敏感信息脱敏 ───────────────────────────────────────────────
# 日志打印配置内容时，自动把 api_key / secret 等敏感字段的值用 *** 遮住。
# 这样即使不小心把日志发到公网，密钥也不会泄露。
#
# 脱敏规则：保留前 3 个字符 + "***" + 后 3 个字符
#   例: "sk-abc123def456ghi789" → "sk-***789"
#   例: "a1b2c3"（太短，<=8 字符）→ 原样返回，不脱敏
# ────────────────────────────────────────────────────────────────

def _mask_value(val):
    """
    对单个字符串值脱敏：保留首 3 + "***" + 尾 3。
    长度 <= 8 的短值不做脱敏，因为无意义（截完后全是 ***）。

    入参:
        val: 待脱敏的字符串
    返回:
        脱敏后的字符串（太短则原样返回）
    """
    # 只处理字符串类型，其他类型直接返回
    if not isinstance(val, str):
        return val
    # 太短的值脱敏没意义，比如空字符串或简短的 token
    if len(val) <= 8:
        return val
    # 截取: 前3字符 + 5个星号占位 + 后3字符
    return val[0:3] + "***" + val[-3:]


def _mask_sensitive_keys(obj):
    """
    递归遍历 dict / list，对所有 key 名中包含 "key" 或 "secret" 的字段进行脱敏。
    用于日志打印前对整个 config 字典做安全处理。

    遍历逻辑：
      - dict: 遍历每个 k-v，若 k 名匹配敏感词则脱敏 v，否则递归进入 v
      - list: 对每个元素递归处理
      - 其他类型: 直接返回

    入参:
        obj: dict、list 或其他任意类型
    返回:
        脱敏后的同结构对象（敏感值已被 _mask_value 处理）
    """
    # 字典：逐个键处理
    if isinstance(obj, dict):
        #判断是否为字典类型
        masked = {}
        for k, v in obj.items():
            #items() 是字典 dict 对象的内置方法，返回一个可遍历的键值对视图对象 dict_items
            # 键名包含 "key" 或 "secret"（不区分大小写）→ 脱敏
            k_lower = k.lower()
            if "key" in k_lower or "secret" in k_lower:
                masked[k] = _mask_value(v) if isinstance(v, str) else v
            else:
                # 不敏感 → 递归检查值内部是否嵌套了敏感结构
                masked[k] = _mask_sensitive_keys(v)
        return masked
    # 列表：逐个元素递归
    elif isinstance(obj, list):
        return [_mask_sensitive_keys(item) for item in obj]
    # 标量类型：直接返回
    else:
        return obj


def drag_sensitive(config):
    """
    敏感信息脱敏的外部调用入口。
    支持 string(dict) 和 dict 两种输入格式，返回安全的 JSON 字符串或 dict。

    调用时机：每次需要把 config 内容输出到日志时。

    入参:
        config: JSON 字符串 或 dict
    返回:
        脱敏后的 JSON 字符串（入参是 str 时）或 dict（入参是 dict 时）
    """
    try:
        # 入参是 JSON 字符串 → 先解析，脱敏后重新序列化
        if isinstance(config, str):
            conf_dict = json.loads(config)
            masked = _mask_sensitive_keys(conf_dict)
            return json.dumps(masked, indent=4, ensure_ascii=False)
        # 入参是 dict → 直接脱敏
        elif isinstance(config, dict):
            return _mask_sensitive_keys(config)
    except Exception:
        # 脱敏失败不能影响主流程，原样返回
        pass
    # fallback: 无法处理的情况，原样返回
    return config


# ── Config 类 ──────────────────────────────────────────────────
# 继承 dict，覆写 get() 方法加入白名单校验。
# 全局只存在一个 config 实例（单例模式），通过 conf() 函数获取。
#
# 白名单校验的意义：
#   如果代码里写了 config.get("opeanai_api_key")（拼写错误），
#   直接返回 None 而不是无声地返回空字符串，方便排查 bug。
# ────────────────────────────────────────────────────────────────

class Config(dict):
    """
    全局配置类，继承自 dict。
    通过覆写 get() 实现对 available_setting 白名单的校验。

    用法:
        from config import conf
        api_key = conf().get("openai_api_key")       # 白名单内的键 → 正常取值
        typo   = conf().get("opeanai_api_key")       # 不在白名单 → 总是返回 None
    """

    def __init__(self, d=None):
        """
        从普通 dict 初始化 Config 实例。

        入参:
            d: dict 或 None，用户从 config.json 解析出的配置字典
        """
        # 调用父类 dict.__init__，把自己初始化为一个空字典
        super().__init__()
        # 如果传入了字典，逐个键值对复制进来
        if d is None:
            d = {}
        for k, v in d.items():
            self[k] = v

    def get(self, key, default=None):
        """
        覆写 dict.get()：对不在 available_setting 中的 key，直接返回 default。
        这是整个配置系统的"安全阀门"——任何未在白名单中注册的键都取不到值。

        逻辑分支:
          1. key 以 "_" 开头 → 注释字段，走原生 dict.get
          2. key 不在 available_setting 中 → 返回 default（拦截未注册的键）
          3. key 在白名单中 → 尝试返回 self[key]，KeyError 时返回 default

        入参:
            key: 配置键名
            default: 键不存在时的默认返回值（默认为 None）
        """
        # 分支 1: 下划线开头的是内部注释字段，不归白名单管
        if key.startswith("_"):
            return super().get(key, default)

        # 分支 2: key 不在白名单中 → 可能是拼写错误 → 返回 default
        if key not in available_setting:
            return default

        # 分支 3: 正常取值
        try:
            return self[key]
        except KeyError:
            return default
        except Exception:
            # 其他异常继续往上抛，不在这里吞掉
            raise


# ── 全局单例 ──────────────────────────────────────────────────
# 所有模块通过 conf() 获取同一个 Config 实例，
# 保证整个应用读到的配置是同一份。
# ────────────────────────────────────────────────────────────────

config = Config()


# ── 工具函数 ──────────────────────────────────────────────────
# 全局最常用的几个辅助函数，被所有模块引用。
# ────────────────────────────────────────────────────────────────

def read_file(path):
    """
    以 utf-8-sig 编码读取文件的全部内容。
    使用 utf-8-sig 而非普通 utf-8，是为了自动跳过文件开头的 BOM 标记（某些编辑器会加）。

    入参:
        path: 文件路径
    返回:
        文件的完整文本内容
    """
    with open(path, mode="r", encoding="utf-8-sig") as f:
        return f.read()


def get_root():
    """
    返回 MyAgent 项目的根目录绝对路径。

    原理：本文件（config.py）位于 <项目根>/config.py，
    所以 os.path.dirname(__file__) 就是项目根目录。

    返回:
        项目根目录的绝对路径字符串，如 "/home/user/MyAgent"
    """
    return os.path.dirname(os.path.abspath(__file__))


def conf():
    """
    返回全局 Config 单例。
    所有模块通过 conf().get("xxx") 读取配置，保证一致性。
    如果直接 import config 对象也能用，但用 conf() 更明确（表明这是函数调用而非直接访问变量）。

    返回:
        全局唯一的 Config 实例
    """
    return config


def _parse_env_value(name: str, value: str):
    """根据配置默认值类型安全解析环境变量。"""
    default = available_setting.get(name)
    normalized = value.strip()

    if isinstance(default, bool):
        if normalized.lower() == "true":
            return True
        if normalized.lower() == "false":
            return False
        return value

    if isinstance(default, int):
        try:
            return int(normalized)
        except ValueError:
            return value

    if isinstance(default, float):
        try:
            return float(normalized)
        except ValueError:
            return value

    return value


# ── 配置加载主流程 ───────────────────────────────────────────
# load_config() 是应用启动时第一个被调用的初始化函数。
# 它按顺序完成：读文件 → 解析 JSON → 环境变量覆盖 → 日志输出。
# 任何一步失败均应记录原因并优雅降级（不中断启动）。
#
# 调用时机：app.py 启动时，在创建 Agent / Channel 之前。
# ────────────────────────────────────────────────────────────────

def load_config():
    """
    配置加载主流程。
    按顺序执行以下步骤：
      1. 定位并读取 config.json（不存在则回退到 config-template.json）
      2. 将 JSON 解析为 dict，构建 Config 实例
      3. 遍历环境变量，对白名单中存在的键进行覆盖（env 优先于文件）
      4. 如果 debug 模式开启，切换日志级别到 DEBUG
      5. 输出启动摘要（模型、渠道、Agent 模式等关键信息）
      6. 把 3 个模型提供商的 API 密钥注入子进程环境变量

    无入参，无返回值。所有结果写入全局 config 对象。
    """
    global config

    configure_logging()

    # ── 步骤 1: 定位配置文件 ──
    # 优先使用用户创建的 config.json，不存在则用模板文件兜底
    logger.info("[Init] MyAgent is starting...")
    config_path = os.path.join(get_root(), "config.json")
    if not os.path.exists(config_path):
        logger.info("[Init] config.json not found, falling back to config-template.json")
        config_path = os.path.join(get_root(), "config-template.json")

    # ── 步骤 2: 读取 + 解析 ──
    # 用 read_file 读文件内容（utf-8-sig 编码，自动跳过 BOM）
    config_str = read_file(config_path)
    logger.debug("[Init] Raw config: %s", drag_sensitive(config_str))
    # json.loads 将 JSON 字符串转为 Python dict
    config = Config(json.loads(config_str))

    # ── 步骤 3: 环境变量覆盖 ──
    # 环境变量名是全小写，如 openai_api_key=sk-xxx
    # 只有 available_setting 白名单中的键才会被覆盖
    # eval() 用于自动转换类型：传 "True" → True, "20" → 20
    overridden = []
    for name, value in os.environ.items():
        name = name.lower()
        if name.startswith("_"):
            continue
        if name in available_setting:
            overridden.append(name)
            logger.debug("[Init] Env override: %s", name)
            config[name] = _parse_env_value(name, value)
    if overridden:
        logger.info("[Init] %d config values overridden by environment variables", len(overridden))

    # ── 步骤 4: 日志级别 ──
    # debug 模式下把日志级别从 INFO 降到 DEBUG，方便排查问题
    configure_logging(config.get("debug", False))
    if config.get("debug", False):
        logger.debug("[Init] Log level set to DEBUG")

    # ── 步骤 5: 启动摘要 ──
    # 打印关键配置信息，方便启动后确认一切正常
    # 敏感字段通过 drag_sensitive 脱敏后再打印
    logger.info("[Init] ========================================")
    logger.info("[Init] Configuration Summary")
    logger.info("[Init] ========================================")
    logger.info("[Init] Language:      %s", config.get("language", "auto"))
    logger.info("[Init] Model:         %s", config.get("model", "unknown"))
    logger.info("[Init] Agent Mode:    %s", "ON" if config.get("agent_enabled", True) else "OFF")
    logger.info("[Init] Web Console:   %s:%s", config.get("web_host", "127.0.0.1"),
                config.get("web_port", 9899))
    logger.info("[Init] WeChat:        %s", "ON" if config.get("wechat_enabled", False) else "OFF")
    logger.info("[Init] Memory:        %s", "ON" if config.get("memory_enabled", True) else "OFF")
    logger.info("[Init] Debug:         %s", "ON" if config.get("debug", False) else "OFF")
    logger.info("[Init] ========================================")

    # ── 步骤 6: 注入子进程环境变量 ──
    # 某些工具（如 shell 脚本）或子进程需要读取 API 密钥，
    # 把配置中的密钥同步到 os.environ，子进程就能直接用。
    # 如果环境变量已经存在，就不覆盖（尊重系统已有配置）。
    _CONFIG_TO_ENV = {
        "openai_api_key": "OPENAI_API_KEY",
        "openai_api_base": "OPENAI_API_BASE",
        "claude_api_key": "CLAUDE_API_KEY",
        "claude_api_base": "CLAUDE_API_BASE",
        "deepseek_api_key": "DEEPSEEK_API_KEY",
        "deepseek_api_base": "DEEPSEEK_API_BASE",
    }
    injected = 0
    for conf_key, env_key in _CONFIG_TO_ENV.items():
        if env_key not in os.environ:
            val = config.get(conf_key, "")
            if val:
                os.environ[env_key] = str(val)
                injected += 1
    if injected:
        logger.info("[Init] Injected %d config values into environment variables", injected)

    logger.info("[Init] Configuration loaded successfully.")
