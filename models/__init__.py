# encoding:utf-8
"""
MyAgent 模型适配层 — 工厂入口
=============================
提供 create_model() 工厂函数，根据 config.json 中的 model 字段创建对应的适配器实例。

本实现用 dict 映射，新增模型只需加一行。

用法:
    from models import create_model
    llm = create_model()  # 自动读取 config.json 决定用哪个模型
    result = llm.chat(messages=[{"role":"user","content":"你好"}])
"""

from config import conf

from models.openai import OpenAIAdapter
from models.claude import ClaudeAdapter
from models.deepseek import DeepSeekAdapter


def create_model(model_name: str = None):
    """
    根据模型名创建对应的 LLM 适配器实例。

    从 config.json 中读取对应模型的 api_key / api_base / model 配置，
    传入适配器的构造函数。

    入参:
        model_name: 可选，指定模型名（"openai" / "claude" / "deepseek"）。
                    为空时使用 config.json 中 "model" 字段的值。

    返回:
        BaseLLM 子类实例（OpenAIAdapter / ClaudeAdapter / DeepSeekAdapter）

    扩展方式:
        新增模型只需两步:
        1. 创建 models/newmodel.py，继承 BaseLLM
        2. 在 _FACTORY dict 中加一行: "newmodel": (NewModelAdapter, "newmodel_api_key", "newmodel_api_base")
    """
    # 未指定模型时，使用 config.json 中的默认值
    if model_name is None:
        model_name = conf().get("model", "openai")

    # ── 模型注册表 ──────────────────────────────────────────
    # 每个模型对应: (适配器类, api_key 配置键, api_base 配置键)
    # 新增模型时只需要在这里加一行，无需修改 create_model 的逻辑
    _FACTORY = {
        "openai": (OpenAIAdapter, "openai_api_key", "openai_api_base", "openai_model"),
        "claude": (ClaudeAdapter, "claude_api_key", "claude_api_base", "claude_model"),
        "deepseek": (DeepSeekAdapter, "deepseek_api_key", "deepseek_api_base", "deepseek_model"),
    }

    # 获取该模型的注册信息
    entry = _FACTORY.get(model_name)
    if entry is None:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available models: {list(_FACTORY.keys())}"
        )

    adapter_cls, key_config, base_config, model_config = entry

    # 从 config.json 读取该模型专属的 api_key / api_base / model
    api_key = conf().get(key_config, "")
    api_base = conf().get(base_config, "")
    model = conf().get(model_config, "")

    # 检查必填项
    if not api_key:
        raise ValueError(
            f"API key not configured for model '{model_name}'. "
            f"Please set '{key_config}' in config.json"
        )

    # 从 config.json 读取通用参数
    proxy = conf().get("proxy", "") or None  # 空字符串转为 None
    temperature = conf().get("agent_temperature", 0.7)
    top_p = conf().get("agent_top_p", 0.9)
    request_timeout = conf().get("request_timeout", 120)

    # 构建模型专属的额外配置
    extra_config = {
        "temperature": temperature,
        "top_p": top_p,
        "request_timeout": request_timeout,
    }

    # Claude 需要 max_tokens 默认值
    if model_name == "claude" or model_name == "deepseek":
        extra_config["max_tokens"] = 4096

    if model_name == "deepseek":
        thinking_enabled = conf().get("deepseek_thinking_enabled", False)
        extra_config["thinking"] = {
            "type": "enabled" if thinking_enabled else "disabled",
        }
        extra_config["reasoning_effort"] = conf().get(
            "deepseek_reasoning_effort", "high",
        )

    return adapter_cls(
        api_key=api_key,
        api_base=api_base,
        model=model,
        config=extra_config,
        proxy=proxy,
    )
