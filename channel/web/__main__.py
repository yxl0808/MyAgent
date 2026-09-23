"""MyAgent Web Console 的本机服务器启动入口。"""

import ipaddress
from typing import Any, Mapping

import uvicorn

from channel.web.app import create_web_app
from config import conf, load_config


def validate_web_settings(values: Mapping[str, Any]) -> tuple[str, int]:
    """校验 Web 启用状态、端口范围和仅本机监听限制。"""
    if not values.get("web_enabled", True):
        raise ValueError("Web Console 已通过 web_enabled 禁用")
    host = str(values.get("web_host", "127.0.0.1")).strip()
    port = int(values.get("web_port", 9899))
    if not 1 <= port <= 65535:
        raise ValueError("web_port 必须在 1 到 65535 之间")
    local_host = host.lower() == "localhost"
    try:
        local_host = local_host or ipaddress.ip_address(host).is_loopback
    except ValueError:
        local_host = False
    if not local_host:
        raise ValueError("Web Console 仅支持 localhost 或回环地址")
    return host, port


def main() -> int:
    """加载配置并启动独立的本机 Web Console 服务。"""
    load_config()
    values = dict(conf())
    host, port = validate_web_settings(values)
    uvicorn.run(create_web_app(values), host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
