"""Jev 门控预判插件 (Jev Gate Plugin)。"""

from .plugin import JevGatePlugin, create_plugin

MaisakaJevGatePlugin = JevGatePlugin

__all__ = ["JevGatePlugin", "MaisakaJevGatePlugin", "create_plugin"]
