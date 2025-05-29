"""Data structures for MCP Knowledge Store"""

from pydantic import BaseModel


class MCPServerMetadata(BaseModel):
    name: str
