from typing import List
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("weather")

@mcp.tool()
async def get_weather(location: str) -> str:
  """
  Get the weather for location.
  """

  return "It's always sunny in New York."


if __name__ == "__main__":
  mcp.run(transport="streamable-http")