from mcp.server.fastmcp import FastMCP


mcp = FastMCP("math")

@mcp.tool()
async def add(a: int, b: int) -> int:
  """
  Add two numbers
  """

  return a + b


@mcp.tool()
async def multiply(a: int, b: int) -> int:
  """
  Multiply two numbers
  """

  return a * b


if __name__ == "__main__":
  mcp.run(transport="stdio")