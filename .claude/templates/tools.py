from fastmcp import Context

# Note: This file is intended to be imported into your main server.py
# from .tools import example_tool
#
# Then register it with your MCP instance:
# mcp.add_tool(example_tool)


async def example_tool(query: str, ctx: Context) -> dict:
    """This is an example tool that takes a query and uses the context.

    Args:
        query: The input query string.
        ctx: The MCP Context object for logging and other operations.

    Returns:
        A dictionary containing the processed query.
    """
    await ctx.info(f"Received query: {query}")

    # Your tool logic here
    processed_query = query.upper()

    await ctx.debug("Query processing complete.")

    return {"original_query": query, "processed_query": processed_query}
