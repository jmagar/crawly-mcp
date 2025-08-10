from fastmcp import Context

# Note: This file is intended to be imported into your main server.py
# from .resources import example_resource_template
#
# Then register it with your MCP instance:
# mcp.add_resource(example_resource_template)


async def example_resource_template(item_id: str, ctx: Context) -> dict:
    """This is an example resource template that provides data for a specific item.

    Args:
        item_id: The unique identifier for the item, extracted from the URI.
        ctx: The MCP Context object.

    Returns:
        A dictionary containing item data.
    """
    await ctx.info(f"Fetching data for resource with item_id: {item_id}")

    # In a real application, you would fetch this data from a database or API
    data = {
        "id": item_id,
        "description": f"This is detailed information for item {item_id}.",
        "timestamp": ctx.request_id,  # Example of using context
    }

    return data
