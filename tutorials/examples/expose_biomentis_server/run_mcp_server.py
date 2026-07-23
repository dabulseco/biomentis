from biomentis.agent.a1 import A1

# Create the agent
agent = A1()

# Create the MCP server
mcp = agent.create_mcp_server(tool_modules=["biomentis.tool.database"])

if __name__ == "__main__":
    # Run the server
    print("Starting Biomentis MCP server...")
    mcp.run(transport="stdio")
