---
allowed-tools: Bash
argument-hint: <tool_name> [json_params]
description: Quickly test a specific FastMCP tool.
---
Running `fastmcp inspect call-tool` for tool: $ARGUMENTS

!bash -c "
  TOOL_NAME=$(echo \"$ARGUMENTS\" | awk '{print $1}')
  JSON_PARAMS=$(echo \"$ARGUMENTS\" | cut -d' ' -f2-)

  if [ -z \"$JSON_PARAMS\" ]; then
    fastmcp inspect call-tool \"$TOOL_NAME\" -- uv run python project_name/server.py
  else
    fastmcp inspect call-tool \"$TOOL_NAME\" --params \"$JSON_PARAMS\" -- uv run python project_name/server.py
  fi
"

