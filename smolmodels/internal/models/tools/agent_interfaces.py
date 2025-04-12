"""
This module provides a tool that helps the smolagents manager agent understand what information it needs
to provide to its sub-agents in order for them to work effectively. In that sense, this module helps define the
'interface' between the manager agent and its sub-agents.
"""

from smolagents import tool


@tool
def get_agent_instruction_requirements(agent_name: str) -> str:
    pass
