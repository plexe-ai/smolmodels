"""
Application entry point for using the smolmodels package as a conversational agent.
"""

from smolagents import GradioUI

from smolmodels.internal.chat_agents import ChatSmolmodelsAgent


def main():
    """Main function to run the Gradio UI for the Smolmodels conversational agent."""
    agent = ChatSmolmodelsAgent("openai/gpt-4o", verbose=True)
    GradioUI(agent.agent).launch()


if __name__ == "__main__":
    main()
