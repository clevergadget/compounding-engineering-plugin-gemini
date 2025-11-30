#!/usr/bin/env python3
"""
Use compounding-engineering agent prompts with Google's Gemini API.

This demonstrates how to extract the domain expertise from the Claude Code
plugin and use it with Gemini instead.

Requirements:
    pip install google-generativeai

Usage:
    export GEMINI_API_KEY='your-api-key'
    python gemini_agent_bridge.py review kieran-rails-reviewer code.rb
    python gemini_agent_bridge.py review security-sentinel code.py
    python gemini_agent_bridge.py list  # List available agents
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai not installed.")
    print("Run: pip install google-generativeai")
    sys.exit(1)


def get_plugin_root() -> Path:
    """Get the plugin root directory."""
    # Navigate from examples/ up to repo root, then into plugins/
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    plugin_root = repo_root / "plugins" / "compounding-engineering"
    
    if not plugin_root.exists():
        # Try relative to current directory
        plugin_root = Path("plugins/compounding-engineering")
    
    return plugin_root


def load_agent_prompt(agent_path: Path) -> tuple[str, str]:
    """
    Load an agent markdown file and extract the system prompt.
    
    Returns:
        tuple: (name, system_prompt)
    """
    content = agent_path.read_text()
    
    # Parse YAML frontmatter
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            frontmatter = parts[1]
            body = parts[2].strip()
            
            # Extract name from frontmatter
            name = None
            for line in frontmatter.strip().split('\n'):
                if line.startswith('name:'):
                    name = line.split(':', 1)[1].strip()
                    break
            
            return name or agent_path.stem, body
    
    return agent_path.stem, content


def list_agents(plugin_root: Path) -> dict[str, list[tuple[str, Path]]]:
    """List all available agents by category."""
    agents_dir = plugin_root / "agents"
    
    if not agents_dir.exists():
        return {}
    
    agents = {}
    for category_dir in agents_dir.iterdir():
        if category_dir.is_dir():
            category_agents = []
            for agent_file in category_dir.glob("*.md"):
                name, _ = load_agent_prompt(agent_file)
                category_agents.append((name, agent_file))
            if category_agents:
                agents[category_dir.name] = sorted(category_agents)
    
    return agents


def find_agent(plugin_root: Path, agent_name: str) -> Path | None:
    """Find an agent file by name."""
    agents_dir = plugin_root / "agents"
    
    # Search all categories
    for category_dir in agents_dir.iterdir():
        if category_dir.is_dir():
            # Try exact match
            agent_file = category_dir / f"{agent_name}.md"
            if agent_file.exists():
                return agent_file
    
    return None


def run_agent(
    agent_prompt: str,
    user_message: str,
    model_name: str = "gemini-2.0-flash",
) -> str:
    """
    Run an agent with the given prompt and user message.
    
    Args:
        agent_prompt: The system prompt from the agent markdown file
        user_message: The user's input/request
        model_name: Gemini model to use
        
    Returns:
        The model's response text
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable not set.\n"
            "Get an API key at: https://makersuite.google.com/app/apikey"
        )
    
    genai.configure(api_key=api_key)
    
    # Create model with agent prompt as system instruction
    model = genai.GenerativeModel(
        model_name,
        system_instruction=agent_prompt
    )
    
    # Generate response
    response = model.generate_content(user_message)
    
    return response.text


def cmd_list(args, plugin_root: Path):
    """List all available agents."""
    agents = list_agents(plugin_root)
    
    if not agents:
        print("No agents found.")
        return
    
    print("Available Agents")
    print("=" * 50)
    
    for category, agent_list in sorted(agents.items()):
        print(f"\n{category.upper()}:")
        for name, path in agent_list:
            print(f"  - {name}")
    
    print("\n" + "=" * 50)
    print("Usage: python gemini_agent_bridge.py review <agent-name> <file>")


def cmd_review(args, plugin_root: Path):
    """Run a code review using an agent."""
    agent_path = find_agent(plugin_root, args.agent)
    
    if not agent_path:
        print(f"Error: Agent '{args.agent}' not found.")
        print("Run with 'list' to see available agents.")
        sys.exit(1)
    
    # Load code to review
    if args.file == '-':
        code = sys.stdin.read()
    else:
        code_path = Path(args.file)
        if not code_path.exists():
            print(f"Error: File '{args.file}' not found.")
            sys.exit(1)
        code = code_path.read_text()
    
    # Load agent
    name, prompt = load_agent_prompt(agent_path)
    
    print(f"🔍 Running {name} review...")
    print("-" * 50)
    
    # Build user message
    user_message = f"Please review this code:\n\n```\n{code}\n```"
    
    if args.context:
        user_message = f"Context: {args.context}\n\n{user_message}"
    
    try:
        result = run_agent(prompt, user_message, args.model)
        print(result)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_ask(args, plugin_root: Path):
    """Ask an agent a question (no code review)."""
    agent_path = find_agent(plugin_root, args.agent)
    
    if not agent_path:
        print(f"Error: Agent '{args.agent}' not found.")
        print("Run with 'list' to see available agents.")
        sys.exit(1)
    
    name, prompt = load_agent_prompt(agent_path)
    
    print(f"🤖 Asking {name}...")
    print("-" * 50)
    
    try:
        result = run_agent(prompt, args.question, args.model)
        print(result)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Use compounding-engineering agents with Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available agents")
    
    # Review command
    review_parser = subparsers.add_parser("review", help="Review code with an agent")
    review_parser.add_argument("agent", help="Agent name (e.g., kieran-rails-reviewer)")
    review_parser.add_argument("file", help="File to review (or - for stdin)")
    review_parser.add_argument(
        "--context", "-c",
        help="Additional context for the review"
    )
    review_parser.add_argument(
        "--model", "-m",
        default="gemini-2.0-flash",
        help="Gemini model to use (default: gemini-2.0-flash)"
    )
    
    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask an agent a question")
    ask_parser.add_argument("agent", help="Agent name")
    ask_parser.add_argument("question", help="Question to ask")
    ask_parser.add_argument(
        "--model", "-m",
        default="gemini-2.0-flash",
        help="Gemini model to use (default: gemini-2.0-flash)"
    )
    
    args = parser.parse_args()
    
    plugin_root = get_plugin_root()
    
    if not plugin_root.exists():
        print(f"Error: Plugin not found at {plugin_root}")
        print("Make sure you're running from the repository root.")
        sys.exit(1)
    
    if args.command == "list":
        cmd_list(args, plugin_root)
    elif args.command == "review":
        cmd_review(args, plugin_root)
    elif args.command == "ask":
        cmd_ask(args, plugin_root)


if __name__ == "__main__":
    main()
