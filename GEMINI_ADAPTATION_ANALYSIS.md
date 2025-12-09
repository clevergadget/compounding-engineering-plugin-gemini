# Gemini Adaptation Analysis

## What This Plugin Actually Does (Code Level)

The Compounding Engineering plugin is a **Claude Code plugin** that implements the philosophy that "each unit of engineering work should make subsequent units of work easier."

### Core Philosophy

The plugin creates a development loop:
1. **Plan** → Research codebase, create detailed issues with acceptance criteria
2. **Work** → Execute plans systematically with isolated git worktrees
3. **Review** → Multi-agent code reviews with 12+ specialized reviewers
4. **Codify** → Document learnings to compound knowledge

### Technical Architecture

The plugin consists of:

| Component | Count | Location | Purpose |
|-----------|-------|----------|---------|
| **Agents** | 24 | `agents/` | AI personas with domain expertise |
| **Commands** | 16 | `commands/` | Slash commands for workflows |
| **Skills** | 11 | `skills/` | Modular capabilities with scripts |
| **MCP Servers** | 2 | `plugin.json` | External tool integrations |

#### 1. Agents (24 total)

Agents are markdown files with YAML frontmatter defining:
- `name`: Agent identifier
- `description`: When to use (includes examples with `<example>` tags)
- Body: System prompt defining the agent's behavior/expertise

**Example agent structure:**
```yaml
---
name: kieran-rails-reviewer
description: Use this agent when you need to review Rails code...
---

You are Kieran, a super senior Rails developer...

## 1. EXISTING CODE MODIFICATIONS - BE VERY STRICT
...
```

**Agent categories:**
- **Review agents** (11): Rails, TypeScript, Python code reviewers, security sentinel, performance oracle, architecture strategist, data integrity guardian
- **Research agents** (4): Best practices, framework docs, git history, repo analyst  
- **Design agents** (3): Design implementation reviewer, iterator, Figma sync
- **Workflow agents** (5): Style editor, bug validator, PR comment resolver, lint, spec-flow
- **Docs agents** (1): Ankane-style README writer

#### 2. Commands (16 total)

Commands are markdown files with YAML frontmatter:
- `name`: Command identifier (used as `/name`)
- `description`: What it does
- `argument-hint`: Expected arguments
- `allowed-tools`: (optional) Tool permissions
- Body: Instructions executed when command is invoked

**Key commands:**
- `/plan` - Research + create GitHub issues from feature descriptions
- `/work` - Execute work plans with git worktrees and todos
- `/review` - Multi-agent parallel code reviews
- `/triage` - Convert findings into trackable todos
- `/resolve_todo_parallel` - Fix todos in parallel
- `/resolve_pr_parallel` - Resolve PR comments in parallel

**Claude Code-specific features used:**
- `@agent-name` syntax for invoking agents
- `Task agent-name(context)` for parallel agent execution
- `Skill(skill-name)` for invoking skills
- `/model` command for switching models
- `TodoWrite` for todo tracking
- MCP server integration

#### 3. Skills (11 total)

Skills are directories with:
- `SKILL.md` - Main skill definition (always loaded)
- `workflows/` - Step-by-step procedures
- `references/` - Domain knowledge
- `templates/` - Output structures
- `scripts/` - Executable code (Python, etc.)

**Key skills:**
- `gemini-imagegen` - Image generation using Gemini API (already uses Gemini!)
- `create-agent-skills` - Guide for creating skills
- `git-worktree` - Isolated development environments
- `file-todos` - Todo tracking in filesystem
- `compound-docs` - Document learnings

#### 4. MCP Servers (2)

Defined in `plugin.json`:
```json
"mcpServers": {
  "playwright": {
    "type": "stdio",
    "command": "npx",
    "args": ["-y", "@playwright/mcp@latest"]
  },
  "context7": {
    "type": "http",
    "url": "https://mcp.context7.com/mcp"
  }
}
```

---

## Difficulty Assessment: Adapting for Gemini

### Challenge Level: **MODERATE TO HIGH**

The main challenge isn't the prompts (those are mostly LLM-agnostic) - it's the **infrastructure**.

### What's Claude-Specific (Hard to Port)

| Feature | Usage | Gemini Equivalent |
|---------|-------|-------------------|
| **Plugin System** | Entire architecture | ❌ None (would need custom implementation) |
| **Slash Commands** | 16 commands | ❌ None (would need CLI/API wrapper) |
| **Agent System** | `@agent-name` syntax | ❌ None (would need prompt chaining) |
| **Parallel Tasks** | `Task agent(context)` | ❌ None (would need orchestration) |
| **Skills** | `Skill(name)` invocation | ❌ None (would need custom loader) |
| **MCP Servers** | Tool integrations | ⚠️ Partial (Gemini has different function calling) |
| **TodoWrite** | Progress tracking | ❌ None (custom implementation needed) |
| `/model` command | Model switching | ⚠️ Different API (Gemini model selection) |

### What's LLM-Agnostic (Easy to Port)

| Feature | Notes |
|---------|-------|
| **Agent prompts** | Just system prompts - work with any LLM |
| **Review criteria** | Domain knowledge in markdown |
| **Skill documentation** | Reference materials, templates |
| **Workflow instructions** | Step-by-step procedures |
| **Python scripts** | Gemini-imagegen already uses Gemini API |

### Lines of Claude References

```
166 occurrences of "Claude" in markdown/json files
```

Most are in:
- `create-agent-skills/references/` - Skill authoring guides that reference "Claude"
- Documentation about "Claude Code" features
- A few in agent descriptions

---

## Adaptation Options

### Option 1: Extract Prompts for Direct Gemini API Use
**Effort: Low | Value: Medium**

Extract the agent system prompts and use them directly with Gemini's API:

```python
import google.generativeai as genai

# Load agent prompt from markdown
with open('agents/review/kieran-rails-reviewer.md') as f:
    agent_prompt = extract_body(f.read())

model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(system_instruction=agent_prompt)
response = chat.send_message("Review this code: ...")
```

**Pros:**
- Quick to implement
- Gets the domain expertise
- No infrastructure needed

**Cons:**
- Loses the orchestration (parallel agents, workflows)
- Manual invocation
- No todo tracking, worktree management, etc.

### Option 2: Build a Gemini Code Agent
**Effort: High | Value: High**

Create equivalent infrastructure for Gemini:

1. **Command Handler** - Parse slash commands, route to handlers
2. **Agent Orchestrator** - Load agent prompts, manage conversations
3. **Parallel Executor** - Run multiple agents concurrently
4. **Skill Loader** - Load skills, execute scripts
5. **Todo System** - Track progress in filesystem
6. **Git Integration** - Worktree management, PR operations

This would essentially be building "Gemini Code" as an open-source project.

**Pros:**
- Full feature parity
- Could be a valuable open-source tool
- Works with any Gemini-based development

**Cons:**
- Significant engineering effort (weeks/months)
- Needs ongoing maintenance
- May be superseded by Google's own tooling

### Option 3: Wait for Google's Equivalent
**Effort: None | Value: Unknown**

Google may release a "Gemini Code" or similar product with plugin support.

**Pros:**
- No work required
- Would have official support

**Cons:**
- Unknown timeline
- May never happen
- Plugin format may differ

### Option 4: Use Existing Gemini Tools + Prompts
**Effort: Low-Medium | Value: Medium**

Use tools like:
- **AI Studio** - For agent prompt testing
- **Vertex AI Agent Builder** - For building conversational agents
- **Custom Python scripts** - For workflow automation

Import the agent prompts as system instructions.

---

## Recommendation

### For Immediate Use

**Option 1 (Extract Prompts)** is the fastest path:

1. The `gemini-imagegen` skill already shows Gemini integration works
2. Create Python scripts that:
   - Load agent prompts from markdown
   - Use Gemini API with those prompts
   - Handle common workflows (code review, planning)

### For Long-term Value

**Option 2 (Build Infrastructure)** if you're committed:

1. Start with a minimal CLI tool
2. Implement one workflow end-to-end (e.g., `/review`)
3. Add features incrementally
4. Consider open-sourcing for community contribution

---

## Quick Start: Using Agent Prompts with Gemini

Here's how to use an agent prompt directly:

```python
#!/usr/bin/env python3
"""Use a compounding-engineering agent with Gemini."""

import google.generativeai as genai
import re
import os

def load_agent(agent_path: str) -> str:
    """Extract system prompt from agent markdown file."""
    with open(agent_path) as f:
        content = f.read()
    
    # Split on the closing --- of frontmatter
    parts = content.split('---', 2)
    if len(parts) >= 3:
        return parts[2].strip()
    return content

def run_review(agent_name: str, code_to_review: str):
    """Run a code review using the specified agent."""
    genai.configure(api_key=os.environ['GEMINI_API_KEY'])
    
    # Load agent prompt
    agent_path = f'plugins/compounding-engineering/agents/review/{agent_name}.md'
    system_prompt = load_agent(agent_path)
    
    # Create model with agent as system instruction
    model = genai.GenerativeModel(
        'gemini-pro',
        system_instruction=system_prompt
    )
    
    # Run review
    response = model.generate_content(
        f"Review this code:\n\n```\n{code_to_review}\n```"
    )
    
    return response.text

# Example usage
if __name__ == '__main__':
    code = '''
    class UsersController < ApplicationController
      def update
        @user = User.find(params[:id])
        @user.update!(user_params)
        render turbo_stream: turbo_stream.replace(@user)
      end
    end
    '''
    
    print(run_review('kieran-rails-reviewer', code))
```

This gives you the agent expertise without the full Claude Code infrastructure.

---

## Conclusion

The plugin's **value is in the prompts and workflows**, not the infrastructure. The infrastructure is Claude Code-specific, but the domain expertise (how to review Rails code, security patterns, etc.) transfers to any LLM.

**Bottom line:** You can use the ideas and prompts with Gemini, but you'll need to build your own orchestration layer or use them manually.
