# Gemini Compounding Engineering: Implementation Plan

A detailed, actionable roadmap for reimplementing the compounding-engineering plugin for Google's Gemini.

## Overview

This plan creates **"Gemini Code Agent" (GCA)** - an open-source CLI tool that brings compounding engineering workflows to Gemini users.

**Key Philosophy:** Agents are used for analysis and recommendations **BEFORE** and **AFTER** work is done, but **NOT** for executing the work itself. Work execution is performed by humans or other tools, while agents provide insights, planning, and validation.

### Target Architecture

```
gemini-code-agent/
├── gca                           # CLI entry point
├── src/
│   ├── core/
│   │   ├── agent_loader.py       # Load agent prompts from markdown
│   │   ├── gemini_client.py      # Gemini API wrapper
│   │   └── config.py             # Configuration management
│   ├── commands/
│   │   ├── plan.py               # /plan implementation
│   │   ├── work.py               # /work implementation
│   │   ├── review.py             # /review implementation
│   │   └── triage.py             # /triage implementation
│   ├── tools/
│   │   ├── git_worktree.py       # Git worktree management
│   │   ├── file_todos.py         # Todo tracking
│   │   ├── github_api.py         # GitHub integration (issues, PRs)
│   │   └── file_ops.py           # File read/write operations
│   └── agents/                   # Imported from plugin (24 agents)
├── tests/
├── requirements.txt
└── README.md
```

---

## Stage 1: Foundation (Week 1-2)
**Goal:** Basic CLI with agent loading and Gemini integration

### 1.1 Project Setup
**Estimated Time:** 2-4 hours

```bash
# Create project structure
mkdir -p gemini-code-agent/src/{core,commands,tools}
cd gemini-code-agent

# Initialize Python project
python -m venv venv
source venv/bin/activate
pip install google-genai click rich pyyaml

# Create pyproject.toml for packaging
```

**Deliverables:**
- [ ] Project directory structure
- [ ] `pyproject.toml` with dependencies
- [ ] Basic `README.md`
- [ ] `.gitignore`

### 1.2 Agent Loader
**Estimated Time:** 4-6 hours

Create `src/core/agent_loader.py`:

```python
"""
Load and parse agent definitions from markdown files.
"""
import yaml
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Agent:
    name: str
    description: str
    system_prompt: str
    category: str
    
    @classmethod
    def from_markdown(cls, path: Path) -> 'Agent':
        """Load agent from markdown file with YAML frontmatter."""
        content = path.read_text()
        
        # Parse frontmatter
        if content.startswith('---'):
            _, frontmatter, body = content.split('---', 2)
            meta = yaml.safe_load(frontmatter)
        else:
            meta = {}
            body = content
            
        return cls(
            name=meta.get('name', path.stem),
            description=meta.get('description', ''),
            system_prompt=body.strip(),
            category=path.parent.name
        )

class AgentRegistry:
    """Registry of all available agents."""
    
    def __init__(self, agents_dir: Path):
        self.agents: dict[str, Agent] = {}
        self._load_agents(agents_dir)
    
    def _load_agents(self, agents_dir: Path):
        """Load all agents from directory."""
        for category_dir in agents_dir.iterdir():
            if category_dir.is_dir():
                for agent_file in category_dir.glob('*.md'):
                    agent = Agent.from_markdown(agent_file)
                    self.agents[agent.name] = agent
    
    def get(self, name: str) -> Agent | None:
        return self.agents.get(name)
    
    def list_by_category(self) -> dict[str, list[Agent]]:
        """Group agents by category."""
        by_category = {}
        for agent in self.agents.values():
            by_category.setdefault(agent.category, []).append(agent)
        return by_category
```

**Deliverables:**
- [ ] `Agent` dataclass
- [ ] `AgentRegistry` class
- [ ] Unit tests for agent loading
- [ ] Import all 24 agents from plugin

### 1.3 Gemini Client Wrapper
**Estimated Time:** 4-6 hours

Create `src/core/gemini_client.py`:

```python
"""
Gemini API client with conversation management.
"""
import os
from google import genai
from dataclasses import dataclass
from typing import Iterator

@dataclass
class GeminiConfig:
    api_key: str
    model: str = "gemini-2.0-flash"
    temperature: float = 0.7
    max_tokens: int = 8192

class GeminiClient:
    """Wrapper for Gemini API with agent support."""
    
    def __init__(self, config: GeminiConfig):
        genai.configure(api_key=config.api_key)
        self.config = config
        
    def create_agent_session(self, system_prompt: str) -> 'AgentSession':
        """Create a new agent session with system prompt."""
        model = genai.GenerativeModel(
            self.config.model,
            system_instruction=system_prompt
        )
        return AgentSession(model, self.config)
    
    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Single-turn generation."""
        model = genai.GenerativeModel(
            self.config.model,
            system_instruction=system_prompt
        )
        response = model.generate_content(prompt)
        return response.text

class AgentSession:
    """Manages multi-turn conversation with an agent."""
    
    def __init__(self, model, config: GeminiConfig):
        self.model = model
        self.config = config
        self.chat = model.start_chat()
        
    def send(self, message: str) -> str:
        """Send message and get response."""
        response = self.chat.send_message(message)
        return response.text
    
    def stream(self, message: str) -> Iterator[str]:
        """Stream response chunks."""
        response = self.chat.send_message(message, stream=True)
        for chunk in response:
            yield chunk.text
```

**Deliverables:**
- [ ] `GeminiConfig` dataclass
- [ ] `GeminiClient` class
- [ ] `AgentSession` for multi-turn conversations
- [ ] Streaming support
- [ ] Unit tests

### 1.4 Basic CLI
**Estimated Time:** 4-6 hours

Create `src/cli.py`:

```python
"""
Main CLI entry point using Click.
"""
import click
from rich.console import Console
from rich.table import Table
from pathlib import Path

from core.agent_loader import AgentRegistry
from core.gemini_client import GeminiClient, GeminiConfig

console = Console()

@click.group()
@click.option('--agents-dir', default='./agents', help='Path to agents directory')
@click.pass_context
def cli(ctx, agents_dir):
    """Gemini Code Agent - Compounding Engineering for Gemini"""
    ctx.ensure_object(dict)
    ctx.obj['registry'] = AgentRegistry(Path(agents_dir))
    ctx.obj['client'] = GeminiClient(GeminiConfig(
        api_key=os.environ.get('GEMINI_API_KEY', '')
    ))

@cli.command()
@click.pass_context
def list_agents(ctx):
    """List all available agents."""
    registry = ctx.obj['registry']
    
    table = Table(title="Available Agents")
    table.add_column("Category", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Description")
    
    for category, agents in registry.list_by_category().items():
        for agent in agents:
            table.add_row(category, agent.name, agent.description[:60] + "...")
    
    console.print(table)

@cli.command()
@click.argument('agent_name')
@click.argument('message')
@click.pass_context
def ask(ctx, agent_name, message):
    """Ask an agent a question."""
    registry = ctx.obj['registry']
    client = ctx.obj['client']
    
    agent = registry.get(agent_name)
    if not agent:
        console.print(f"[red]Agent '{agent_name}' not found[/red]")
        return
    
    console.print(f"[cyan]Asking {agent.name}...[/cyan]")
    response = client.generate(message, agent.system_prompt)
    console.print(response)

if __name__ == '__main__':
    cli()
```

**Deliverables:**
- [ ] Click-based CLI framework
- [ ] `list` command
- [ ] `ask` command
- [ ] Rich console output
- [ ] Environment variable handling

### Stage 1 Success Criteria
- [ ] Can list all 24 agents
- [ ] Can ask any agent a question and get response
- [ ] Multi-turn conversations work
- [ ] Proper error handling
- [ ] Clear that agents provide analysis, not execution

---

## Stage 2: Core Workflows (Week 3-4)
**Goal:** Implement /review command for pre/post-work analysis

### 2.1 Sequential Agent Execution
**Estimated Time:** 4-6 hours

Agents provide analysis and recommendations before and after work, but do not execute changes themselves.

Key principles:
- Agents analyze code and provide insights
- Agents generate recommendations and todos
- Agents do NOT make code changes directly
- Work execution happens separately via human or automated processes

Create simple sequential agent runner in `src/core/agent_runner.py`:

```python
"""
Run agents sequentially for analysis and recommendations.
"""
from dataclasses import dataclass

@dataclass
class AgentResult:
    agent_name: str
    output: str
    success: bool
    error: str | None = None

class AgentRunner:
    """Execute agents sequentially for analysis."""
    
    def __init__(self, registry, client):
        self.registry = registry
        self.client = client
        
    def run_agent(self, agent_name: str, context: str) -> AgentResult:
        """Run a single agent for analysis."""
        agent = self.registry.get(agent_name)
        if not agent:
            return AgentResult(agent_name, "", False, f"Agent not found")
        
        try:
            result = self.client.generate(context, agent.system_prompt)
            return AgentResult(agent_name, result, True)
        except Exception as e:
            return AgentResult(agent_name, "", False, str(e))
    
    def run_sequence(self, agent_names: list[str], context: str) -> list[AgentResult]:
        """Run multiple agents sequentially."""
        results = []
        for agent_name in agent_names:
            result = self.run_agent(agent_name, context)
            results.append(result)
        return results
```

**Deliverables:**
- [ ] `AgentRunner` class for sequential execution
- [ ] Error handling per agent
- [ ] Clear separation: agents analyze, don't execute
- [ ] Unit tests

### 2.2 Git Integration
**Estimated Time:** 6-8 hours

Create `src/tools/git_worktree.py`:

```python
"""
Git worktree management for isolated development.
"""
import subprocess
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Worktree:
    path: Path
    branch: str
    is_main: bool = False

class GitWorktreeManager:
    """Manage git worktrees for isolated development."""
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        
    def create(self, branch: str, base: str = "main") -> Worktree:
        """Create a new worktree for a branch."""
        worktree_path = self.repo_root / ".worktrees" / branch
        worktree_path.parent.mkdir(exist_ok=True)
        
        # Create branch if it doesn't exist
        subprocess.run(
            ["git", "branch", branch, base],
            cwd=self.repo_root, 
            capture_output=True
        )
        
        # Create worktree
        subprocess.run(
            ["git", "worktree", "add", str(worktree_path), branch],
            cwd=self.repo_root,
            check=True
        )
        
        return Worktree(worktree_path, branch)
    
    def list(self) -> list[Worktree]:
        """List all worktrees."""
        result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=self.repo_root,
            capture_output=True,
            text=True
        )
        # Parse output...
        
    def remove(self, branch: str):
        """Remove a worktree."""
        subprocess.run(
            ["git", "worktree", "remove", branch],
            cwd=self.repo_root,
            check=True
        )
```

**Deliverables:**
- [ ] `GitWorktreeManager` class
- [ ] Create/list/remove worktrees
- [ ] Branch management
- [ ] Integration tests

### 2.3 /review Command
**Estimated Time:** 8-12 hours

Create `src/commands/review.py`:

```python
"""
Code review command using agents for analysis.
"""
from dataclasses import dataclass
from pathlib import Path
import subprocess

from core.agent_runner import AgentRunner
from tools.git_worktree import GitWorktreeManager

# Review agents to run for analysis
REVIEW_AGENTS = [
    "kieran-rails-reviewer",
    "security-sentinel", 
    "performance-oracle",
    "architecture-strategist",
    "code-simplicity-reviewer",
    "data-integrity-guardian",
    "pattern-recognition-specialist",
]

@dataclass  
class ReviewTarget:
    """What we're reviewing."""
    pr_number: int | None
    branch: str
    files: list[str]
    diff: str

class ReviewCommand:
    """Implementation of /review command."""
    
    def __init__(self, agent_runner: AgentRunner, worktree_mgr: GitWorktreeManager):
        self.agent_runner = agent_runner
        self.worktree_mgr = worktree_mgr
        
    def get_review_target(self, target: str) -> ReviewTarget:
        """Parse review target (PR number, branch, or URL)."""
        # Parse target into ReviewTarget...
        
    def get_diff(self, target: ReviewTarget) -> str:
        """Get the diff for the review target."""
        result = subprocess.run(
            ["git", "diff", f"main...{target.branch}"],
            capture_output=True,
            text=True
        )
        return result.stdout
        
    def run(self, target: str) -> list[dict]:
        """Execute the review analysis (BEFORE work is merged)."""
        review_target = self.get_review_target(target)
        diff = self.get_diff(review_target)
        
        # Build context for agents
        context = f"""
## Review Target
Branch: {review_target.branch}
Files Changed: {len(review_target.files)}

## Diff
```diff
{diff}
```

Please review this code change and provide your analysis.
Focus on identifying issues, risks, and recommendations.
Do NOT make code changes - only provide analysis.
"""
        
        # Run review agents sequentially to analyze the code
        results = self.agent_runner.run_sequence(REVIEW_AGENTS, context)
        
        # Aggregate and return findings
        return self._aggregate_findings(results)
    
    def _aggregate_findings(self, results) -> list[dict]:
        """Combine findings from all agents."""
        findings = []
        for result in results:
            if result.success:
                findings.append({
                    "agent": result.agent_name,
                    "analysis": result.output,
                    "recommendations": "Extract from output"
                })
        return findings
```

**Deliverables:**
- [ ] `ReviewCommand` class
- [ ] PR/branch target parsing
- [ ] Sequential agent execution for analysis
- [ ] Finding aggregation (recommendations, not changes)
- [ ] CLI integration: `gca review [PR#]`

### Stage 2 Success Criteria
- [ ] `/review` command analyzes PRs before merging
- [ ] Review agents provide analysis and recommendations
- [ ] Results aggregated and displayed clearly
- [ ] Worktree creation works for isolated analysis
- [ ] Agents do NOT execute changes - only analyze

---

## Stage 3: Todo System & Triage (Week 5-6)
**Goal:** Implement todo tracking and triage workflow

### 3.1 File-based Todo System
**Estimated Time:** 8-12 hours

Create `src/tools/file_todos.py`:

```python
"""
File-based todo tracking system.
"""
import yaml
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class TodoStatus(Enum):
    PENDING = "pending"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"

class TodoPriority(Enum):
    P1 = "p1"  # Critical
    P2 = "p2"  # Important
    P3 = "p3"  # Nice-to-have

@dataclass
class Todo:
    id: str
    status: TodoStatus
    priority: TodoPriority
    title: str
    description: str
    findings: list[str]
    proposed_solutions: list[str]
    acceptance_criteria: list[str]
    tags: list[str]
    created_at: datetime
    
    def to_markdown(self) -> str:
        """Convert to markdown file format."""
        frontmatter = yaml.dump({
            "status": self.status.value,
            "priority": self.priority.value,
            "issue_id": self.id,
            "tags": self.tags,
        })
        
        content = f"""---
{frontmatter}---

# {self.title}

## Problem Statement
{self.description}

## Findings
{chr(10).join(f"- {f}" for f in self.findings)}

## Proposed Solutions
{chr(10).join(f"### Option {i+1}{chr(10)}{s}" for i, s in enumerate(self.proposed_solutions))}

## Acceptance Criteria
{chr(10).join(f"- [ ] {c}" for c in self.acceptance_criteria)}

## Work Log

### {self.created_at.strftime('%Y-%m-%d')} - Created
- Todo created from code review
"""
        return content
    
    def filename(self) -> str:
        """Generate filename following convention."""
        slug = self.title.lower().replace(" ", "-")[:30]
        return f"{self.id}-{self.status.value}-{self.priority.value}-{slug}.md"

class TodoManager:
    """Manage todos in the filesystem."""
    
    def __init__(self, todos_dir: Path):
        self.todos_dir = todos_dir
        self.todos_dir.mkdir(exist_ok=True)
        
    def create(self, todo: Todo) -> Path:
        """Create a new todo file."""
        path = self.todos_dir / todo.filename()
        path.write_text(todo.to_markdown())
        return path
    
    def list(self, status: TodoStatus | None = None) -> list[Todo]:
        """List all todos, optionally filtered by status."""
        todos = []
        for path in self.todos_dir.glob("*.md"):
            todo = self._parse_todo(path)
            if status is None or todo.status == status:
                todos.append(todo)
        return todos
    
    def update_status(self, todo_id: str, new_status: TodoStatus):
        """Update a todo's status and rename file."""
        # Find and update todo...
```

**Deliverables:**
- [ ] `Todo` dataclass
- [ ] `TodoManager` class
- [ ] Markdown serialization/parsing
- [ ] Status transitions
- [ ] File renaming on status change

### 3.2 /triage Command
**Estimated Time:** 8-12 hours

Create `src/commands/triage.py`:

```python
"""
Interactive triage of review findings.
"""
from rich.console import Console
from rich.prompt import Prompt, Confirm
from tools.file_todos import TodoManager, Todo, TodoStatus, TodoPriority

console = Console()

class TriageCommand:
    """Interactive triage of pending todos."""
    
    def __init__(self, todo_manager: TodoManager):
        self.todo_manager = todo_manager
        
    def run(self):
        """Run interactive triage session."""
        pending = self.todo_manager.list(TodoStatus.PENDING)
        
        console.print(f"[cyan]Found {len(pending)} pending items to triage[/cyan]")
        
        approved = 0
        skipped = 0
        
        for i, todo in enumerate(pending, 1):
            console.print(f"\n{'='*60}")
            console.print(f"[bold]Issue #{todo.id}: {todo.title}[/bold]")
            console.print(f"[dim]Progress: {i}/{len(pending)}[/dim]\n")
            
            self._display_todo(todo)
            
            choice = Prompt.ask(
                "Action",
                choices=["yes", "next", "custom"],
                default="yes"
            )
            
            if choice == "yes":
                self._approve_todo(todo)
                approved += 1
            elif choice == "next":
                self._skip_todo(todo)
                skipped += 1
            elif choice == "custom":
                todo = self._customize_todo(todo)
                self._approve_todo(todo)
                approved += 1
        
        self._show_summary(approved, skipped)
    
    def _display_todo(self, todo: Todo):
        """Display todo details."""
        console.print(f"[yellow]Severity: {todo.priority.value.upper()}[/yellow]")
        console.print(f"\n{todo.description}")
        console.print("\n[bold]Proposed Solution:[/bold]")
        for solution in todo.proposed_solutions:
            console.print(f"  • {solution}")
```

**Deliverables:**
- [ ] Interactive CLI for triage
- [ ] Approve/skip/customize workflow
- [ ] File operations (rename, delete)
- [ ] Summary report

### 3.3 Post-Work Validation
**Estimated Time:** 6-8 hours

After work is completed, use agents to validate the results.

Create `src/commands/validate.py`:

```python
"""
Validate completed work using agents.
"""
from tools.file_todos import TodoManager, TodoStatus

class ValidationCommand:
    """Validate completed work using agents for analysis."""
    
    def __init__(self, agent_runner, todo_manager: TodoManager):
        self.agent_runner = agent_runner
        self.todo_manager = todo_manager
        
    def run(self, work_branch: str):
        """Validate completed work (AFTER work is done)."""
        completed = self.todo_manager.list(TodoStatus.COMPLETE)
        
        console.print(f"[cyan]Validating {len(completed)} completed todos[/cyan]")
        
        # Get the changes made
        diff = self._get_work_diff(work_branch)
        
        # Build validation context
        context = f"""
## Completed Work
Branch: {work_branch}
Todos: {len(completed)}

## Changes
```diff
{diff}
```

## Task
Validate that the completed work:
1. Meets the acceptance criteria
2. Follows best practices
3. Has no regressions
4. Is properly tested

Provide your validation analysis.
"""
        
        # Use validation agents to analyze (not execute)
        validation_agents = [
            "architecture-strategist",
            "security-sentinel",
            "performance-oracle",
        ]
        
        results = self.agent_runner.run_sequence(validation_agents, context)
        
        return self._create_validation_report(results)
    
    def _create_validation_report(self, results) -> dict:
        """Create validation report from agent analysis."""
        return {
            "passed": all(r.success for r in results),
            "issues": [r.output for r in results if not r.success],
            "recommendations": [r.output for r in results if r.success]
        }
```

**Deliverables:**
- [ ] Post-work validation command
- [ ] Agent-based quality analysis
- [ ] Validation report generation
- [ ] Clear pass/fail criteria

### Stage 3 Success Criteria
- [ ] Todos created from review findings
- [ ] Interactive triage works
- [ ] Post-work validation analyzes completed work
- [ ] Status transitions tracked properly
- [ ] Validation reports help decide if work is ready to merge

---

## Stage 4: Planning & Research (Week 7-8)
**Goal:** Implement /plan command with research agents

### 4.1 Research Agent for Pre-Work Analysis
**Estimated Time:** 6-8 hours

```python
"""
Research agents for pre-work codebase analysis.
"""
RESEARCH_AGENTS = [
    "repo-research-analyst",      # Analyze repo patterns
    "best-practices-researcher",  # Research best practices
    "framework-docs-researcher",  # Framework documentation
    "git-history-analyzer",       # Git history context
]

class ResearchRunner:
    """Run research agents to gather context BEFORE work begins."""
    
    def research_feature(self, feature_description: str, repo_path: Path) -> dict:
        """Gather research for a new feature (BEFORE implementation)."""
        
        # Get repo context
        repo_context = self._get_repo_context(repo_path)
        
        # Run research agents sequentially to gather insights
        context = f"Feature: {feature_description}\n\nRepo: {repo_context}"
        results = self.agent_runner.run_sequence(RESEARCH_AGENTS, context)
        
        return {
            "patterns": results[0].output if results[0].success else "",
            "best_practices": results[1].output if results[1].success else "",
            "docs": results[2].output if results[2].success else "",
            "history": results[3].output if results[3].success else "",
        }
```

### 4.2 /plan Command
**Estimated Time:** 10-12 hours

```python
"""
Create detailed plans from feature descriptions (BEFORE work begins).
"""
class PlanCommand:
    """Generate structured plans for features using agents for research."""
    
    DETAIL_LEVELS = {
        "minimal": ["description", "acceptance_criteria"],
        "more": ["description", "technical_considerations", "acceptance_criteria", "risks"],
        "alot": ["description", "technical_considerations", "implementation_phases", 
                 "alternatives", "acceptance_criteria", "risks", "resources"]
    }
    
    def run(self, feature_description: str, detail_level: str = "more"):
        """Create a plan for a feature (PRE-WORK phase)."""
        
        # 1. Run research agents to gather insights
        research = self.research.research_feature(feature_description, self.repo_path)
        
        # 2. Generate plan using Gemini (plan creation, not execution)
        plan = self._generate_plan(feature_description, research, detail_level)
        
        # 3. Save to plans/ directory
        plan_path = self._save_plan(plan)
        
        # 4. Present plan for human/tool execution
        console.print("[green]Plan created successfully[/green]")
        console.print("Next steps:")
        console.print("  1. Review the plan")
        console.print("  2. Execute the plan using your preferred tools/process")
        console.print("  3. Run validation after completion")
        
        return plan_path
```

### 4.3 GitHub Integration
**Estimated Time:** 6-8 hours

```python
"""
GitHub API integration for issues and PRs.
"""
class GitHubClient:
    """Interact with GitHub API."""
    
    def create_issue(self, title: str, body: str, labels: list[str] = None) -> int:
        """Create a GitHub issue from plan."""
        # Use gh CLI or PyGithub
        
    def get_pr(self, pr_number: int) -> dict:
        """Get PR details."""
        
    def get_pr_diff(self, pr_number: int) -> str:
        """Get PR diff."""
        
    def add_pr_comment(self, pr_number: int, comment: str):
        """Add comment to PR."""
```

### Stage 4 Success Criteria
- [ ] Research agents gather codebase context BEFORE work
- [ ] `/plan` generates structured plans for human/tool execution
- [ ] Plans saved to `plans/` directory
- [ ] Optional GitHub issue creation
- [ ] Multiple detail levels work
- [ ] Clear separation: planning (agents) vs. execution (humans/tools)

---

## Stage 5: Work Tracking (Week 9-10)
**Goal:** Track work execution (done by humans/tools, not agents)

### 5.1 Plan Parser
**Estimated Time:** 4-6 hours

```python
"""
Parse plan files into trackable tasks.
"""
@dataclass
class PlanTask:
    id: str
    description: str
    acceptance_criteria: list[str]
    dependencies: list[str]
    
class PlanParser:
    """Parse plan markdown into tasks."""
    
    def parse(self, plan_path: Path) -> list[PlanTask]:
        """Extract tasks from plan file."""
```

### 5.2 /track Command
**Estimated Time:** 8-10 hours

```python
"""
Track work execution (NOT execute work).
"""
class TrackCommand:
    """Track progress of work execution."""
    
    def run(self, plan_path: Path):
        """Setup tracking for a plan (work executed elsewhere)."""
        
        # 1. Parse plan into tasks
        tasks = self.parser.parse(plan_path)
        
        # 2. Create worktree for isolation
        worktree = self.worktree_mgr.create(f"feature/{plan_path.stem}")
        
        # 3. Create todos for each task
        for task in tasks:
            self.todo_manager.create(task.to_todo())
        
        # 4. Display work tracking dashboard
        console.print("[cyan]Work tracking initialized[/cyan]")
        console.print(f"Worktree: {worktree.path}")
        console.print(f"Tasks: {len(tasks)}")
        console.print("\nExecute the work using your preferred tools/IDE")
        console.print("Then run: gca validate [branch] to check results")
```

### Stage 5 Success Criteria
- [ ] Plans parsed into trackable tasks
- [ ] Worktree created for isolated work
- [ ] Todos created for tracking progress
- [ ] Clear that work execution happens externally
- [ ] Dashboard shows work progress

---

## Stage 6: Polish & Distribution (Week 11-12)
**Goal:** Production-ready CLI tool

### 6.1 Error Handling & Logging
- [ ] Comprehensive error handling
- [ ] Structured logging
- [ ] Debug mode
- [ ] API error recovery

### 6.2 Configuration System
- [ ] `.gca.yaml` config file
- [ ] Environment variable support
- [ ] Project-level overrides
- [ ] Model selection

### 6.3 Documentation
- [ ] Full README with examples
- [ ] Command reference
- [ ] Agent customization guide
- [ ] Contributing guide

### 6.4 Distribution
- [ ] PyPI package: `pip install gemini-code-agent`
- [ ] Docker image
- [ ] Homebrew formula
- [ ] GitHub releases

### 6.5 Testing
- [ ] Unit tests for all modules
- [ ] Integration tests
- [ ] End-to-end workflow tests
- [ ] CI/CD pipeline

---

## Summary Timeline

| Stage | Duration | Key Deliverable |
|-------|----------|-----------------|
| **1. Foundation** | 2 weeks | CLI + Agent loading + Gemini client |
| **2. Core Workflows** | 2 weeks | `/review` for pre-merge analysis |
| **3. Todo System** | 2 weeks | `/triage` + post-work validation |
| **4. Planning** | 2 weeks | `/plan` with research agents (pre-work) |
| **5. Work Tracking** | 2 weeks | `/track` command for progress |
| **6. Polish** | 2 weeks | Production-ready distribution |

**Total: ~12 weeks (3 months) for a solo developer**

With a team of 2-3, this could be compressed to 6-8 weeks.

**Key Philosophy:** Agents are used BEFORE work (planning, research) and AFTER work (review, validation), but NOT during execution. Work is done by humans or other tools.

---

## Quick Wins (Can Do Right Now)

1. **Copy agents/** - Import all 24 agent prompts
2. **Use `gemini_agent_bridge.py`** - Already works for basic usage
3. **Start with `/review`** - Highest value, clearest scope (pre-merge analysis)
4. **Skip MCP servers initially** - Can add later with Gemini function calling

**Remember:** Use agents for analysis and recommendations only:
- `/plan` - Research and create plans (BEFORE work)
- `/review` - Analyze PRs (BEFORE merging)
- `/validate` - Check completed work (AFTER work)
- Agents do NOT execute code changes

---

## Dependencies & Requirements

### Python Packages
```
google-genai>=1.5.4
click>=8.0
rich>=13.0
pyyaml>=6.0
gitpython>=3.1
PyGithub>=2.0  # Optional, for GitHub integration
```

### External Tools
- Git (for worktrees)
- `gh` CLI (optional, for GitHub integration)

### API Keys
- `GEMINI_API_KEY` - Required
- `GITHUB_TOKEN` - Optional, for GitHub features

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Gemini API rate limits | Medium | High | Implement backoff, sequential execution |
| Agent prompts don't work well with Gemini | Low | Medium | Test each agent, adjust prompts |
| Confusion about agent role | Medium | Medium | Clear docs: agents analyze, don't execute |
| Google releases competing tool | Low | High | Focus on speed to market, open-source |

---

## Next Steps

1. **Create GitHub repo** for `gemini-code-agent`
2. **Implement Stage 1** foundation
3. **Validate** with `/review` end-to-end
4. **Iterate** based on real usage
