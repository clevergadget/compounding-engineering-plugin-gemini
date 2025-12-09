# Gemini Compounding Engineering: Implementation Plan

A detailed, actionable roadmap for reimplementing the compounding-engineering plugin for Google's Gemini.

## Overview

This plan creates **"Gemini Code Agent" (GCA)** - an open-source CLI tool that brings compounding engineering workflows to Gemini users.

### Target Architecture

```
gemini-code-agent/
├── gca                           # CLI entry point
├── src/
│   ├── core/
│   │   ├── agent_loader.py       # Load agent prompts from markdown
│   │   ├── gemini_client.py      # Gemini API wrapper
│   │   ├── orchestrator.py       # Parallel agent execution
│   │   └── config.py             # Configuration management
│   ├── commands/
│   │   ├── plan.py               # /plan implementation
│   │   ├── work.py               # /work implementation
│   │   ├── review.py             # /review implementation
│   │   ├── triage.py             # /triage implementation
│   │   └── resolve.py            # /resolve_* implementations
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

---

## Stage 2: Core Workflows (Week 3-4)
**Goal:** Implement /review command with parallel agents

### 2.1 Parallel Agent Orchestrator
**Estimated Time:** 8-12 hours

Create `src/core/orchestrator.py`:

```python
"""
Run multiple agents in parallel and aggregate results.
"""
import asyncio
from dataclasses import dataclass
from typing import Callable

@dataclass
class AgentTask:
    agent_name: str
    context: str
    
@dataclass
class AgentResult:
    agent_name: str
    output: str
    success: bool
    error: str | None = None

class ParallelOrchestrator:
    """Execute multiple agents concurrently."""
    
    def __init__(self, registry, client, max_concurrent: int = 5):
        self.registry = registry
        self.client = client
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
    async def run_agent(self, task: AgentTask) -> AgentResult:
        """Run a single agent task."""
        async with self.semaphore:
            agent = self.registry.get(task.agent_name)
            if not agent:
                return AgentResult(task.agent_name, "", False, f"Agent not found")
            
            try:
                # Run in thread pool since genai is sync
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self.client.generate(task.context, agent.system_prompt)
                )
                return AgentResult(task.agent_name, result, True)
            except Exception as e:
                return AgentResult(task.agent_name, "", False, str(e))
    
    async def run_all(self, tasks: list[AgentTask]) -> list[AgentResult]:
        """Run all tasks concurrently."""
        return await asyncio.gather(*[self.run_agent(t) for t in tasks])
    
    def run_parallel(self, tasks: list[AgentTask]) -> list[AgentResult]:
        """Synchronous wrapper for parallel execution."""
        return asyncio.run(self.run_all(tasks))
```

**Deliverables:**
- [ ] `ParallelOrchestrator` class
- [ ] Rate limiting with semaphore
- [ ] Error handling per agent
- [ ] Progress reporting
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
**Estimated Time:** 12-16 hours

Create `src/commands/review.py`:

```python
"""
Multi-agent code review command.
"""
from dataclasses import dataclass
from pathlib import Path
import subprocess

from core.orchestrator import ParallelOrchestrator, AgentTask
from tools.git_worktree import GitWorktreeManager

# Review agents to run in parallel
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
    
    def __init__(self, orchestrator: ParallelOrchestrator, worktree_mgr: GitWorktreeManager):
        self.orchestrator = orchestrator
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
        """Execute the review."""
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
"""
        
        # Create tasks for each review agent
        tasks = [
            AgentTask(agent_name=agent, context=context)
            for agent in REVIEW_AGENTS
        ]
        
        # Run all agents in parallel
        results = self.orchestrator.run_parallel(tasks)
        
        # Aggregate and return findings
        return self._aggregate_findings(results)
    
    def _aggregate_findings(self, results) -> list[dict]:
        """Combine findings from all agents."""
        findings = []
        for result in results:
            if result.success:
                findings.append({
                    "agent": result.agent_name,
                    "analysis": result.output
                })
        return findings
```

**Deliverables:**
- [ ] `ReviewCommand` class
- [ ] PR/branch target parsing
- [ ] Parallel agent execution
- [ ] Finding aggregation
- [ ] CLI integration: `gca review [PR#]`

### Stage 2 Success Criteria
- [ ] `/review` command works with PR numbers
- [ ] At least 5 agents run in parallel
- [ ] Results aggregated and displayed nicely
- [ ] Worktree creation works

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

### 3.3 /resolve_todo_parallel Command
**Estimated Time:** 8-12 hours

Create `src/commands/resolve.py`:

```python
"""
Resolve multiple todos in parallel.
"""
from tools.file_todos import TodoManager, TodoStatus

class ResolveTodoCommand:
    """Resolve ready todos using agents."""
    
    def __init__(self, orchestrator, todo_manager: TodoManager, client):
        self.orchestrator = orchestrator
        self.todo_manager = todo_manager
        self.client = client
        
    def run(self, max_parallel: int = 3):
        """Resolve all ready todos."""
        ready = self.todo_manager.list(TodoStatus.READY)
        
        console.print(f"[cyan]Found {len(ready)} ready todos to resolve[/cyan]")
        
        # Analyze dependencies
        dependency_graph = self._build_dependency_graph(ready)
        execution_order = self._topological_sort(dependency_graph)
        
        # Display execution plan
        self._display_mermaid_diagram(execution_order)
        
        if not Confirm.ask("Proceed with this execution plan?"):
            return
            
        # Execute in parallel batches
        for batch in execution_order:
            self._execute_batch(batch)
    
    def _execute_batch(self, todos: list[Todo]):
        """Execute a batch of todos in parallel."""
        tasks = []
        for todo in todos:
            task = AgentTask(
                agent_name=self._select_agent(todo),
                context=self._build_resolution_context(todo)
            )
            tasks.append(task)
        
        results = self.orchestrator.run_parallel(tasks)
        
        for todo, result in zip(todos, results):
            if result.success:
                self._apply_resolution(todo, result.output)
```

**Deliverables:**
- [ ] Dependency analysis
- [ ] Mermaid diagram generation
- [ ] Parallel batch execution
- [ ] Resolution application

### Stage 3 Success Criteria
- [ ] Todos created from review findings
- [ ] Interactive triage works
- [ ] `/resolve_todo_parallel` executes correctly
- [ ] Status transitions tracked properly

---

## Stage 4: Planning & Research (Week 7-8)
**Goal:** Implement /plan command with research agents

### 4.1 Research Agent Orchestration
**Estimated Time:** 8-12 hours

```python
"""
Research agents for codebase analysis.
"""
RESEARCH_AGENTS = [
    "repo-research-analyst",      # Analyze repo patterns
    "best-practices-researcher",  # Research best practices
    "framework-docs-researcher",  # Framework documentation
    "git-history-analyzer",       # Git history context
]

class ResearchOrchestrator:
    """Run research agents to gather context."""
    
    def research_feature(self, feature_description: str, repo_path: Path) -> dict:
        """Gather research for a new feature."""
        
        # Get repo context
        repo_context = self._get_repo_context(repo_path)
        
        # Run research agents in parallel
        tasks = [
            AgentTask(
                agent_name=agent,
                context=f"Feature: {feature_description}\n\nRepo: {repo_context}"
            )
            for agent in RESEARCH_AGENTS
        ]
        
        results = self.orchestrator.run_parallel(tasks)
        
        return {
            "patterns": results[0].output if results[0].success else "",
            "best_practices": results[1].output if results[1].success else "",
            "docs": results[2].output if results[2].success else "",
            "history": results[3].output if results[3].success else "",
        }
```

### 4.2 /plan Command
**Estimated Time:** 12-16 hours

```python
"""
Create detailed plans from feature descriptions.
"""
class PlanCommand:
    """Generate structured plans for features."""
    
    DETAIL_LEVELS = {
        "minimal": ["description", "acceptance_criteria"],
        "more": ["description", "technical_considerations", "acceptance_criteria", "risks"],
        "alot": ["description", "technical_considerations", "implementation_phases", 
                 "alternatives", "acceptance_criteria", "risks", "resources"]
    }
    
    def run(self, feature_description: str, detail_level: str = "more"):
        """Create a plan for a feature."""
        
        # 1. Run research agents
        research = self.research.research_feature(feature_description, self.repo_path)
        
        # 2. Generate plan using Gemini
        plan = self._generate_plan(feature_description, research, detail_level)
        
        # 3. Save to plans/ directory
        plan_path = self._save_plan(plan)
        
        # 4. Offer next steps
        return self._present_options(plan_path)
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
- [ ] Research agents gather codebase context
- [ ] `/plan` generates structured plans
- [ ] Plans saved to `plans/` directory
- [ ] Optional GitHub issue creation
- [ ] Multiple detail levels work

---

## Stage 5: Work Execution (Week 9-10)
**Goal:** Implement /work command for plan execution

### 5.1 Plan Parser
**Estimated Time:** 4-6 hours

```python
"""
Parse plan files into actionable tasks.
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

### 5.2 /work Command
**Estimated Time:** 12-16 hours

```python
"""
Execute work plans systematically.
"""
class WorkCommand:
    """Execute a plan with tracking and validation."""
    
    def run(self, plan_path: Path):
        """Execute a plan."""
        
        # 1. Parse plan into tasks
        tasks = self.parser.parse(plan_path)
        
        # 2. Create worktree for isolation
        worktree = self.worktree_mgr.create(f"feature/{plan_path.stem}")
        
        # 3. Create todos for each task
        for task in tasks:
            self.todo_manager.create(task.to_todo())
        
        # 4. Execute tasks
        for task in tasks:
            self._execute_task(task, worktree)
            self._run_tests()
            self._update_progress(task)
        
        # 5. Create PR
        self._create_pr(plan_path, worktree)
```

### Stage 5 Success Criteria
- [ ] Plans parsed into tasks
- [ ] Worktree created for isolation
- [ ] Tasks executed systematically
- [ ] Tests run after each task
- [ ] Progress tracked via todos
- [ ] PR created on completion

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
| **2. Core Workflows** | 2 weeks | `/review` with parallel agents |
| **3. Todo System** | 2 weeks | `/triage` + `/resolve_todo_parallel` |
| **4. Planning** | 2 weeks | `/plan` with research agents |
| **5. Work Execution** | 2 weeks | `/work` command |
| **6. Polish** | 2 weeks | Production-ready distribution |

**Total: ~12 weeks (3 months) for a solo developer**

With a team of 2-3, this could be compressed to 6-8 weeks.

---

## Quick Wins (Can Do Right Now)

1. **Copy agents/** - Import all 24 agent prompts
2. **Use `gemini_agent_bridge.py`** - Already works for basic usage
3. **Start with `/review`** - Highest value, clearest scope
4. **Skip MCP servers initially** - Can add later with Gemini function calling

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
| Gemini API rate limits | Medium | High | Implement backoff, batching |
| Agent prompts don't work well with Gemini | Low | Medium | Test each agent, adjust prompts |
| Parallel execution complexity | Medium | Medium | Start with serial, add parallel later |
| Google releases competing tool | Low | High | Focus on speed to market, open-source |

---

## Next Steps

1. **Create GitHub repo** for `gemini-code-agent`
2. **Implement Stage 1** foundation
3. **Validate** with `/review` end-to-end
4. **Iterate** based on real usage
