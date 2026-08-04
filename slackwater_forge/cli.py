"""
CLI entry point — Click-based command interface for Slackwater Forge.

Commands:
  forge run      — Start the overnight forge loop
  forge brief    — Generate a morning briefing from artifacts
  forge job      — Manage job specs (create, list, edit, show)
  forge status   — Show current forge state
  forge models   — List available Ollama models
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.live import Live
from rich.layout import Layout

from . import __version__
from .forge import Forge, ForgeState
from .briefer import Briefer
from .jobs import (
    JobManager,
    JobSpec,
    JobType,
    Priority,
    ForgeSession,
    BUILTIN_TEMPLATES,
)
from .models import OllamaClient

console = Console()


def _parse_duration(duration_str: str) -> int | None:
    """Parse duration strings like '8h', '30m', '3600s' into seconds."""
    if not duration_str:
        return None
    duration_str = duration_str.strip().lower()
    units = {"h": 3600, "m": 60, "s": 1}
    if duration_str[-1] in units:
        try:
            return int(float(duration_str[:-1]) * units[duration_str[-1]])
        except ValueError:
            console.print(f"[red]Invalid duration: {duration_str}[/red]")
            raise click.Abort()
    try:
        return int(duration_str)
    except ValueError:
        console.print(f"[red]Invalid duration: {duration_str}[/red]")
        raise click.Abort()


@click.group()
@click.version_option(__version__, prog_name="slackwater-forge")
@click.option("--host", default="localhost", envvar="OLLAMA_HOST", help="Ollama host")
@click.option("--port", default=11434, envvar="OLLAMA_PORT", type=int, help="Ollama port")
@click.option("--output", "-o", default="forge-output", envvar="FORGE_OUTPUT", help="Output directory")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
@click.pass_context
def main(ctx: click.Context, host: str, port: int, output: str, verbose: bool) -> None:
    """🔥 Slackwater Forge — Overnight GPU production line & morning briefing generator."""
    ctx.ensure_object(dict)
    ctx.obj["host"] = host
    ctx.obj["port"] = port
    ctx.obj["output"] = output
    if verbose:
        import logging
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        )
    else:
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(message)s",
        )


@main.command()
@click.option("--session", "-s", default=None, help="Session name (from job specs)")
@click.option("--model", "-m", default=None, help="Override model for all jobs")
@click.option("--iterations", "-i", type=int, default=None, help="Max total iterations")
@click.option("--duration", "-d", default=None, help="Time limit (e.g., '8h', '30m', '3600s')")
@click.option("--continuous", "-c", is_flag=True, help="Run until stopped (Ctrl-C)")
@click.option("--dry-run", is_flag=True, help="Don't call Ollama, just log what would happen")
@click.option("--jobs-dir", default="jobs", help="Directory for job specs")
@click.pass_context
def run(
    ctx: click.Context,
    session: str | None,
    model: str | None,
    iterations: int | None,
    duration: str | None,
    continuous: bool,
    dry_run: bool,
    jobs_dir: str,
) -> None:
    """Start the forge loop — generates artifacts via Ollama."""
    output = ctx.obj["output"]
    host = ctx.obj["host"]
    port = ctx.obj["port"]

    # Load or create a session
    if session:
        mgr = JobManager(jobs_dir)
        try:
            sess = mgr.load_session(session)
            console.print(f"[green]Loaded session:[/green] {session} ({len(sess.jobs)} jobs)")
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            sys.exit(1)
    else:
        # Create a default session from built-in templates
        console.print("[yellow]No session specified, using built-in templates.[/yellow]")
        sess = _create_default_session(model or "granite3.1-dense:2b")

    # Override model if specified
    if model:
        for job in sess.jobs:
            job.model = model
        sess.models = [model]

    if not sess.jobs:
        console.print("[red]No jobs in session. Use 'forge job create' to add jobs first.[/red]")
        sys.exit(1)

    # Show session summary
    _print_session_summary(sess)

    # Check Ollama
    if not dry_run:
        client = OllamaClient(host=host, port=port)
        if not client.is_available():
            console.print(f"[red]Ollama not available at http://{host}:{port}[/red]")
            console.print("[dim]Start it with: ollama serve[/dim]")
            sys.exit(1)

        available = client.list_models()
        model_names = [m.name for m in available]
        needed = set(j.model for j in sess.jobs)
        missing = needed - set(model_names)
        if missing:
            console.print(f"[yellow]Missing models: {', '.join(missing)}[/yellow]")
            console.print("[dim]Pull with: ollama pull <model-name>[/dim]")
            if not click.confirm("Continue anyway?", default=False):
                sys.exit(0)
        client.close()

    # Parse duration
    max_duration = _parse_duration(duration) if duration else None

    # Create forge and run
    forge = Forge(output_dir=output, ollama_host=host, ollama_port=port)

    console.print(f"\n[bold green]🔥 Forge started![/bold green]")
    if max_duration:
        console.print(f"   Duration: {max_duration}s ({duration})")
    if iterations:
        console.print(f"   Max iterations: {iterations}")
    if continuous:
        console.print("   Mode: [red]continuous[/red] (Ctrl-C to stop)")
    console.print(f"   Output: {output}/")
    console.print()

    stats = forge.run_session(
        session=sess,
        max_iterations=iterations,
        max_duration_seconds=max_duration,
        continuous=continuous,
        dry_run=dry_run,
    )

    # Final summary
    console.print()
    _print_stats(stats)


@main.command()
@click.option("--model", "-m", default=None, help="Ollama model for AI summary")
@click.option("--recipient", "-r", default="Operator", help="Briefing recipient name")
@click.option("--format", "-f", "formats", multiple=True, default=["md"],
              type=click.Choice(["md", "html"]), help="Output format(s)")
@click.option("--no-ai", is_flag=True, help="Skip AI synthesis (offline mode)")
@click.option("--open", "open_browser", is_flag=True, help="Open HTML in browser")
@click.pass_context
def brief(
    ctx: click.Context,
    model: str | None,
    recipient: str,
    formats: tuple[str, ...],
    no_ai: bool,
    open_browser: bool,
) -> None:
    """Generate a morning briefing from accumulated artifacts."""
    output = ctx.obj["output"]
    host = ctx.obj["host"]
    port = ctx.obj["port"]

    briefer = Briefer(output_dir=output, ollama_host=host, ollama_port=port)

    console.print("[bold]📋 Generating morning briefing...[/bold]")

    use_ai = not no_ai
    briefing_model = model

    # Auto-detect model if not specified and AI is enabled
    if use_ai and not briefing_model:
        client = OllamaClient(host=host, port=port)
        if client.is_available():
            models = client.list_models()
            if models:
                # Prefer smaller/faster models for summarization
                briefing_model = models[0].name
                console.print(f"[dim]Using model: {briefing_model}[/dim]")
        client.close()

    briefing = briefer.generate(
        model=briefing_model,
        recipient=recipient,
        use_ai_summary=use_ai,
    )

    if briefing.get("error"):
        console.print(f"[yellow]{briefing['error']}[/yellow]")
        console.print(f"[dim]Run 'forge run' first to produce artifacts.[/dim]")
        sys.exit(0)

    # Save
    paths = briefer.save_briefing(briefing, formats=list(formats))

    # Display
    console.print()
    md_text = briefer.to_markdown(briefing)
    console.print(Panel(Markdown(md_text), border_style="blue"))

    console.print("\n[bold green]✅ Briefing saved:[/bold green]")
    for p in paths:
        console.print(f"   📄 {p}")

    # Open in browser
    if open_browser and "html" in formats:
        import webbrowser
        html_path = [p for p in paths if p.suffix == ".html"]
        if html_path:
            webbrowser.open(f"file://{html_path[0].resolve()}")


@main.group()
@click.pass_context
def job(ctx: click.Context) -> None:
    """Manage job specs — create, list, edit, show, delete."""
    pass


@job.command("list")
@click.option("--jobs-dir", default="jobs", help="Directory for job specs")
@click.pass_context
def job_list(ctx: click.Context, jobs_dir: str) -> None:
    """List all saved job sessions."""
    mgr = JobManager(jobs_dir)
    sessions = mgr.list_sessions()

    if not sessions:
        console.print("[yellow]No sessions found.[/yellow]")
        console.print("[dim]Create one with: forge job create[/dim]")
        return

    table = Table(title="📋 Forge Sessions")
    table.add_column("Name", style="cyan")
    table.add_column("Jobs", justify="right")
    table.add_column("Models", style="green")
    table.add_column("Description", style="dim")

    for path in sessions:
        try:
            sess = ForgeSession.from_file(path)
            table.add_row(
                sess.name,
                str(len(sess.jobs)),
                ", ".join(sess.models[:3]),
                sess.description[:50],
            )
        except Exception:
            table.add_row(path.stem, "?", "?", "[red]Error loading[/red]")

    console.print(table)


@job.command("create")
@click.option("--name", "-n", required=True, help="Session name")
@click.option("--description", "-d", default="", help="Session description")
@click.option("--model", "-m", default="granite3.1-dense:2b", help="Default model")
@click.option("--jobs-dir", default="jobs", help="Directory for job specs")
@click.option("--template", "-t", is_flag=True, help="Start with built-in templates")
@click.pass_context
def job_create(
    ctx: click.Context,
    name: str,
    description: str,
    model: str,
    jobs_dir: str,
    template: bool,
) -> None:
    """Create a new forge session."""
    mgr = JobManager(jobs_dir)

    session = mgr.create_session(
        name=name,
        description=description,
        models=[model],
    )

    if template:
        for tmpl_key, tmpl_data in BUILTIN_TEMPLATES.items():
            job_spec = JobSpec(**tmpl_data)
            job_spec.model = model
            session.add_job(job_spec)

    path = mgr.save_session(session)
    console.print(f"[green]✅ Created session:[/green] {path}")
    console.print(f"   {len(session.jobs)} jobs")
    console.print(f"\n[dim]Add jobs with: forge job add {name}[/dim]")


@job.command("add")
@click.argument("session_name")
@click.option("--id", "job_id", required=True, help="Job ID")
@click.option("--name", "job_name", required=True, help="Job name")
@click.option("--prompt", "-p", required=True, help="Prompt template")
@click.option("--type", "job_type", default="custom",
              type=click.Choice([t.value for t in JobType]))
@click.option("--model", "-m", default=None, help="Model override")
@click.option("--priority", default="medium",
              type=click.Choice([p.value for p in Priority]))
@click.option("--system", "system_prompt", default="", help="System prompt")
@click.option("--max-iterations", type=int, default=1, help="Max iterations for this job")
@click.option("--jobs-dir", default="jobs", help="Directory for job specs")
@click.pass_context
def job_add(
    ctx: click.Context,
    session_name: str,
    job_id: str,
    job_name: str,
    prompt: str,
    job_type: str,
    model: str | None,
    priority: str,
    system_prompt: str,
    max_iterations: int,
    jobs_dir: str,
) -> None:
    """Add a job to an existing session."""
    mgr = JobManager(jobs_dir)
    session = mgr.load_session(session_name)

    spec = JobSpec(
        id=job_id,
        name=job_name,
        prompt=prompt,
        type=JobType(job_type),
        model=model or session.models[0] if session.models else "granite3.1-dense:2b",
        priority=Priority(priority),
        system_prompt=system_prompt,
        max_iterations=max_iterations,
    )
    session.add_job(spec)
    mgr.save_session(session)

    console.print(f"[green]✅ Added job '{job_name}' to session '{session_name}'[/green]")


@job.command("show")
@click.argument("session_name")
@click.option("--jobs-dir", default="jobs", help="Directory for job specs")
@click.pass_context
def job_show(ctx: click.Context, session_name: str, jobs_dir: str) -> None:
    """Show details of a specific session."""
    mgr = JobManager(jobs_dir)
    session = mgr.load_session(session_name)

    _print_session_summary(session)

    if session.jobs:
        table = Table(title=f"Jobs in '{session_name}'")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Type", style="blue")
        table.add_column("Model", style="green")
        table.add_column("Priority", style="yellow")
        table.add_column("Max Iter", justify="right")

        for job in session.jobs:
            table.add_row(
                job.id,
                job.name,
                job.type.value,
                job.model,
                job.priority.value,
                str(job.max_iterations),
            )

        console.print(table)


@job.command("remove")
@click.argument("session_name")
@click.argument("job_id")
@click.option("--jobs-dir", default="jobs", help="Directory for job specs")
@click.pass_context
def job_remove(ctx: click.Context, session_name: str, job_id: str, jobs_dir: str) -> None:
    """Remove a job from a session."""
    mgr = JobManager(jobs_dir)
    session = mgr.load_session(session_name)

    if session.remove_job(job_id):
        mgr.save_session(session)
        console.print(f"[green]Removed job '{job_id}' from '{session_name}'[/green]")
    else:
        console.print(f"[red]Job '{job_id}' not found in '{session_name}'[/red]")


@job.command("delete")
@click.argument("session_name")
@click.option("--jobs-dir", default="jobs", help="Directory for job specs")
@click.option("--force", is_flag=True, help="Skip confirmation")
@click.pass_context
def job_delete(ctx: click.Context, session_name: str, jobs_dir: str, force: bool) -> None:
    """Delete an entire session."""
    if not force:
        if not click.confirm(f"Delete session '{session_name}'?"):
            return
    mgr = JobManager(jobs_dir)
    if mgr.delete_session(session_name):
        console.print(f"[green]Deleted session '{session_name}'[/green]")
    else:
        console.print(f"[red]Session '{session_name}' not found[/red]")


@main.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show current forge state — running jobs, artifact count, GPU status."""
    output = ctx.obj["output"]
    host = ctx.obj["host"]
    port = ctx.obj["port"]

    forge = Forge(output_dir=output, ollama_host=host, ollama_port=port)
    status_data = forge.get_status()

    # Status panel
    state = status_data["state"]
    state_color = {
        "running": "green",
        "idle": "dim",
        "stopped": "yellow",
        "error": "red",
        "paused": "yellow",
    }.get(state, "white")

    console.print(Panel(
        f"[{state_color}]{state.upper()}[/{state_color}]",
        title="🔥 Forge Status",
        border_style=state_color,
    ))

    # GPU status
    ollama_ok = status_data["ollama_available"]
    gpu_icon = "✅" if ollama_ok else "❌"
    console.print(f"\n{gpu_icon} Ollama: {status_data['ollama_url']}")

    if ollama_ok:
        client = OllamaClient(host=host, port=port)
        models = client.list_models()
        if models:
            table = Table(title="Available Models")
            table.add_column("Model", style="cyan")
            table.add_column("Size", style="green")
            table.add_column("Family", style="blue")
            table.add_column("Params", style="yellow")

            for m in models[:15]:  # Limit display
                table.add_row(
                    m.name,
                    f"{m.size / 1e9:.1f}GB" if m.size and isinstance(m.size, (int, float)) and m.size > 0 else str(m.size),
                    m.family,
                    m.parameter_size,
                )
            console.print(table)
        client.close()

    # Artifact stats
    stats = status_data.get("stats", {})
    console.print(f"\n📊 Artifacts: [bold]{status_data['artifact_count']}[/bold]")
    if stats:
        console.print(f"   Iterations: {stats.get('iterations', 0)}")
        console.print(f"   Total tokens: {stats.get('total_tokens', 0):,}")
        console.print(f"   Errors: {stats.get('errors', 0)}")
        if stats.get("models_used"):
            console.print(f"   Models used: {', '.join(stats['models_used'])}")

    # Output dir
    console.print(f"\n📁 Output: {status_data['output_dir']}")

    if status_data["last_updated"]:
        console.print(f"🕐 Last update: {status_data['last_updated']}")


@main.command()
@click.pass_context
def models(ctx: click.Context) -> None:
    """List available Ollama models."""
    host = ctx.obj["host"]
    port = ctx.obj["port"]

    client = OllamaClient(host=host, port=port)

    if not client.is_available():
        console.print(f"[red]Ollama not available at http://{host}:{port}[/red]")
        sys.exit(1)

    available = client.list_models()

    if not available:
        console.print("[yellow]No models installed.[/yellow]")
        console.print("[dim]Pull with: ollama pull <model-name>[/dim]")
        return

    table = Table(title=f"🧠 Ollama Models ({len(available)} available)")
    table.add_column("#", style="dim", justify="right")
    table.add_column("Name", style="cyan")
    table.add_column("Size", style="green")
    table.add_column("Quantization", style="blue")
    table.add_column("Family", style="magenta")
    table.add_column("Params", style="yellow")

    for i, m in enumerate(available, 1):
        size_str = f"{m.size / 1e9:.1f}GB" if m.size and isinstance(m.size, (int, float)) and m.size > 0 else str(m.size)
        table.add_row(
            str(i),
            m.name,
            size_str,
            m.quantization,
            m.family,
            m.parameter_size,
        )

    console.print(table)
    client.close()


@main.command()
@click.option("--model", "-m", default=None, help="Model to test")
@click.option("--prompt", "-p", default="Hello! What are you?", help="Test prompt")
@click.pass_context
def test(ctx: click.Context, model: str | None, prompt: str) -> None:
    """Test Ollama connection with a quick prompt."""
    host = ctx.obj["host"]
    port = ctx.obj["port"]

    client = OllamaClient(host=host, port=port)

    if not client.is_available():
        console.print(f"[red]Ollama not available at http://{host}:{port}[/red]")
        sys.exit(1)

    available = client.list_models()
    if not available:
        console.print("[red]No models installed.[/red]")
        sys.exit(1)

    if not model:
        model = available[0].name
        console.print(f"[dim]Using first available model: {model}[/dim]")

    console.print(f"[bold]Testing {model}...[/bold]")
    console.print(f"[dim]Prompt: {prompt}[/dim]\n")

    start = time.monotonic()
    result = client.generate(model=model, prompt=prompt)
    elapsed = time.monotonic() - start

    console.print(Panel(result.text, title=f"Response from {model}", border_style="green"))
    console.print(
        f"\n[dim]{result.eval_count} tokens in {elapsed:.1f}s "
        f"({result.tokens_per_second:.1f} tok/s)[/dim]"
    )
    client.close()


# --- Helpers ---

def _create_default_session(model: str) -> ForgeSession:
    """Create a default session with built-in templates."""
    session = ForgeSession(
        name="default",
        description="Built-in default session",
        models=[model],
    )
    for tmpl_key, tmpl_data in BUILTIN_TEMPLATES.items():
        spec = JobSpec(**tmpl_data)
        spec.model = model
        session.add_job(spec)
    return session


def _print_session_summary(session: ForgeSession) -> None:
    """Print a rich summary of a forge session."""
    console.print(Panel(
        f"[bold cyan]{session.name}[/bold cyan]\n"
        f"{session.description or 'No description'}\n"
        f"Models: {', '.join(session.models)}",
        title="📋 Forge Session",
        border_style="blue",
    ))

    if session.jobs:
        table = Table(show_header=True, header_style="bold")
        table.add_column("#", style="dim", justify="right")
        table.add_column("Job", style="cyan")
        table.add_column("Type", style="blue")
        table.add_column("Model", style="green")
        table.add_column("Priority", style="yellow")
        table.add_column("Enabled", justify="center")

        for i, job in enumerate(session.jobs, 1):
            enabled = "✅" if job.enabled else "❌"
            table.add_row(
                str(i),
                job.name,
                job.type.value,
                job.model,
                job.priority.value,
                enabled,
            )
        console.print(table)


def _print_stats(stats: Any) -> None:
    """Print forge run statistics."""
    console.print(Panel(
        f"   Iterations: [bold]{stats.iterations}[/bold]\n"
        f"   Artifacts:  [bold green]{stats.artifacts_produced}[/bold green]\n"
        f"   Tokens:     [bold]{stats.total_tokens:,}[/bold]\n"
        f"   Errors:     [{'red' if stats.errors else 'green'}]{stats.errors}[/]\n"
        f"   Models:     {', '.join(stats.models_used) if stats.models_used else 'none'}\n"
        f"   Elapsed:    {stats.to_dict().get('elapsed_seconds', 0):.1f}s",
        title="📊 Forge Complete",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
