"""EBRM System CLI."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from ebrm_system import __version__
from ebrm_system.intent import RuleBasedClassifier
from ebrm_system.verifiers import SymPyVerifier, VerifierChain, chain_for_intent

app = typer.Typer(
    name="ebrm-system",
    help="EBRM System — production reasoning pipeline (CLI).",
    no_args_is_help=True,
)
console = Console()


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"ebrm-system {__version__}")


@app.command()
def classify(query: str) -> None:
    """Classify a query's intent and suggested compute budget."""
    clf = RuleBasedClassifier()
    pred = clf.classify(query)

    table = Table(title="Intent Classification")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("intent", pred.intent.value)
    table.add_row("difficulty", f"{pred.difficulty:.3f}")
    table.add_row("langevin_steps", str(pred.suggested_langevin_steps))
    table.add_row("restarts", str(pred.suggested_restarts))
    table.add_row("trace_count", str(pred.suggested_trace_count))
    table.add_row("reasoning", pred.reasoning)
    console.print(table)


@app.command()
def verify(
    candidate: str = typer.Argument(..., help="Candidate answer expression"),
    expected: str = typer.Argument(..., help="Expected value (SymPy expression or number)"),
) -> None:
    """Run the SymPy verifier on a candidate/expected pair."""
    chain = VerifierChain([SymPyVerifier()])
    results = chain.verify(candidate, {"expected": expected})
    for r in results:
        status = "[green]PASS[/green]" if r.verified else "[red]FAIL[/red]"
        console.print(f"{status} [{r.verifier}] {r.reason}")


@app.command("verify-routed")
def verify_routed(
    query: str = typer.Argument(..., help="Original user query (used for intent routing)"),
    candidate: str = typer.Argument(..., help="Candidate answer to verify"),
    expected: str = typer.Option("", "--expected", help="Expected value for math chains"),
) -> None:
    """Classify the query, then run the intent-routed verifier chain."""
    clf = RuleBasedClassifier()
    pred = clf.classify(query)
    chain = chain_for_intent(pred.intent)
    if not chain.verifiers:
        console.print(
            f"[yellow]intent={pred.intent.value}: no hard verifiers; EBRM soft score only.[/yellow]"
        )
        return
    context: dict[str, object] = {"expected": expected} if expected else {}
    results = chain.verify(candidate, context)
    console.print(f"[cyan]intent={pred.intent.value}[/cyan]")
    for r in results:
        status = "[green]PASS[/green]" if r.verified else "[red]FAIL[/red]"
        console.print(f"{status} [{r.verifier}] {r.reason}")


if __name__ == "__main__":
    app()
