#!/usr/bin/env python3
"""Upload a full run directory (checkpoints + final) to Hugging Face Hub."""

from __future__ import annotations

import os
from pathlib import Path

import typer
from huggingface_hub import HfApi
from rich.console import Console

app = typer.Typer()
console = Console()


def _strip_quotes(value: str) -> str:
    if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
        return value[1:-1]
    return value


def load_token(token_env: str, env_file: Path) -> str:
    """Load HF token from environment or a local .env file."""
    token = os.environ.get(token_env)
    if token:
        return token

    if not env_file.exists():
        raise FileNotFoundError(f"Token env var '{token_env}' not set and env file missing: {env_file}")

    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == token_env:
            return _strip_quotes(value.strip())

    raise ValueError(f"Token '{token_env}' not found in env file: {env_file}")


@app.command()
def main(
    run_dir: Path = typer.Option(
        Path("outputs/runs/unsloth__Llama-3.1-8B-Instruct__seed1"),
        "--run-dir",
        help="Run directory to upload (checkpoints + final).",
    ),
    repo_id: str = typer.Option(
        ...,
        "--repo-id",
        help="Target Hugging Face repo (e.g., user/repo).",
    ),
    repo_type: str = typer.Option("model", "--repo-type", help="Repo type (model/dataset/space)."),
    path_in_repo: str = typer.Option(".", "--path-in-repo", help="Destination path inside the repo."),
    commit_message: str = typer.Option(
        "Add run outputs and checkpoints",
        "--commit-message",
        help="Commit message for the upload.",
    ),
    token_env: str = typer.Option(
        "HF_TOKEN",
        "--token-env",
        help="Environment variable name for the HF token.",
    ),
    env_file: Path = typer.Option(
        Path(".env"),
        "--env-file",
        help="Optional .env file to read the token from.",
    ),
    create_repo: bool = typer.Option(
        False,
        "--create-repo",
        help="Create the repo if it does not exist (requires permission).",
    ),
) -> None:
    """Upload a run directory to Hugging Face Hub."""
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    token = load_token(token_env, env_file)
    api = HfApi(token=token)

    if create_repo:
        api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)

    console.print(f"[blue]Uploading {run_dir} to {repo_id}...[/blue]")
    api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(run_dir),
        path_in_repo=path_in_repo,
        commit_message=commit_message,
    )
    console.print(f"[green]Upload complete: https://huggingface.co/{repo_id}[/green]")


if __name__ == "__main__":
    app()
