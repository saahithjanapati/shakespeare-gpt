from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

RESULTS_FILENAME = "results.json"
DETAILS_FILENAME = "details.txt"
GENERATION_FILENAME = "generation.txt"
FULL_LEADERBOARD = "full_leaderboard.md"

README_MARKER_START = "<!-- leaderboard:start -->"
README_MARKER_END = "<!-- leaderboard:end -->"


@dataclass
class ExperimentResult:
    name: str
    path: Path
    train_loss: Optional[float]
    val_loss: Optional[float]
    test_loss: Optional[float]
    details: str
    wandb_url: Optional[str]
    generation_path: Optional[Path]


def read_experiments(repo_root: Path) -> List[ExperimentResult]:
    experiments: List[ExperimentResult] = []

    for child in sorted(repo_root.iterdir()):
        if not child.is_dir():
            continue

        results_path = child / RESULTS_FILENAME
        if not results_path.exists():
            continue

        try:
            data = json.loads(results_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse {results_path}: {exc}") from exc

        train_loss = _to_optional_float(data.get("train_loss"))
        val_loss = _to_optional_float(
            data.get("validation_loss", data.get("val_loss"))
        )
        test_loss = _to_optional_float(data.get("test_loss"))

        details = ""
        details_path = child / DETAILS_FILENAME
        if details_path.exists():
            raw_details = details_path.read_text(encoding="utf-8").strip()
            if raw_details:
                details = " ".join(line.strip() for line in raw_details.splitlines())

        wandb_raw = (data.get("wandb_run") or "").strip()
        wandb_url = wandb_raw or None

        generation_path = child / GENERATION_FILENAME
        generation_link = generation_path if generation_path.exists() else None

        experiments.append(
            ExperimentResult(
                name=child.name,
                path=child,
                train_loss=train_loss,
                val_loss=val_loss,
                test_loss=test_loss,
                details=details,
                wandb_url=wandb_url,
                generation_path=generation_link,
            )
        )

    experiments.sort(key=_experiment_sort_key)
    return experiments


def _experiment_sort_key(exp: ExperimentResult) -> tuple[float, str]:
    test_loss = exp.test_loss if exp.test_loss is not None else math.inf
    return (test_loss, exp.name)


def _to_optional_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_table(experiments: Iterable[ExperimentResult]) -> str:
    rows = list(experiments)
    if not rows:
        return "_No experiments have been scored yet._"

    header = (
        "| Experiment | Test Loss | Val Loss | Train Loss | Details | Generation | W&B |\n"
    )
    separator = "| --- | --- | --- | --- | --- | --- | --- |\n"

    body_lines = []
    for exp in rows:
        exp_link = f"[{exp.name}]({exp.path.name}/)"
        test_loss = _format_loss(exp.test_loss)
        val_loss = _format_loss(exp.val_loss)
        train_loss = _format_loss(exp.train_loss)
        details = _sanitize_cell(exp.details) if exp.details else "—"
        generation = _format_generation(exp)
        wandb = _format_wandb(exp.wandb_url)

        body_lines.append(
            f"| {exp_link} | {test_loss} | {val_loss} | {train_loss} | {details} | {generation} | {wandb} |"
        )

    return header + separator + "\n".join(body_lines)


def _format_loss(loss: Optional[float]) -> str:
    if loss is None or math.isinf(loss) or math.isnan(loss):
        return "—"
    return f"{loss:.4f}"


def _sanitize_cell(text: str) -> str:
    if not text:
        return "—"
    sanitized = text.replace("|", "\\|")
    return sanitized.replace("\n", " ").strip()


def _format_generation(exp: ExperimentResult) -> str:
    if not exp.generation_path:
        return "—"
    rel_path = exp.generation_path.relative_to(exp.path.parent)
    return f"[text]({rel_path.as_posix()})"


def _format_wandb(url: Optional[str]) -> str:
    if not url:
        return "—"
    if url.startswith(("http://", "https://")):
        return f"[link]({url})"
    return url


def write_full_leaderboard(path: Path, table: str) -> None:
    content = (
        "# Full Leaderboard\n\n"
        "All recorded experiments ranked by test loss.\n\n"
        f"{table}\n"
    )
    path.write_text(content, encoding="utf-8")


def update_readme(readme_path: Path, table: str) -> None:
    readme_text = readme_path.read_text(encoding="utf-8")

    leaderboard_section = (
        "## Leaderboard\n\n"
        "Top 5 experiments by test loss. "
        "[View the full leaderboard](full_leaderboard.md).\n\n"
        f"{README_MARKER_START}\n"
        f"{table}\n"
        f"{README_MARKER_END}\n"
    )

    if README_MARKER_START in readme_text and README_MARKER_END in readme_text:
        start_idx = readme_text.index(README_MARKER_START)
        end_idx = readme_text.index(README_MARKER_END) + len(README_MARKER_END)
        replacement = f"{README_MARKER_START}\n{table}\n{README_MARKER_END}"
        readme_text = readme_text[:start_idx] + replacement + readme_text[end_idx:]

        if "## Leaderboard" not in readme_text:
            readme_text += "\n" + leaderboard_section
    else:
        if not readme_text.endswith("\n"):
            readme_text += "\n"
        readme_text += "\n" + leaderboard_section

    readme_path.write_text(readme_text, encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    experiments = read_experiments(repo_root)

    full_table = build_table(experiments)
    top_table = build_table(experiments[:5])

    write_full_leaderboard(repo_root / FULL_LEADERBOARD, full_table)
    update_readme(repo_root / "README.md", top_table)
    print("Leaderboard updated.")


if __name__ == "__main__":
    main()
