Experimenting with GPT on Shakespeare Dataset

## Environment

The project uses [uv](https://docs.astral.sh/uv/) for Python version and dependency management.

1. Install dependencies and create the virtual environment with `uv sync`.
2. Run project scripts via `uv run python exp1/run.py` (add flags as needed).
3. Optionally activate the environment for interactive work: `source .venv/bin/activate`.

## Leaderboard

Top 5 experiments by test loss. [View the full leaderboard](full_leaderboard.md).

<!-- leaderboard:start -->
| Experiment | Test Loss | Val Loss | Train Loss | Details | Generation | W&B |
| --- | --- | --- | --- | --- | --- | --- |
| [exp1](exp1/) | 1.5314 | 1.4494 | 1.3134 | A standard 6-layer GPT, 8 head, 128 embd-dim, block size 64, trained for one epoch | [text](exp1/generation.txt) | [link](https://wandb.ai/saahith/shakespeare-gpt/runs/ukt0p32d) |
<!-- leaderboard:end -->
