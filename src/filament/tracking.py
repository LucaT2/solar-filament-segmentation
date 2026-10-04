"""Optional Weights & Biases tracking.

``mode`` (config key ``wandb``):
    "auto"  log if wandb is installed and credentials exist (``WANDB_API_KEY`` or a prior ``wandb login``),
            otherwise silently skip
    "on"    always log (fails loudly if not logged in); "offline" logs locally for a later ``wandb sync``
    "off"   never log
``WANDB_MODE=disabled`` in the environment also turns logging off.

Runs are grouped by experiment name, so all folds of one experiment appear together in the wandb UI.
"""

from __future__ import annotations

import os
from typing import Any

PROJECT = "solar-filament-segmentation"


def _credentials_available() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        import netrc

        return netrc.netrc().authenticators("api.wandb.ai") is not None
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        return False


class Tracker:
    """Thin wrapper; every method is a no-op when tracking is off."""

    def __init__(self, run=None):
        self.run = run

    @property
    def active(self) -> bool:
        return self.run is not None

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        if self.run is not None:
            self.run.log({k: v for k, v in data.items() if isinstance(v, (int, float, bool))}, step=step)

    def summary(self, data: dict[str, Any]) -> None:
        if self.run is not None:
            self.run.summary.update(data)

    def table(self, key: str, df) -> None:
        if self.run is not None:
            import wandb

            self.run.log({key: wandb.Table(dataframe=df)})

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
            self.run = None


def start_run(mode: str, name: str, group: str, job_type: str, config: dict, tags: list[str] | None = None,
              project: str | None = None, entity: str | None = None) -> Tracker:
    if mode == "off" or os.environ.get("WANDB_MODE") == "disabled":
        return Tracker()
    try:
        import wandb
    except ImportError:
        if mode == "auto":
            return Tracker()
        raise
    if mode == "auto" and not _credentials_available():
        print("wandb: no credentials found, tracking skipped (run `wandb login` or set WANDB_API_KEY)")
        return Tracker()
    run = wandb.init(
        project=project or os.environ.get("WANDB_PROJECT", PROJECT),
        entity=entity or os.environ.get("WANDB_ENTITY"),
        name=name,
        group=group,
        job_type=job_type,
        config=config,
        tags=tags,
        mode="offline" if mode == "offline" else None,
        reinit="finish_previous",
    )
    return Tracker(run)
