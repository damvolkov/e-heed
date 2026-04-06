"""Cyclopts CLI entry point for eheed."""

from __future__ import annotations

import contextlib

import cyclopts

from e_heed.cli.commands.config import app as config_app
from e_heed.cli.commands.run import app as run_app
from e_heed.cli.commands.sentinel import app as sentinel_app
from e_heed.cli.commands.session import app as session_app

app = cyclopts.App(
    name="eheed",
    help="e-heed: always-on voice channel daemon.",
)

app.command(run_app)
app.command(sentinel_app)
app.command(session_app)
app.command(config_app)

# Trainer commands — available only when train group is installed
with contextlib.suppress(ImportError):
    from trainer.cli import record_app, train_app

    app.command(train_app)
    app.command(record_app)
