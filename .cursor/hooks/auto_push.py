#!/usr/bin/env python3
"""Increment user-command counter; push to origin/main every N commands."""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COUNTER_FILE = ROOT / ".cursor" / "command_counter.json"
LOG_FILE = ROOT / ".cursor" / "auto_push.log"
THRESHOLD = 10
REMOTE = "origin"
BRANCH = "main"


def log(message: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(f"[{stamp}] {message}\n")


def load_counter() -> dict:
    if COUNTER_FILE.exists():
        return json.loads(COUNTER_FILE.read_text(encoding="utf-8"))
    return {"count": 0}


def save_counter(data: dict) -> None:
    COUNTER_FILE.parent.mkdir(parents=True, exist_ok=True)
    COUNTER_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def commits_to_push() -> int:
    upstream = git("rev-parse", "--abbrev-ref", f"{BRANCH}@{REMOTE}")
    if upstream.returncode != 0:
        count = git("rev-list", "--count", f"{REMOTE}/{BRANCH}..HEAD")
        if count.returncode != 0:
            return 0
        return int(count.stdout.strip() or 0)

    count = git("rev-list", "--count", f"{REMOTE}/{BRANCH}..HEAD")
    if count.returncode != 0:
        return 0
    return int(count.stdout.strip() or 0)


def push_if_needed(command_count: int) -> None:
    if not (ROOT / ".git").exists():
        log("Skip push: not a git repository.")
        return

    ahead = commits_to_push()
    if ahead == 0:
        log(f"Command #{command_count}: threshold reached, nothing to push.")
        return

    git("fetch", REMOTE)
    rebase = git("pull", "--rebase", REMOTE, BRANCH)
    if rebase.returncode != 0:
        log(f"Command #{command_count}: rebase failed.\n{rebase.stderr.strip()}")
        return

    push = git("push", REMOTE, BRANCH)
    if push.returncode == 0:
        log(f"Command #{command_count}: pushed {ahead} commit(s) to {REMOTE}/{BRANCH}.")
    else:
        log(f"Command #{command_count}: push failed.\n{push.stderr.strip()}")


def main() -> int:
    data = load_counter()
    data["count"] = int(data.get("count", 0)) + 1
    command_count = data["count"]
    data["last_updated"] = datetime.now(timezone.utc).isoformat()
    save_counter(data)

    if command_count % THRESHOLD == 0:
        push_if_needed(command_count)

    return 0


if __name__ == "__main__":
    sys.exit(main())
