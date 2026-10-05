"""Unbuffered subprocess output and durable progress, without ML imports."""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path


class ProgressReporter:
    def __init__(self, output_dir: Path, *, enabled: bool = True):
        self.output_dir = output_dir
        self.enabled = enabled
        self.started = time.monotonic()

    def emit(self, phase: str, **fields) -> None:
        if not self.enabled:
            return
        record = {"time_utc": datetime.now(UTC).isoformat(), "phase": phase,
                  "elapsed_seconds": round(time.monotonic() - self.started, 1), **fields}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        text = json.dumps(record, ensure_ascii=False)
        temporary = self.output_dir / "progress.json.tmp"
        temporary.write_text(text + "\n", encoding="utf-8")
        temporary.replace(self.output_dir / "progress.json")
        with (self.output_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print("[progress] " + text, flush=True)


def stream_command(command, *, log_path: Path, status_path: Path | None = None,
                   heartbeat_seconds: float = 30.0) -> None:
    """Show both streams, retain errors, and report liveness during silent work.

    A heartbeat proves only that the child has not exited, never GPU progress.
    No silence timeout kills slow training or Drive checkpoint writes.
    """
    if heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be positive")
    command = list(map(str, command))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = last_output = time.monotonic()
    last_line = ""
    messages = queue.Queue()

    def status(value, **extra):
        if status_path is not None:
            record = {"status": value, "time_utc": datetime.now(UTC).isoformat(),
                      "elapsed_seconds": round(time.monotonic() - started, 1),
                      "silent_seconds": round(time.monotonic() - last_output, 1),
                      "last_output": last_line[-2000:], "command": command, **extra}
            status_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = status_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            temporary.replace(status_path)

    with log_path.open("a", encoding="utf-8") as logfile:
        def show(line):
            print(line, end="", flush=True)
            logfile.write(line)
            logfile.flush()

        show(f"\n[command] {datetime.now(UTC).isoformat()} {subprocess.list2cmdline(command)}\n")
        status("starting")
        process = None
        try:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )

            def read_output():
                try:
                    for line in process.stdout:
                        messages.put(line)
                    process.wait()
                finally:
                    messages.put(None)

            reader = threading.Thread(target=read_output, daemon=True)
            reader.start()
            status("running", pid=process.pid)
            last_status = time.monotonic()
            while True:
                try:
                    line = messages.get(timeout=heartbeat_seconds)
                except queue.Empty:
                    show(f"[heartbeat] elapsed={time.monotonic() - started:.0f}s | "
                         f"no child output for {time.monotonic() - last_output:.0f}s | "
                         "child has not exited; this is NOT proof of training progress\n")
                    status("running", pid=process.pid)
                    continue
                if line is None:
                    break
                last_output = time.monotonic()
                last_line = line.rstrip()
                show(line)
                if last_output - last_status >= heartbeat_seconds:
                    status("running", pid=process.pid)
                    last_status = last_output
            returncode = process.wait()
            reader.join(timeout=1)
            process.stdout.close()
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)
            status("complete", returncode=returncode)
            show(f"[complete] elapsed={time.monotonic() - started:.1f}s\n")
        except BaseException as error:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            status("failed", error=str(error))
            show(f"[failed] {error}\nFull log: {log_path}\n")
            raise
