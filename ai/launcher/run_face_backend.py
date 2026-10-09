"""Run the face (foreground) and the voice loop (background) with one command.

Usage:
    python3 -m ai.launcher.run_face_backend --fullscreen
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading

from ai.frontend.app import FrontendApp

BACKEND_LOG = "voice_backend.log"


def _tee_backend_output(proc: subprocess.Popen, log_path: str) -> None:
    """Mirror backend output to the log file and this terminal."""
    with open(log_path, "a", encoding="utf-8") as log_file:
        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()
            sys.stdout.write(f"[Backend] {line}")
            sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GangubAI face + voice loop together")
    parser.add_argument("--fullscreen", action="store_true", help="Start the face in fullscreen")
    args = parser.parse_args()

    backend = subprocess.Popen(
        [sys.executable, "-m", "ai.voice.app"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    threading.Thread(target=_tee_backend_output, args=(backend, BACKEND_LOG), daemon=True).start()
    print(f"[Launcher] Voice backend started (pid={backend.pid}), logging to {BACKEND_LOG}", flush=True)

    try:
        return FrontendApp(fullscreen=args.fullscreen).run()
    finally:
        backend.terminate()
        try:
            backend.wait(timeout=4)
        except subprocess.TimeoutExpired:
            backend.kill()


if __name__ == "__main__":
    raise SystemExit(main())
