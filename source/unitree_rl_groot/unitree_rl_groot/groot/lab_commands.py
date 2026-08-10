"""Non-blocking command inbox for a persistent Isaac laboratory session."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from enum import Enum

from unitree_rl_groot.groot.multitask import G1FruitTask, resolve_fruit_task


class LabCommandKind(str, Enum):  # noqa: UP042 -- SONIC runtime is Python 3.10
    MISSION = "mission"
    HELP = "help"
    STATUS = "status"
    STOP = "stop"
    RESET = "reset"
    QUIT = "quit"
    INVALID = "invalid"


@dataclass(frozen=True)
class LabCommand:
    kind: LabCommandKind
    text: str
    task: G1FruitTask | None = None
    error: str | None = None


_CONTROL_COMMANDS = {
    "help": LabCommandKind.HELP,
    "?": LabCommandKind.HELP,
    "status": LabCommandKind.STATUS,
    "stop": LabCommandKind.STOP,
    "dung": LabCommandKind.STOP,
    "dừng": LabCommandKind.STOP,
    "reset": LabCommandKind.RESET,
    "quit": LabCommandKind.QUIT,
    "exit": LabCommandKind.QUIT,
    "thoat": LabCommandKind.QUIT,
    "thoát": LabCommandKind.QUIT,
}


def parse_lab_command(text: str) -> LabCommand:
    """Parse one control command or bind one safe fruit mission."""

    normalized = " ".join(text.split())
    if not normalized:
        return LabCommand(LabCommandKind.INVALID, text, error="empty command")
    control = _CONTROL_COMMANDS.get(normalized.lower())
    if control is not None:
        return LabCommand(control, normalized)
    try:
        task = resolve_fruit_task(normalized)
    except ValueError as exc:
        return LabCommand(LabCommandKind.INVALID, normalized, error=str(exc))
    return LabCommand(LabCommandKind.MISSION, normalized, task=task)


class LabCommandInbox:
    """Thread-safe inbox whose stdin reader never blocks the simulation loop."""

    def __init__(self) -> None:
        self._queue: queue.SimpleQueue[LabCommand] = queue.SimpleQueue()
        self._reader: threading.Thread | None = None

    def submit(self, text: str) -> LabCommand:
        command = parse_lab_command(text)
        self._queue.put(command)
        return command

    def poll(self) -> LabCommand | None:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def start_stdin_reader(self) -> None:
        if self._reader is not None:
            return

        def read_commands() -> None:
            while True:
                try:
                    line = input("lab> ")
                except (EOFError, KeyboardInterrupt):
                    return
                self.submit(line)

        self._reader = threading.Thread(target=read_commands, name="lab-command-reader", daemon=True)
        self._reader.start()
