"""TLP session state machine."""
from __future__ import annotations

from enum import Enum


class State(str, Enum):
    CLOSED = "CLOSED"
    HELLO_SENT = "HELLO_SENT"
    HELLO_RCVD = "HELLO_RCVD"
    VERIFYING = "VERIFYING"
    NEGOTIATING = "NEGOTIATING"
    KEYING = "KEYING"
    ESCROW = "ESCROW"
    ESTABLISHED = "ESTABLISHED"
    CLOSING = "CLOSING"


S = State
TRANSITIONS: dict[State, set[State]] = {
    S.CLOSED: {S.HELLO_SENT, S.HELLO_RCVD},
    S.HELLO_SENT: {S.VERIFYING, S.CLOSED},
    S.HELLO_RCVD: {S.VERIFYING, S.CLOSED},
    S.VERIFYING: {S.NEGOTIATING, S.KEYING, S.CLOSED},
    S.NEGOTIATING: {S.VERIFYING, S.CLOSED},
    S.KEYING: {S.ESCROW, S.ESTABLISHED, S.CLOSED},  # ESTABLISHED directly only with audit off
    S.ESCROW: {S.ESTABLISHED, S.CLOSED},
    S.ESTABLISHED: {S.CLOSING, S.CLOSED},
    S.CLOSING: {S.CLOSED},
}


class StateError(RuntimeError):
    pass


class StateMachine:
    def __init__(self, on_change=None):
        self.state = State.CLOSED
        self.history: list[State] = [State.CLOSED]
        self.on_change = on_change

    def to(self, new: State, reason: str = ""):
        if new not in TRANSITIONS[self.state]:
            raise StateError(f"illegal transition {self.state.value} -> {new.value}")
        old, self.state = self.state, new
        self.history.append(new)
        if self.on_change:
            self.on_change(old, new, reason)

    def reset(self, reason: str = ""):
        """Drop to CLOSED from any state (RESET)."""
        if self.state != State.CLOSED:
            old, self.state = self.state, State.CLOSED
            self.history.append(State.CLOSED)
            if self.on_change:
                self.on_change(old, State.CLOSED, reason)

    @property
    def established(self) -> bool:
        return self.state == State.ESTABLISHED

    def accepts(self, msg_type) -> bool:
        """DATA (and its ACKs) are accepted only in ESTABLISHED."""
        from .header import MsgType
        if msg_type in (MsgType.DATA,):
            return self.state == State.ESTABLISHED
        return True
