"""Minimal entities for Team Doctor.

Only what the diagnosis needs — Members and Workstreams. Kept as plain
dataclasses so the deterministic engine can stay identical to Team OS.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

# RACI role codes. Exactly one Accountable per workstream; one or more Responsible.
RACI_CODES = {
    "A": "Accountable — owns the outcome (exactly one)",
    "R": "Responsible — does the work",
    "C": "Consulted — gives input before it's done",
    "I": "Informed — told after it's done",
    "": "Not involved",
}


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass
class Member:
    id: str
    name: str
    role: str = ""

    @staticmethod
    def create(name: str, role: str = "") -> "Member":
        return Member(id=_new_id("m"), name=name.strip(), role=role.strip())


@dataclass
class Workstream:
    id: str
    name: str
    description: str = ""

    @staticmethod
    def create(name: str, description: str = "") -> "Workstream":
        return Workstream(id=_new_id("w"), name=name.strip(),
                          description=description.strip())
