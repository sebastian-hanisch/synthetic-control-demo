"""Gemeinsame Fixtures: kleine Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import syn_constants as C  # noqa: E402


@pytest.fixture
def flat_world():
    """Mini-Instanz für die Schätzer: 3 Welten, 2 Kontrollen, 24 Perioden. Behandelter Standort vorher 0,90 und nachher 0,95;
    Kontrollen vorher 0,92 und nachher 0,94. Handrechnung: Vorher-Nachher 0,05; Kontrollen 0,02; DiD = 0,05 - 0,02 = 0,03."""
    r, j = 3, 2
    y0 = np.full((r, C.N_PERIODS), 0.90)
    y0[:, C.T0:] = 0.95
    yc = np.full((r, j, C.N_PERIODS), 0.92)
    yc[:, :, C.T0:] = 0.94
    return y0, yc
