"""serve/test_keep_heads.py - long prompt heads kept in the idle batch slots (serve/keep_heads.py, STRATA_KEEP_HEADS=1).

    python3 -m unittest serve.test_keep_heads -v

Skeleton (2026-11-25): the cases to write are listed in strata-mesh IMPL-PLAN-2026-11-25-keep-heads.md section 5.3.
The pure rules (no engine), pick_slot with the rule on and off, and the server path over the fake engine of
serve/test_parallel.py (FakeEngineCase: GEN / BGEN / BT / BDONE / BADM / BSTOP lines, `--log` of each command).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import keep_heads  # noqa: E402,F401


class Env(unittest.TestCase):
    """STRATA_KEEP_HEADS / STRATA_KEEP_HEADS_MIN, and KeepHeads.usable()."""


class Rules(unittest.TestCase):
    """turn_points, head_of, served_by, value_of, loss_of, request_pin, request_ckpt."""


class Decide(unittest.TestCase):
    """KeepHeads.decide / check_reuse / note_admitted / held_after_admission."""


class Pick(unittest.TestCase):
    """KeepHeads.pick and StrataEngine.pick_slot with the rule on (off: exactly the old order)."""


class ServerPath(unittest.TestCase):
    """generate_batched over the fake engine: the commands it sends with STRATA_KEEP_HEADS unset and set."""


if __name__ == "__main__":
    unittest.main()
