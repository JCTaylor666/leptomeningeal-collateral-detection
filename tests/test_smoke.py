#!/usr/bin/env python3
"""Minimal smoke test for single-frame inference.

Runs scripts/infer_one.py on the bundled example frame and asserts that the
four expected output files are produced. This is a functional smoke test
(does the pipeline run end-to-end and emit outputs?), not a numerical
regression test.

Requires:
  - ckpt/ populated (see ckpt/README.md),
  - a local DINOv3 repo, set via LMC_DINOV3_REPO or --dino_repo_dir,
  - a CUDA device.

Run:
    LMC_DINOV3_REPO=/path/to/dinov3 python tests/test_smoke.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = REPO_ROOT / "examples" / "sample_frame" / "s50_f12.png"

EXPECTED_OUTPUTS = [
    "s50_f12_pred.json",
    "s50_f12_overlay.png",
    "s50_f12_vessel_mask.png",
    "s50_f12_graph_pred.json",
]


def test_infer_one_smoke() -> None:
    assert EXAMPLE.is_file(), f"missing example frame: {EXAMPLE}"
    with tempfile.TemporaryDirectory(prefix="lmc_smoke_") as tmp:
        out_dir = Path(tmp) / "s50_f12"
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "infer_one.py"),
            "--image", str(EXAMPLE),
            "--output_dir", str(out_dir),
            "--threshold", "0.5",
        ]
        subprocess.run(cmd, check=True)
        for name in EXPECTED_OUTPUTS:
            assert (out_dir / name).is_file(), f"missing output: {name}"
    print("smoke test OK: all 4 outputs produced")


if __name__ == "__main__":
    test_infer_one_smoke()
