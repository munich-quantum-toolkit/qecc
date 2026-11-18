# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Prep synthesis modes: MQT QECC's synthesis built in, other tools plugged in through ``Context.synthesizers``."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from mqt.qecc.circuit_synthesis.ft_search import synth
from mqt.qecc.circuit_synthesis.ft_search._core import memo
from mqt.qecc.circuit_synthesis.ft_search._core.blocks import dual
from mqt.qecc.circuit_synthesis.ft_search._core.verify import CSSState, check_noiseless

H = np.array([[1, 1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 1, 0], [0, 0, 0, 1, 1, 0, 1]], dtype=np.uint8)
STATE = CSSState(H, dual(H))


_CALLS: list[int] = []  # the ``k`` of every call of ``_greedy_tool``


def _greedy_tool(h: np.ndarray, k: int) -> list:
    _CALLS.append(k)
    return synth.candidates(h, "greedy", k)


def test_plugged_in_mode_is_used_and_memoized() -> None:
    """A synthesizer given by the caller serves its mode, and is asked only once per state."""
    _CALLS.clear()
    tools = {"my_tool": _greedy_tool}
    first = synth.candidates(H, "my_tool", 2, synthesizers=tools)
    second = synth.candidates(H, "my_tool", 2, synthesizers=tools)
    assert [str(c.to_stim()) for c in first] == [str(c.to_stim()) for c in second]
    assert _CALLS == [2]
    for c in first:
        check_noiseless(c, STATE)


@pytest.mark.parametrize("mode", ["qecc_heuristic", "qecc_gate", "greedy", "random"])
def test_built_in_modes_prepare_the_state(mode: str) -> None:
    """The built-in modes prepare the state."""
    cands = synth.candidates(H, mode, 2)
    assert cands
    for c in cands:
        check_noiseless(c, STATE)


def test_unknown_mode_raises() -> None:
    """An unknown synthesis mode raises a ValueError."""
    with pytest.raises(ValueError, match="unknown synthesis mode"):
        synth.candidates(H, "no_such_tool", 1)


@pytest.mark.skipif(sys.platform == "win32", reason="shell script")
def test_command_synthesizer_reads_the_external_tool_output(tmp_path) -> None:
    """An external command gets the state as ``<dir>/H.npz`` and its ``cand_H_<i>.stim`` files come back as circuits."""
    script = tmp_path / "tool.sh"
    script.write_text('#!/bin/sh\nprintf "CX 0 1\\n" > "$1/cand_H_0.stim"\n')
    script.chmod(0o755)
    cands = synth.candidates(H[:1, :2], "ext", 3, synthesizers={"ext": synth.CommandSynthesizer(str(script), "x")})
    assert len(cands) == 1
    assert [op.name for op in cands[0].ops if op.name == "CX"] == ["CX"]  # plus implicit resets


def _three_circuits(h: np.ndarray, _k: int) -> list:
    return [c.to_stim() for c in synth.candidates(h, "greedy", 3)]


def _one_circuit(h: np.ndarray, _k: int) -> list:
    return synth.candidates(h, "random", 1)


def test_plugged_in_tool_may_return_stim_circuits_and_is_cut_to_k() -> None:
    """A tool may return stim circuits; at most ``k`` circuits come back."""
    cands = synth.candidates(H, "tool", 2, synthesizers={"tool": _three_circuits})
    assert len(cands) == 2
    for c in cands:
        check_noiseless(c, STATE)


def test_two_tools_under_one_mode_name_are_not_confused() -> None:
    """The memo tells two tools under the same mode name apart."""
    synth.candidates(H, "tool", 1, synthesizers={"tool": _three_circuits})
    second = synth.candidates(H, "tool", 1, synthesizers={"tool": _one_circuit})
    memo.clear()
    assert second == synth.candidates(H, "tool", 1, synthesizers={"tool": _one_circuit})


@pytest.mark.skipif(sys.platform == "win32", reason="shell script")
def test_failing_command_reports_its_error_output(tmp_path) -> None:
    """A failing external tool raises a RuntimeError with its error output."""
    script = tmp_path / "tool.sh"
    script.write_text('#!/bin/sh\necho "no license" >&2\nexit 3\n')
    script.chmod(0o755)
    with pytest.raises(RuntimeError, match="no license"):
        synth.CommandSynthesizer(str(script), "x")(H, 1)


@pytest.mark.skipif(sys.platform == "win32", reason="shell script")
def test_command_synthesizer_keeps_the_first_k_files_by_number(tmp_path) -> None:
    """With more than ``k`` files, the tool's files 0 to k - 1 are kept (``cand_H_10`` comes after ``cand_H_2``)."""
    script = tmp_path / "tool.sh"
    script.write_text(
        "#!/bin/sh\nfor i in 0 1 2 3 4 5 6 7 8 9 10 11; do\n"
        '  j=0; while [ $j -le $i ]; do printf "CX 0 1\\n"; j=$((j + 1)); done > "$1/cand_H_$i.stim"\ndone\n'
    )
    script.chmod(0o755)
    cands = synth.CommandSynthesizer(str(script), "x")(H[:1, :2], 3)
    assert sorted(sum(op.name == "CX" for op in c.ops) for c in cands) == [1, 2, 3]
