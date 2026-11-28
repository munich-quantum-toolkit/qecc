# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""LightsOut MaxSAT-based decoder for the hexagonal color code."""

from __future__ import annotations

import json
import locale
import subprocess  # ruff:ignore[suspicious-subprocess-import]
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from timeit import default_timer as timer
from typing import TYPE_CHECKING, Any

import numpy as np
from z3 import Bool, BoolVal, Not, Optimize, Xor, simplify

from . import code_from_string

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt
    from z3 import BoolRef, ModelRef

    from ..codes.constructions.color_codes.color_code import ColorCode


# Exit codes of MaxSAT solvers in the MaxSAT Evaluations: 0 for an unknown result, 10 for a satisfiable problem,
# 20 for an unsatisfiable problem, and 30 for an optimal solution.
MAXSAT_EXIT_CODES = frozenset({0, 10, 20, 30})


@dataclass
class LightsOut:
    """Lights out problem representation."""

    lights_to_switches: dict[int, list[int]]
    switches_to_lights: dict[int, list[int]]
    switch_vars: list[BoolRef] | None = None
    helper_vars: dict[int, list[BoolRef]] = field(default_factory=dict)
    optimizer: Optimize = field(default_factory=Optimize)

    def preconstruct_parity_constraint(self, light: int, indices: list[int]) -> None:
        """Preconstruct the parity constraints for a light.

        Adds all constraint to the optimizer that are independent of the value of the light.
        """
        helper_vars = self.helper_vars[light]
        if len(indices) <= 1:
            # nothing to preconstruct for no switch or a single switch
            return

        assert self.switch_vars is not None
        for i in range(1, len(indices) - 1):
            constraint = Xor(self.switch_vars[indices[i]], helper_vars[i]) == helper_vars[i - 1]
            self.optimizer.add(simplify(constraint))

        constraint = self.switch_vars[indices[-1]] == helper_vars[-1]
        self.optimizer.add(simplify(constraint))

    def complete_parity_constraint(self, light: int, indices: list[int], val: bool) -> None:
        """Completes the parity constraints for a light.

        Adds the constraint that is dependent on the value of the light.
        """
        helper_vars = self.helper_vars[light]
        assert self.switch_vars is not None
        if not indices:
            # No switch controls the light, so a light that is on makes the problem unsatisfiable.
            if val:
                self.optimizer.add(BoolVal(False))
            return
        if len(indices) == 1:
            constraint = self.switch_vars[indices[0]] == val
        else:
            constraint = Xor(self.switch_vars[indices[0]], helper_vars[0]) == val
        self.optimizer.add(simplify(constraint))

    def preconstruct_z3_instance(self, weights: npt.NDArray[np.float64] | None = None) -> None:
        """Preconstruct the z3 instance for the lights-out problem.

        Creates all necessary variables, adds the known parts of the parity constraints.
        Soft constraints are added to the optimizer with default weights.
        """
        if self.switch_vars is None:
            self.switch_vars = [Bool(f"switch_{i}") for i in range(len(self.switches_to_lights))]

        for light, switches in self.lights_to_switches.items():
            if light not in self.helper_vars:
                self.helper_vars[light] = [Bool(f"helper_{light}_{i}") for i in range(len(switches) - 1)]
            self.preconstruct_parity_constraint(light, switches)

        if weights is not None and len(weights) > 0:
            for switch, weight in zip(self.switch_vars, weights, strict=False):
                self.optimizer.add_soft(Not(switch), weight)
        else:
            for switch in self.switch_vars:
                self.optimizer.add_soft(Not(switch))

    def validate_model(self, model: ModelRef, lights: list[bool]) -> bool:
        """Validate the model by checking if pressing the switches turns off all lights."""
        assert self.switch_vars is not None
        # Work on a copy, so that the caller's light pattern stays unchanged.
        lights = list(lights)
        for i, var in enumerate(self.switch_vars):
            if model[var]:
                # flip all lights that are controlled by this switch
                for light in self.switches_to_lights[i]:
                    lights[light] = not lights[light]

        return all(not light for light in lights)

    def count_switches(self, model: ModelRef) -> int:
        """Count the number of switches that are set to true."""
        assert self.switch_vars is not None
        return sum(1 for var in self.switch_vars if model[var])

    def solve(self, lights: list[bool], solver_path: str = "z3", timeout: int = 1800) -> tuple[list[int], bool, float]:
        """Solve the lights-out problem for a given pattern.

        Assumes that the z3 instance has already been pre-constructed.
        A solver other than z3 only gets timed: the method writes the problem to a WCNF file, passes the path of
        this file to the solver, and does not read the solver output.

        Args:
            lights: The light pattern. The method does not change this list.
            solver_path: ``"z3"``, or the path to an external MaxSAT solver executable.
            timeout: The time limit of the z3 solver in milliseconds.

        Returns:
            The switches to press, whether the solver found a valid solution, and the time spent in the solver call.
            For an external solver, the list of switches is empty and the flag is always True, because the method
            does not read the solver output.

        Raises:
            RuntimeError: If an external solver exits with a code that is not in :data:`MAXSAT_EXIT_CODES`.
        """
        # push a new context to the optimizer
        self.optimizer.push()
        self.optimizer.set("timeout", timeout)
        # add the problem specific constraints
        for light, val in enumerate(lights):
            self.complete_parity_constraint(light, self.lights_to_switches[light], val)
        switches: list[int] = []
        if solver_path == "z3":
            # solve the problem
            start = timer()
            result = self.optimizer.check()
            solve_time = timer() - start
            if str(result) != "sat":
                self.optimizer.pop()
                res_string = [0] * len(self.switch_vars) if self.switch_vars is not None else []
                return res_string, False, solve_time

            # validate the model
            model = self.optimizer.model()
            if self.validate_model(model, lights) is False:
                self.optimizer.pop()
                res_string = [0] * len(self.switch_vars) if self.switch_vars is not None else []
                return res_string, False, solve_time

            assert self.switch_vars is not None
            switches = [1 if model[var] else 0 for var in self.switch_vars]
        else:
            self.optimizer.set("pp.wcnf", True)
            wcnf = str(self.optimizer)
            # Note: This merely calls the solver. It does not interpret the output.
            #       This is just to measure the time it takes to solve the problem.
            with (
                tempfile.TemporaryDirectory() as tmp_dir,
                Path("./solver-out_" + solver_path.rsplit("/", maxsplit=1)[-1] + ".txt").open(
                    "a+", encoding=locale.getpreferredencoding(False)
                ) as out,
            ):
                wcnf_path = Path(tmp_dir) / "problem.wcnf"
                wcnf_path.write_text(wcnf, encoding="utf-8")
                start = timer()
                completed = subprocess.run([solver_path, str(wcnf_path)], stdout=out, check=False)  # ruff:ignore[subprocess-without-shell-equals-true]
                solve_time = timer() - start
            if completed.returncode not in MAXSAT_EXIT_CODES:
                self.optimizer.pop()
                msg = f"The MaxSAT solver {solver_path} failed with exit code {completed.returncode}."
                raise RuntimeError(msg)

        # pop the context from the optimizer
        self.optimizer.pop()

        return switches, True, solve_time


def simulate_error_rate(code: ColorCode, error_rate: float, nr_sims: int, solver_path: str = "z3") -> dict[str, Any]:
    """Simulate the logical error rate for a given distance and error rate.

    Only the z3 solver returns a decoding estimate. An external solver only gets timed, so the logical error rates
    and their error bars are NaN in that case, and ``nr_unsolved`` is None.

    Args:
        code: The color code to simulate.
        error_rate: The bit-flip error rate.
        nr_sims: The number of samples.
        solver_path: ``"z3"``, or the path to an external MaxSAT solver executable.

    Returns:
        The simulation results. ``avg_constr_time`` is the average time per sample in
        :meth:`LightsOut.solve` outside the solver call, mainly for building the constraints of the sample.
        ``nr_unsolved`` is the number of samples for which the solver did not find a valid solution.
    """
    timing_only = solver_path != "z3"
    problem = LightsOut(code.faces_to_qubits, code.qubits_to_faces)

    start = timer()
    problem.preconstruct_z3_instance()
    preconstr_time = timer() - start
    min_wt_logicals: npt.NDArray[np.int_] = np.full(len(code.L), -1).astype(int)
    logical_errors: npt.NDArray[np.int_] = np.zeros(len(code.L)).astype(int)
    avg_constr_time = 0.0
    avg_solve_time = 0.0
    nr_unsolved = 0
    rng = np.random.default_rng()
    for i in range(nr_sims):
        # sample error
        error = rng.choice([0, 1], size=code.n, p=[1 - error_rate, error_rate])

        # get syndrome
        syndrome = code.get_syndrome(error)
        lights = [bool(b) for b in syndrome]

        # compute estimate
        start = timer()
        estimate, solved, solve_time = problem.solve(lights, solver_path=solver_path)
        constr_time = timer() - start - solve_time
        if not solved:
            nr_unsolved += 1
        if not timing_only:
            # check if the estimate is correct
            residual = (error + np.array(estimate)) % 2
            for logical in range(len(code.L)):
                if (code.L[logical] @ residual % 2).any():
                    logical_errors[logical] += 1
                    wt: int = np.sum(residual)  # compute the min weight of a logical error
                    if min_wt_logicals[logical] == -1 or wt < min_wt_logicals[logical]:
                        min_wt_logicals[logical] = int(wt)
                    break

        # compute rolling average of the times
        avg_constr_time = (avg_constr_time * i + constr_time) / (i + 1)
        avg_solve_time = (avg_solve_time * i + solve_time) / (i + 1)

    logical_error_rates: list[float] = [nr_errors / nr_sims for nr_errors in logical_errors]
    logical_error_rate_ebs: list[float] = [np.sqrt((1 - ler) * ler / nr_sims) for ler in logical_error_rates]
    if timing_only:
        logical_error_rates = [float("nan")] * len(code.L)
        logical_error_rate_ebs = [float("nan")] * len(code.L)
    avg_total_time = avg_constr_time + avg_solve_time

    return {
        "lattice": code.lattice_type,
        "distance": code.distance,
        "p": error_rate,
        "logical_error_rates": logical_error_rates,
        "logical_error_rate_ebs": logical_error_rate_ebs,
        "preconstr_time": preconstr_time,
        "avg_constr_time": avg_constr_time,
        "avg_solve_time": avg_solve_time,
        "avg_total_time": avg_total_time,
        "min_wts_logical_err": min_wt_logicals.tolist(),
        "nr_unsolved": None if timing_only else nr_unsolved,
    }


def run(
    lattice_type: str,
    distance: int,
    error_rate: float,
    nr_sims: int = 10000,
    results_dir: str = "./results_maxsat",
    solver: str = "z3",
) -> None:
    """Run the decoding simulation for a given distance and error rate."""
    code = code_from_string(lattice_type, distance)
    data = simulate_error_rate(code, error_rate, nr_sims, solver)
    strg = solver.rsplit("/", maxsplit=1)[-1]
    filename = f"./code={code.lattice_type},distance={code.distance},p={round(error_rate, 4)},solver={strg}.json"
    path = Path(results_dir)
    path.mkdir(parents=True, exist_ok=True)
    with (path / filename).open("w") as out:
        out.write(json.dumps(data))
