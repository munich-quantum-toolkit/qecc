# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Generate the lifted hypergraph product of the protographs a and b. From Roffe's LDPC library."""

from __future__ import annotations

from typing import TYPE_CHECKING

# taken from https://github.com/quantumgizmos/bias_tailored_qldpc
import ldpc.protograph as pt
import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt
from bposd.css import css_code
from bposd.stab import stab_code


def identity(n: int) -> npt.NDArray[np.int_]:
    """Return the identity matrix of size n."""
    return pt.identity(n)


class LiftedHgp(css_code):
    """Lifted Hypergraph Product code."""

    def __init__(
        self,
        lift_parameter: int,
        a: npt.NDArray[np.int_],
        b: npt.NDArray[np.int_] | None = None,
    ) -> None:
        """Generate the lifted hypergraph product of the protographs a and b."""
        self.a = a

        self.a_m, self.a_n = self.a.shape

        if b is None:
            self.b = pt.copy(self.a)
        else:
            self.b = b

        self.b_m, self.b_n = self.b.shape

        self.hx1_proto = np.kron(self.a, identity(self.b_n))
        self.hx2_proto = np.kron(identity(self.a_m), self.b.T)
        self.hx_proto = pt.hstack([self.hx1_proto, self.hx2_proto])

        self.hz1_proto = np.kron(identity(self.a_n), self.b)
        self.hz2_proto = np.kron(self.a.T, identity(self.b_m))
        self.hz_proto = pt.hstack([self.hz1_proto, self.hz2_proto])

        self.lift_parameter = lift_parameter

        super().__init__(
            self.hx_proto.to_binary(lift_parameter),
            self.hz_proto.to_binary(lift_parameter),
        )

    @property
    def protograph(self) -> npt.NDArray[np.int_]:
        """Returns the protograph of the lifted hypergraph product."""
        px = pt.vstack([pt.zeros(self.hz_proto.shape), self.hx_proto])
        pz = pt.vstack([self.hz_proto, pt.zeros(self.hx_proto.shape)])
        return pt.hstack([px, pz])

    @property
    def hx1(self) -> npt.NDArray[np.int_]:
        """Returns the first horizontal protograph of the lifted hypergraph product."""
        return self.hx1_proto.to_binary(self.lift_parameter)

    @property
    def hx2(self) -> npt.NDArray[np.int_]:
        """Returns the second horizontal protograph of the lifted hypergraph product."""
        return self.hx2_proto.to_binary(self.lift_parameter)

    @property
    def hz1(self) -> npt.NDArray[np.int_]:
        """Returns the first vertical protograph of the lifted hypergraph product."""
        return self.hz1_proto.to_binary(self.lift_parameter)

    @property
    def hz2(self) -> npt.NDArray[np.int_]:
        """Returns the second vertical protograph of the lifted hypergraph product."""
        return self.hz2_proto.to_binary(self.lift_parameter)


class BiasTailoredLiftedProduct(stab_code):
    """Generate the bias-tailored lifted hypergraph product of the protographs a and b. From Roffe's LDPC library."""

    def __init__(
        self,
        lift_parameter: int,
        a: npt.NDArray[np.int_],
        b: npt.NDArray[np.int_] | None = None,
    ) -> None:
        """Generate the bias-tailored lifted hypergraph product of the protographs a and b."""
        lhgp = LiftedHgp(lift_parameter, a, b)

        # Hadamard rotation
        temp1 = pt.hstack([pt.zeros(lhgp.hx1_proto.shape), lhgp.hz2_proto])
        temp2 = pt.hstack([lhgp.hx1_proto, pt.zeros(lhgp.hz2_proto.shape)])
        self.hx_proto = pt.vstack([temp1, temp2])
        temp1 = pt.hstack([lhgp.hz1_proto, pt.zeros(lhgp.hx2_proto.shape)])
        temp2 = pt.hstack([pt.zeros(lhgp.hz1_proto.shape), lhgp.hx2_proto])
        self.hz_proto = pt.vstack([temp1, temp2])

        super().__init__(
            self.hx_proto.to_binary(lift_parameter),
            self.hz_proto.to_binary(lift_parameter),
        )

    @property
    def protograph(self) -> npt.NDArray[np.int_]:
        """Returns the protograph of the lifted hypergraph product."""
        px = pt.vstack([pt.zeros(self.hz_proto.shape), self.hx_proto])
        pz = pt.vstack([self.hz_proto, pt.zeros(self.hx_proto.shape)])
        return pt.hstack([px, pz])
