# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""example simulation script."""

from __future__ import annotations

import numpy as np
import scipy
from matplotlib import pyplot as plt

from mqt.qecc.analog_information_decoding.simulators.analog_tannergraph_decoding import AtdSimulator
from mqt.qecc.analog_information_decoding.utils.data_utils import BpParams

code_path = "src/mqt/qecc/analog_information_decoding/codes/lifted_product/lp_l="
s = np.linspace(0.10, 0.4, 11)
p = 0.05
for bp_method in ["msl"]:
    for decoder in ["atd"]:
        for c in [16, 21, 30]:
            Hx = scipy.sparse.load_npz(code_path + str(c) + "_hx.txt").toarray()
            Hz = scipy.sparse.load_npz(code_path + str(c) + "_hz.txt").toarray()
            Lx = scipy.sparse.load_npz(code_path + str(c) + "_lx.txt").toarray()
            Lz = scipy.sparse.load_npz(code_path + str(c) + "_lz.txt").toarray()
            lers = []
            ebs = []
            for sigma in s:
                print(sigma)
                sim = AtdSimulator(
                    hx=Hx,
                    lx=Lx,
                    hz=Hz,
                    lz=Lz,
                    codename=str(c),
                    data_err_rate=p,
                    sigma=sigma,
                    seed=666,
                    bp_params=BpParams(
                        max_bp_iter=100,
                        bp_method=bp_method,
                        osd_order=10,
                        osd_method="osd_cs",
                        ms_scaling_factor=0.75,
                        cutoff=5.0,
                    ),
                    decoding_method=decoder,
                )
                out = sim.run(samples=1500)

                ler = (
                    out["z_ler"] * (1 - out["x_ler"]) + out["x_ler"] * (1 - out["z_ler"]) + out["x_ler"] * out["z_ler"]
                )
                ler_eb = (
                    out["z_ler_eb"] * (1 - out["x_ler_eb"])
                    + out["x_ler_eb"] * (1 - out["z_ler_eb"])
                    + out["x_ler_eb"] * out["z_ler_eb"]
                )
                lers.append(ler)
                ebs.append(ler_eb)

            plt.errorbar(
                s,
                lers,
                ebs,
                label=f"l={c}, {decoder} {bp_method}",
                marker="o",
                linestyle="solid",
            )

plt.legend()
plt.xlabel("sigma")
plt.ylabel("LER")
plt.yscale("log")
plt.show()
