---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

```{code-cell} ipython3
:tags: [remove-cell]
%config InlineBackend.figure_formats = ['svg']
```

# Encoder Circuit Synthesis for CSS Codes

QECC provides functionality for synthesizing encoding circuits of arbitrary CSS codes. An encoder for an $[[n,k,d]]$ code is an isometry that encodes $k$ logical qubits into $n$ physical qubits.

Let's consider the synthesis of the encoding circuit of the $[[7,1,3]]$ Steane code.

```{code-cell} ipython3
from mqt.qecc import CSSCode
from mqt.qecc.circuit_synthesis import (
    depth_optimal_encoding_circuit,
    gate_optimal_encoding_circuit,
    heuristic_encoding_circuit,
)

steane_code = CSSCode.from_code_name("steane")

print("Stabilizers:\n")
print(steane_code.stabs_as_pauli_strings())
print("\nLogicals:\n")
print(steane_code.x_logicals_as_pauli_strings())
```

There is not a unique encoding circuit but usually we would like to obtain an encoding circuit that is optimal with respect to some metric. QECC has functionality for synthesizing gate- or depth-optimal encoding circuits.

Under the hood, this uses the SMT solver [z3](https://github.com/Z3Prover/z3). Of course this method scales only up to a few qubits. Synthesizing depth-optimal circuits is usually faster than synthesizing gate-optimal circuits.

```{code-cell} ipython3
depth_opt, q_enc = depth_optimal_encoding_circuit(steane_code, max_timeout=5)

print(f"Encoding qubits are qubits {q_enc}.")
print(f"Circuit has depth {depth_opt.depth()}.")
print(f"Circuit has {depth_opt.num_nonlocal_gates()} CNOTs.")

depth_opt.draw('mpl')
```

```{code-cell} ipython3
gate_opt, q_enc = gate_optimal_encoding_circuit(steane_code, max_timeout=5)

print(f"Encoding qubits are qubits {q_enc}.")
print(f"Circuit has depth {gate_opt.depth()}.")
print(f"Circuit has {gate_opt.num_nonlocal_gates()} CNOTs.")

gate_opt.draw('mpl')
```

QECC obtains optimal solutions for circuits by iteratively trying out different parameters to close in on the optimum. Each run will only be run until the number of seconds specified by `max_timeout`. If a solution is found in this time it is returned. Otherwise, `None` will be returned.

In addition to the circuit, the synthesis methods also return the encoding qubits. All other qubits are assumed to be initialized in the $|0\rangle$ state.

For larger codes, synthesizing optimal circuits is not feasible. In this case, QECC provides a heuristic synthesis method that tries to use as few CNOTs with the lowest depth as possible.

```{code-cell} ipython3
heuristic_circ, q_enc = heuristic_encoding_circuit(steane_code)

print(f"Encoding qubits are qubits {q_enc}.")
print(f"Circuit has depth {heuristic_circ.depth()}.")
print(f"Circuit has {heuristic_circ.num_nonlocal_gates()} CNOTs.")

heuristic_circ.draw('mpl')
```

## Synthesizing Encoders for Concatenated Codes

Encoders for concatenated codes can be constructed by concatenating encoding circuits. We can concatenate the $[[4,2,2]]$ code (with stabilizer generators $XXXX$ and $ZZZZ$) with itself by encoding $4$ qubits into two blocks of the code and then encoding these qubits one more time. This gives an $[[8,4,2]]$ code. The distance is still $2$ but if done the right way, some minimal-weight logicals have weight $4$.

As an exercise, let's construct the concatenated circuit.

We start off by defining the code:

```{code-cell} ipython3
import numpy as np

d = 2
x_stabs = np.ones((1, 4), dtype=np.int8)
z_stabs = x_stabs
code = CSSCode(x_stabs, z_stabs, d)

print("Stabilizers:\n")
print(code.stabs_as_pauli_strings())
print("\nLogicals:\n")
print(code.x_logicals_as_pauli_strings())
print(code.z_logicals_as_pauli_strings())
```

We have to be careful with the logicals. Each _anticommuting_ pair of logicals defines one logical qubit.

As before, we synthesize the encoding circuit:

```{code-cell} ipython3
encoder, q_enc = depth_optimal_encoding_circuit(code, max_timeout=5)

print(f"Encoding qubits are qubits {q_enc}.")
print(f"Circuit has depth {encoder.depth()}.")
print(f"Circuit has {encoder.num_nonlocal_gates()} CNOTs.")

encoder.draw('mpl')
```

Propagating Paulis from the encoding qubits at the input to the output will not necessarily yield the exact logicals given above. But the logical operators will be stabilizer equivalent.

Concatenating the circuits can be done as follows with qiskit:

```{code-cell} ipython3
from qiskit import QuantumCircuit

from mqt.qecc.circuit_synthesis.circuit_utils import reorder_qubits

n = 4

first_layer = QuantumCircuit(n).tensor(encoder)
second_layer = encoder.tensor(encoder)

initialized_qubits = set(range(2 * n)) - set(q_enc) - {q + n for q in q_enc}
qubit_mapping = {q_enc[0]: 0, q_enc[1]: 3, q_enc[0] + n: 2, q_enc[1] + n: 1}
qubit_mapping.update({q: i + 2 * len(q_enc) for i, q in enumerate(initialized_qubits)})
second_layer = reorder_qubits(second_layer, qubit_mapping)

encoder_concat_naive = first_layer.compose(second_layer)

print(f"Encoding qubits are qubits {q_enc}.")
print(f"Circuit has depth {encoder_concat_naive.depth()}.")
print(f"Circuit has {encoder_concat_naive.num_nonlocal_gates()} CNOTs.")

encoder_concat_naive.draw('mpl')
```

Qubits $1$ and $2$ are still the encoding qubits and if we propagate Pauli $X$ and $Z$ to the output, we find that this is indeed the encoder for an $[[8,2,2]]$ code.

This circuit has $3$ times as many CNOT gates as the encoder for the unconcatenated code because we needed to encode 3 times. Instead of concatenating the encoder circuits we can synthesize the encoders directly from the stabilizers of the concatenated code. The stabilizers of the concatenated code are the stabilizers of the original code on the respective subset of qubits with the addition of the "encoded" stabilizers of the inner code. We have a choice of how exactly we encode the stabilizers of the inner code. In the circuit picture, we have a choice of how we "wire the qubits together". Depending on how we do this, the code might have different logical operators.

```{code-cell} ipython3
permutation = [3, 0, 2, 1]

x_prod = (code.Lx[0] + code.Lx[1]) % 2
Hx = np.vstack(
    (
        np.kron(np.eye(2, dtype=np.int8), code.Hx),
        np.hstack((x_prod, x_prod[permutation])),
    )
)

z_prod = (code.Lz[0] + code.Lz[1]) % 2
Hz = np.vstack(
    (
        np.kron(np.eye(2, dtype=np.int8), code.Hz),
        np.hstack((z_prod, z_prod[permutation])),
    )
)

concatenated = CSSCode(Hx, Hz, 4)

print("Stabilizers:\n")
print(concatenated.stabs_as_pauli_strings())

print("\nLogicals:\n")
print(concatenated.x_logicals_as_pauli_strings())
print(concatenated.z_logicals_as_pauli_strings())
```

Now we can directly synthesize the encoder:

```{code-cell} ipython3
encoder_concat_direct, q_enc = depth_optimal_encoding_circuit(
    concatenated, max_timeout=5
)

print(f"Encoding qubits are qubits {q_enc}.")
print(f"Circuit has depth {encoder_concat_direct.depth()}.")
print(f"Circuit has {encoder_concat_direct.num_nonlocal_gates()} CNOTs.")

encoder_concat_direct.draw('mpl')
```

We see that the circuit is more compact then the naively concatenated one. This is because the synthesis method exploits redundancy in the check matrix of the concatenated code.
