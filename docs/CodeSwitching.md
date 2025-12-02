---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# Code Switching Optimization

Different Quantum Error Correction Codes (QECCs) support distinct sets of gates that can be implemented transversally. Transversal
gates, which act on individual physical qubits of different logical code blocks, are inherently fault-tolerant as they do not spread
errors uncontrollably through a quantum circuit. Code switching has been proposed as a technique that employs multiple QECCs
whose respective sets of transversal gates complement each other to achieve universality. Logical qubits are dynamically transferred
between these codes depending on which gate needs to be applied; in other words, the logical information is switched from one code
to the other.

However, code switching is a costly operation in terms of space and time overhead. Therefore, given a quantum circuit, we want to find the **minimum number of switches** required to execute it.

For this documentation we consider the combination of a 2D and 3D color code as a possible QECC pair for code switching.
2D color codes implement, among others, CNOT and Hadamard gates transversally. On the other hand, 3D color codes
have CNOT and T gates in their transversal gate set. The union of both sets provides a universal gate set {$H$,$T$,$CNOT$}.
So for simplicity we will only consider these three gates in the following examples.

We model this problem as a **Min-Cut / Max-Flow** problem on a directed graph. The graph is constructed such that:

- **Source (SRC):** Represents the first code (e.g., 2D Color Code).
- **Sink (SNK):** Represents the second code (e.g., 3D Color Code).
- **Nodes:** Quantum gates in the circuit.
- **Edges:**
  - **Infinite Capacity:** Connect gates unique to one code (e.g., T gates) to their respective terminal (Sink).
  - **Temporal Edges:** Finite capacity edges connecting sequential operations on the same qubit. A "cut" here represents a code switch.

The minimum cut separating the Source from the Sink corresponds to the optimal switching strategy.

## Basic Usage

Let's look at how to use the `MinimalCodeSwitchingCompiler` to analyze a simple quantum circuit. We start by defining the gate sets supported by our two hypothetical codes.

```{code-cell} ipython3
from qiskit import QuantumCircuit
from mqt.qecc.code_switching import MinimalCodeSwitchingCompiler, CompilerConfig

# Define the transversal gate sets (names must be uppercase to match Qiskit's node names)
# Code A (Source): 2D Color Code
SOURCE_GATES = {"H", "CX"}

# Code B (Sink): 3D Color Code
SINK_GATES = {"T", "CX"}

# Initialize the compiler
compiler = MinimalCodeSwitchingCompiler(
    gate_set_code_source=SOURCE_GATES,
    gate_set_code_sink=SINK_GATES
)
```

Next, we create a Qiskit circuit that forces the compiler to make decisions. We will interleave Hadamard gates (Source-favored) and T gates (Sink-favored), separated by CNOTs (Common to both).

```{code-cell} ipython3
qc = QuantumCircuit(6)

qc.h(range(3))
qc.t(range(3,6))

qc.barrier()

qc.cx(1, 4)
qc.cx(3, 4)
qc.cx(2, 3)
qc.cx(2, 4)
qc.cx(0, 4)
qc.cx(5, 3)

qc.barrier()

qc.h(range(3))
qc.t(range(3,6))
```

```{code-cell} ipython3
:tags: [hide-input]
qc.draw('mpl')
```

The only optimization potential lies in the middle for the CNOT portion of the circuit, as the initial and final layers of single qubit gates force us to be in specific codes.
We can now build the graph from the circuit and compute the minimum cut.

```{code-cell} ipython3
# Build the graph representation of the circuit
compiler.build_from_qiskit(qc)

# Compute Min-Cut
num_switches, positions, set_S, set_T = compiler.compute_min_cut()

print(f"Total switches required: {num_switches}")
print("Switch locations (qubit, depth):")
for pos in positions:
    print(f" - Qubit {pos[0]} after operation depth {pos[1]}")
```

The output positions provides the exact locations (qubit, depth) where a code switch operation must be inserted into the circuit.
