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

QECC has functionality to automatically determine the minimum number of switching operations required to perform a circuit using two complementary gate sets.
The problem can be modelled as a **Min-Cut / Max-Flow** problem on a directed graph. The graph is constructed such that:

- **Source (SRC):** Represents the first code (e.g., 2D Color Code).
- **Sink (SNK):** Represents the second code (e.g., 3D Color Code).
- **Nodes:** Quantum gates in the circuit.
- **Edges:**
  - **Infinite Capacity:** Connect gates unique to one code (e.g., T gates) to their respective terminal (Sink).
  - **Temporal Edges:** Finite capacity edges connecting sequential operations on the same qubit. A "cut" here represents a code switch.

The minimum cut separating the Source from the Sink corresponds to the optimal switching strategy.

## Basic Usage

Let's look at how to use the `MinimalCodeSwitchingCompiler` to analyze a simple quantum circuit. Assume the two codes in question are the 2D and 3D color codes, which have transversal gate sets $\{H, CX\}$ and $\{T, CX\}$, respectively.

```{code-cell} ipython3
from qiskit import QuantumCircuit
from mqt.qecc.code_switching import MinimalCodeSwitchingCompiler, CompilerConfig

# Define the transversal gate sets:
# Code A (Source): 2D Color Code
SOURCE_GATES = {"H", "CX"}

# Code B (Sink): 3D Color Code
SINK_GATES = {"T", "CX"}

# Initialize the compiler
mcsc = MinimalCodeSwitchingCompiler(
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
mcsc.build_from_qiskit(qc)

# Compute Min-Cut
num_switches, switch_pos, set_S, set_T = mcsc.compute_min_cut()

print(f"Total switches required: {num_switches}")
print("Switch locations (qubit, depth):")
for pos in switch_pos:
    print(f" - Qubit {pos[0]} after operation depth {pos[1]}")
```

The output positions provide the exact locations (qubit, depth) where a code switch operation must be inserted into the circuit.

Note that under specific conditions, CNOT operations can be implemented transversally even when the control and target qubits
are encoded in different codes. This property, however, is directional. In the 2D-3D color code scheme, it holds only when the
control qubit is encoded in the 3D color code and the target qubit in the 2D color code.

To account for this, we can pass a dictionary specifying gates that can be implemented one-way transverally together with their direction.
To see how this affect the optimization, consider the following circuit:

```{code-cell} ipython3
qc = QuantumCircuit(2)
qc.t(0)
qc.h(1)
qc.cx(0, 1)
```

```{code-cell} ipython3
:tags: [hide-input]
qc.draw('mpl')
```

Calculating the minimum number of switches without considering the one-way transversal CNOT property yields:

```{code-cell} ipython3
mcsc = MinimalCodeSwitchingCompiler({"H", "CX"}, {"T", "CX"})
mcsc.build_from_qiskit(qc)
num_switches, switch_pos, _, _ = mcsc.compute_min_cut()
```

```{code-cell} ipython3
:tags: [hide-input]
print(f"Total switches required (without one-way CNOT): {num_switches}")
print("Switch locations (qubit, depth):")
for pos in switch_pos:
    print(f" - Qubit {pos[0]} after operation depth {pos[1]}")
```

Hence, a single switch right after the T gate on qubit 0 is required. However, if we consider the one-way transversal CNOT property, we can avoid this switch:

```{code-cell} ipython3
mcsc = MinimalCodeSwitchingCompiler({"H", "CX"}, {"T", "CX"})
mcsc.build_from_qiskit(qc, one_way_gates={"CX": ("SNK", "SRC")})
num_switches, switch_pos, _, _ = mcsc.compute_min_cut()
```

```{code-cell} ipython3
:tags: [hide-input]
print(f"Total switches required: {num_switches}")
```

We specify in the graph-construction method `build_from_qiskit` that CNOTs (`CX`) are implemented transversally when the control qubit is encoded in the Sink code (3D color code) and the target qubit is encoded in the Source code (2D color code), i.e., `(control, target) <=> ("SNK", "SRC")`.

## Extensions to the Min-Cut Model

Finding the minimum number of switches is a good starting point, but in practice, we might want to consider additional factors such as:

- **Depth Optimization:** Choosing the placing of the switching positions such that switching operations are placed preferably on idling qubits while keeping the total number of switches minimal. This has the potential to reduce the overall circuit depth increase caused by the switching operations.
- **Code Bias:** If one of the codes has a significantly higher overhead for switching operations, we might want to minimize switches into that code specifically.

### Depth Optimization

To incorporate depth optimization, we can assign an idle bonus to weights of the temporal edges based on whether the qubit is idling or active. For example, we can assign a lower weight to edges corresponding to idling qubits, encouraging the min-cut algorithm to place switches there.

```{code-cell} ipython3
qc = QuantumCircuit(3)
qc.h(0)
qc.t(1)
qc.t(2)
qc.cx(1, 2)
qc.cx(0, 2)
```

```{code-cell} ipython3
:tags: [hide-input]
qc.draw('mpl')
```

Running the regular min-cut computation yields a switch on qubit 2 after the T gate (we allow one-way CNOTs here):

```{code-cell} ipython3
mcsc = MinimalCodeSwitchingCompiler({"H", "CX"}, {"T", "CX"})
mcsc.build_from_qiskit(qc, one_way_gates={"CX": ("SNK", "SRC")})
num_switches, switch_pos, _, _ = mcsc.compute_min_cut()
```

```{code-cell} ipython3
:tags: [hide-input]
print(f"Total switches required (with one-way CNOT): {num_switches}")
print("Switch locations (qubit, depth):")
for pos in switch_pos:
    print(f" - Qubit {pos[0]} after operation depth {pos[1]}")
```

However, if we assign a lower weight to the temporal edge of qubit 0 (which is idling), the algorithm chooses to place the switch there instead:

```{code-cell} ipython3
mcsc = MinimalCodeSwitchingCompiler({"H", "CX"}, {"T", "CX"})
mcsc.build_from_qiskit(qc, one_way_gates={"CX": ("SNK", "SRC")}, idle_bonus=True)
num_switches, switch_pos, _, _ = mcsc.compute_min_cut()
```

```{code-cell} ipython3
:tags: [hide-input]
print(f"Total switches required (with one-way CNOT): {num_switches}")
print("Switch locations (qubit, depth):")
for pos in switch_pos:
    print(f" - Qubit {pos[0]} after operation depth {pos[1]}")
```

### Code Bias

To minimize switches into a specific code, we can add a certain code bias. The implementation already has a bias towards the source code.

```{code-cell} ipython3
qc = QuantumCircuit(2)
qc.h(1)
qc.t(0)
qc.cx(0, 1)
```

```{code-cell} ipython3
:tags: [hide-input]
qc.draw('mpl')
```

The default behavior with a bias towards the source code yields that the switch is placed after the T gate on qubit 0 such that the CNOT is in the source code:

```{code-cell} ipython3
mcsc = MinimalCodeSwitchingCompiler({"H", "CX"}, {"T", "CX"})
mcsc.build_from_qiskit(qc, one_way_gates={"CX": ("SNK", "SRC")})
num_switches, switch_pos, _, _ = mcsc.compute_min_cut()
```

```{code-cell} ipython3
:tags: [hide-input]
print(f"Total switches required (with one-way CNOT): {num_switches}")
print("Switch locations (qubit, depth):")
for pos in switch_pos:
    print(f" - Qubit {pos[0]} after operation depth {pos[1]}")
```

However, if we wanted to instead bias towards the sink code, we could adjust the compiler configuration as follows:

```{code-cell} ipython3
config = CompilerConfig(biased_code="SNK")
mcsc = MinimalCodeSwitchingCompiler({"H", "CX"}, {"T", "CX"}, config=config)
mcsc.build_from_qiskit(qc, one_way_gates={"CX": ("SNK", "SRC")}, code_bias=True)
num_switches, switch_pos, _, _ = mcsc.compute_min_cut()
```

```{code-cell} ipython3
:tags: [hide-input]
print(f"Total switches required (with one-way CNOT): {num_switches}")
print("Switch locations (qubit, depth):")
for pos in switch_pos:
    print(f" - Qubit {pos[0]} after operation depth {pos[1]}")
```
