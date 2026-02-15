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

# `cococo` color code compilation

This submodule contains routing routines as described in in the papers (1) `Lattice Surgery Compilation Beyond the Surface Code` ([arXiv:2504.10591](https://arxiv.org/pdf/2504.10591)) as well as (2) `Exploiting Movable Logical Qubits for Lattice Surgery Compilation` ([arXiv:2512.04169](https://arxiv.org/pdf/2512.04169)).

The elementary routing routines for CNOT + T compilation in (1) assume a hexagonal routing graph and accessible Z and X operators along each boundary of a patch for lattice surgery. Hence, it is tailored to the color code, but could in principle be adapted for e.g. the folded surface code substrate as described in Fig. 8 of (1), by adapting the Dijkstra routine.

The routing routines for movable qubits (2) are tailored for the color code as well and would require more fundamental changes to allow other codes.

## Layouts

Different layouts ("row", "pair", "hex", "triple", "single") can be generated with the function [gen_layout_scalable](mqt.qecc.cococo.layouts.gen_layout_scalable). One can place factory patches along the boundary. The construction of such layouts is described in the notebook [scripts/cococo/layouts_general.ipynb](https://github.com/munich-quantum-toolkit/qecc/tree/main/scripts/cococo/layouts_general.ipynb).

A layout describes which nodes on the routing graph are used as logical data qubits and factory locations. The remainder is the routing ancilla space.
The mapping of logical qubit labels onto those chosen data qubit locations on the graph is another task.

## Randomly Sampled CNOT + T circuits

This submodule considers CNOT + T circuits without single qubit Clifford gates. Different types of random circuits can be generated using the functions [generate_random_circuit](mqt.qecc.cococo.circuit_construction.generate_random_circuit), [generate_max_parallel_circuit](mqt.qecc.cococo.circuit_construction.generate_max_parallel_circuit), [generate_min_parallel_circuit](mqt.qecc.cococo.circuit_construction.generate_min_parallel_circuit) as well as [create_random_sequential_circuit_dag](mqt.qecc.cococo.circuit_construction.create_random_sequential_circuit_dag). However, one is welcome to write own circuit constructions.

## Basic Routing and Qubit Label Allocation from (1)

### Basic Compilation with given Layout and Mapping

The higher level compilation follows a simple greedy routine for solving the VDP problem. We greedily extended this to include paths to factories as well.
Note that the class [BasicRouter](mqt.qecc.cococo.utils_routing.BasicRouter)
and particularly the method `find_total_vdp_layers_dyn` should be used to perform routing as described in the paper.

### Optimization of Qubit Label Allocation by Hill Climbing

Once chosen a layout, one can optimize the qubit label allocation. This is important to exploit more parallelism of the original circuit.
The class [HillClimbing](mqt.qecc.cococo.hill_climber.HillClimbing) performs a simple hill climbing routine to optimize the qubit label mapping based on a heuristic metric which computes the initial crossing of shortest paths as well as a more reliable (yet expensive) metric which computes the routing for each Hill climbing iteration and directly aims to reduce the resulting layers. How it works can be seen in the notebook [scripts/cococo/hill_climbing_examples.ipynb](https://github.com/munich-quantum-toolkit/qecc/tree/main/scripts/cococo/hill_climbing_examples.ipynb).

Plots shown in (1) can be reproduced from pickle files in [scripts/cococo/evaluations_beyond_the_surface_code](https://github.com/munich-quantum-toolkit/qecc/tree/main/scripts/cococo/evaluations_beyond_the_surface_code).

### Microscopic Details of Snakes

In (1), we consider two microscopic substrates, both leading to a hexagonal routing graph.
First, the class [SnakeBuilderSTDW](mqt.qecc.cococo.snake_builder.SnakeBuilderSTDW) builds stabilizers and subsets of stabilizers to perform logical measurements for the color code connected by semi transparent domain walls (STDW).
The class [SnakeBuilderSC](mqt.qecc.cococo.snake_builder.SnakeBuilderSC) builds the surface code snakes required to perform lattice surgery between logical folded surface codes. However, this can only display snakes where you can easily embed the snake in 2d.
A notebook with example constructions can be found in [/scripts/cococo/snake_examples.ipynb](https://github.com/munich-quantum-toolkit/qecc/tree/main/scripts/cococo/snake_examples.ipynb).

## Compilation with Movable Logical Qubits (2)

Compilation with movable logical qubits as described in (2) builds upon the BasicRouter from above. Based on the basic router we constructed a lookahead routine with simulated annealing which can be used via the [TeleportationRouter](mqt.qecc.cococo.utils_routing.TeleportationRouter). Examples are shown in the notebook [scripts/cococo/movable_qubit_router_examples.ipynb](https://github.com/munich-quantum-toolkit/qecc/tree/main/scripts/cococo/movable_qubit_router_examples.ipynb).

Results shown in (2) can be reproduced in [scripts/cococo/evaluations_movable_qubits](https://github.com/munich-quantum-toolkit/qecc/tree/main/scripts/cococo/evaluations_movable_qubits)

## Selected Examples

### Microscopic Color Code Snake

One can create the stabilizers of the joint codes after the merge between two logical qubits and a snake in between. For instance consider an example for the color code.

```{code-cell} ipython3
import networkx as nx
from mqt.qecc.cococo import snake_builder

m = 12
n = 12

g = nx.hexagonal_lattice_graph(m=m, n=n, periodic=False, with_positions=True, create_using=None)

# qubit positions within each patch, must be given in the right order of adjacent patches
positions = [
    [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (5, 2), (4, 2), (3, 2), (2, 2), (2, 3), (3, 3), (4, 3), (4, 4), (3, 4), (2, 4), (3, 5), (4, 5), (3, 6), (3, 7),],
    [(6, 2), (6, 3), (6, 4), (7, 4), (7, 5), (6, 5), (5, 5), (5, 6), (6, 6), (7, 6), (8, 7), (7, 7), (6, 7), (5, 7), (4, 8), (5, 8), (6, 8), (7, 8), (8, 8),],
    [(4, 10), (5, 10), (6, 10), (7, 10), (8, 10), (8, 11), (7, 11), (6, 11), (5, 11), (5, 12), (6, 12), (7, 12), (7, 13), (6, 13), (5, 13), (6, 14), (7, 14), (6, 15), (6, 16),],
    [(9, 11), (9, 12), (9, 13), (10, 13), (10, 14), (9, 14), (8, 14), (8, 15), (9, 15), (10, 15), (11, 16), (10, 16), (9, 16), (8, 16), (7, 17), (8, 17), (9, 17), (10, 17), (11, 17),],
]

d = 5 #distance
snake = snake_builder.SnakeBuilderSTDW(g, positions, d)

z_plaquettes, x_plaquettes = snake.find_stabilizers()

print("=====X stabilizers if ZZ merge=====")
snake.plot_stabilizers(x_plaquettes)
print("=====Z stabilizers if ZZ merge=====")
snake.plot_stabilizers(z_plaquettes)
```

Furthermore one can specify the subset of stabilizers to be measured to retrieve the logical ZZ result (Fig. 7 in the paper (1)).

```{code-cell} ipython3
# consider the boundary patches to be logical and find the subset of stabilizers to measure the logical ZZ between them
subset_stabs = snake.find_stabilizers_zz()
assert snake.test_zz_stabs(subset_stabs) is True

snake.plot_stabilizers(subset_stabs)
```

### Compilation with Movable Qubits

To run the compilation with movable qubits, construct a layout first.

```{code-cell} ipython3
:tags: [hide-input]

import sys
sys.path.append("../scripts/cococo") #since plotting is part of scripts and not part of the submodule it must be included separately.
```

```{code-cell} ipython3
import plotting

import mqt.qecc.cococo.utils_routing as utils
from mqt.qecc.cococo import circuit_construction, layouts

layout_type = "triple"
m = 2
n = 2
factories = []
remove_edges = False
g, data_qubit_locs, factory_ring = layouts.gen_layout_scalable(layout_type, m, n, factories, remove_edges)
layout = dict(enumerate(data_qubit_locs)) #standard mapping

plotting.plot_lattice_paths(g, {}, {}, layout, factories, size=(12, 4))
```

Then, choose a circuit.

```{code-cell} ipython3
q = len(data_qubit_locs)
j = 8 #the number of gates per logical layer
num_gates = q * 2
dag, pairs = circuit_construction.create_random_sequential_circuit_dag(
    j,
    q,
    num_gates,
)

#randomly chosen circuit:
pairs = [(8, 13), (15, 2), (9, 10), (0, 3), (23, 20), (1, 19), (4, 6), (22, 5), (13, 20), (2, 1), (10, 6), (0, 3), (23, 15), (9, 19), (5, 4), (8, 22), (3, 20), (1, 19), (6, 10), (4, 0), (2, 15), (22, 9), (13, 5), (8, 23), (3, 1), (8, 19)]
```

Then, one can run the basic router first to receive a reference result.

```{code-cell} ipython3
terminal_pairs = layouts.translate_layout_circuit(pairs, layout)  # let's stick to the simple layout
t=4 #mock reset time for pure cnot circuit
router = utils.BasicRouter(g, data_qubit_locs, factories, valid_path="cc", t=t, metric="exact", use_dag=True)
layers = router.split_layer_terminal_pairs(terminal_pairs)
vdp_layers, _ = router.find_total_vdp_layers_dyn(layers, data_qubit_locs, router.factory_times, layout, testing=False) #usually recommended to use `testing=True`
print("Len of schedule without teleportation: ", len(vdp_layers))
```

Afterward, let's run the router with movable qubits, i.e. logical qubits can be moved during the execution of a CNOT gate as described in paper (2). A couple of parameters need to be defined, which are described in detail in [optimize_layers](mqt.qecc.cococo.utils_routing.TeleportationRouter.optimize_layers).

```{code-cell} ipython3
router = utils.TeleportationRouter(
    g, data_qubit_locs, factories, valid_path="cc", t=t, metric="exact", use_dag=True, seed=1
)
layers = router.split_layer_terminal_pairs(terminal_pairs)

max_iters = 100
T_start = 100.0
T_end = 0.1
alpha = 0.95
radius = 10
k_lookahead = 5
steiner_init_type = "full_random"
jump_harvesting = True
reduce_steiner = True
idle_move_type = "later"
reduce_init_steiner = False
stimtest = True

schedule, _ = router.optimize_layers(
    terminal_pairs,
    layout,
    max_iters,
    T_start,
    T_end,
    alpha,
    radius=radius,
    k_lookahead=k_lookahead,
    steiner_init_type=steiner_init_type,
    jump_harvesting=jump_harvesting,
    reduce_steiner=reduce_steiner,
    idle_move_type=idle_move_type,
    reduce_init_steiner=reduce_init_steiner,
    stimtest=stimtest,
)

print("Len of schedule with teleport router: ", len(schedule))
```

Overall the router with movable qubits reduces the schedule depth.

```{code-cell} ipython3
print("Reduction Delta: ", len(vdp_layers) - len(schedule))
```

Let's plot the first few layers of the routing explicitly:

```{code-cell} ipython3
plotting.plot_schedule(g, schedule[:3], factories, size = (12,4))
```

In the first layer one can see that a tree for moving a data qubit was found.

A larger example, with larger absolute improvement, can be found in the notebook mentioned above: [scripts/cococo/movable_qubit_router_examples.ipynb](https://github.com/munich-quantum-toolkit/qecc/tree/main/scripts/cococo/movable_qubit_router_examples.ipynb).
