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

# Compilation beyond the Surface Code `co3`

This submodule contains an elementary routing routine for CNOT + T compilation on a hexagonal routing graph.
Moreover the routing assumes that each accessible boundary can host both Z and X operators for lattice surgery.
Hence, this code is valid for color codes only. Adaptions are needed to incorporate valid paths for e.g. the folded surface code substrate. Specifications of valid paths for the folded surface code are described in Fig. 8 of [arXiv:2504.10591](https://arxiv.org/pdf/2504.10591). Incorporating this would require to adapt the Dijkstra routine in the VDP solving (see Section V of the paper).

## Layouts

Basic layouts (sparse, pair, row, hex) can be generated automatically in the [HexagonalLattice](co3.utils.lattice_router.HexagonalLattice) class.
However, to place factories in a suitable and controlled manner, it is advisable to construct layouts manually as shown in [scripts/co3/construct_layouts.ipynb](https://github.com/munich-quantum-toolkit/qecc/blob/main/scripts/co3/construct_layouts.ipynb).
A layout describes which nodes on the routing graph are used as logical data qubits and factory locations. The remainder is the routing ancilla space.
The mapping of logical qubit labels onto those chosen data qubit locations on the graph is another task.

## Compilation of given layout and qubit label allocation

The higher level compilation follows a simple greedy routine for solving the VDP problem. We greedily extended this to include paths to factories as well.
Note that the class [ShortestFirstRouterTGatesDyn](co3.utils.lattice_router.ShortestFirstRouterTGatesDyn) and particularly the method [find_total_vdp_layers_dyn](co3.utils.lattice_router.ShortestFirstRouterTGatesDyn.find_total_vdp_layers_dyn) should be used to perform routing as described in the paper.

An example run of the routing routine can be done in the bottom cells of [scripts/co3/construct_layouts.ipynb](https://github.com/munich-quantum-toolkit/qecc/blob/main/scripts/co3/construct_layouts.ipynb), where one can also illustrate the parallel paths in a layer.

## Optimization of qubit label allocation by Hill Climbing

Once chosen a layout, one can optimize the qubit label allocation. This is crucial to exploit parallelism of the original circuit.
The class [HillClimbing](co3.utils.hill_climber.HillClimbing) performs a simple hill climbing routine to optimize the qubit label mapping based on a heuristic metric which computes the initial crossing of shortest paths as well as a more reliable (yet expensive) metric which computes the routing for each Hill climbing iteration and directly aims to reduce the resulting layers.

Benchmark plots can be reproduced from pickle files in [scripts/co3/numerics_summarized.ipynb](https://github.com/munich-quantum-toolkit/qecc/blob/main/scripts/co3/numerics_summarized.ipynb). The benchmarks (modulo random influences from randomly sampled circuits and randomized hill climbing restarts) can be done via [scripts/co3/f_vs_t_q24_row_small.py](https://github.com/munich-quantum-toolkit/qecc/blob/main/scripts/co3/f_vs_t_q24_row_small.py) as well as [scripts/co3/circuit_types.py](https://github.com/munich-quantum-toolkit/qecc/blob/main/scripts/co3/circuit_types.py).

## Microscopic Details

We consider two microscopic substrates, both leading to a hexagonal routing graph.
First, the class [SnakeBuilderSTDW](co3.microscopic.snake_builder.SnakeBuilderSTDW) builds stabilizers and subsets of stabilizers to perform logical meausurements for the color code connected by semi transparent domain walls (STDW).
The class [SnakeBuilderSC](co3.microscopic.snake_builder.SnakeBuilderSC) builds the surface code snakes required to perform lattice surgery between logical folded surface codes. However, this can only display snakes where you can embed the snake in 2d.
A notebook with example constructions can be found in [/scripts/co3/snake_examples.ipynb](https://github.com/munich-quantum-toolkit/qecc/blob/main/scripts/co3/snake_examples.ipynb).

## Macroscopic Compilation Example

To summarize the macroscopic routing, let's go through it step by step for a very small example with a short circuit. First, we create a layout as specified in the `./scripts` and plot it.

```{code-cell} ipython3
import mqt.qecc.co3 as co
from mqt.qecc.co3.utils.lattice_router import plot_lattice_paths

import sys
#some codes are in ./scripts and thus not part of the module. therefore it must be manually imported
sys.path.append("../scripts/co3") #you may have to adapt the path depending on where you run your code

import layouts as layout

#specify factory locations, this requires to know how layout.gen_layout works.
factories = [
    (0, 3),
    (1, 8),
    (2, 13),
    (7, 3),
    (8, 8),
    (9, 13),
    (4, 2),
    (5, 14),
]

g, data_qubit_locs, factory_ring = layout.gen_layout("hex", 24, factories) #factories can be placed on elements in factory_ring.
lat = co.HexagonalLattice(3, 3) #since we overwrite the g anyways, one can initialize lat with mock values.
lat.G = g
size = (5, 5) #size of the plot
lat.plot_lattice(
    size=size, data_qubit_locs=data_qubit_locs, factory_locs=factories
)
```

The plot displays the magic state patches in pink at the boundary of the layout (assuming some magic state factory outside the layout) and logical data patches in orange. Note that there are no edges between adjacent logical data patches as a direct path between them would not be a valid path since it could not host a logical ancilla as required. Note that one can create other custom layouts by hand as well, the resulting structure merely has to ensure that no forbidden paths can happen.

Next, generate some circuit (which is fairly low-depth for illustrative purposes). The list `pairs` contains tuples of two integers for CNOT gates and a single integer represents a T gate which consumes a T state from one of the factory patches.

```{code-cell} ipython3
pairs = [
    (22, 6),
    (11, 2),
    1,
    (14, 3),
    12,
    (18, 19),
    0,
    (7, 20),
    (8, 21),
    9,
    (17, 13),
    23,
    (10, 15),
    5,
    4,
    16
]
```

This circuit is defined logical qubit labels, meaning that the location of those logical qubits is not yet fixed among the possible positions of logical data qubits defined in the layout. Let's define where each logical data qubit is placed on the layout. We just do the most naive layout which is possible.

```{code-cell} ipython3
layout = {}
for i, j in zip(range(len(data_qubit_locs)), data_qubit_locs):
    layout.update({i: (int(j[0]), int(j[1]))})
```

At this point one could also optimize the labeling via Hill Climbing. For such a small circuit it will not be really helpful which is why we only show how one can use the hill climbing in principle here but we do not expect it to be very helpful.

```{code-cell} ipython3
custom_layout = [data_qubit_locs, g]

hc = co.HillClimbing(
    max_restarts=10,
    max_iterations=50,
    circuit=pairs,
    layout_type="custom", #because we want to use our own layout
    m=5, #mock variable for custom layout
    n=5, #mock variable for custom layout
    metric="crossing",
    possible_factory_positions=factories,
    num_factories=len(factories),
    free_rows=None, #only needed for crude basic layouts
    t=1,
    optimize_factories=False,
    custom_layout=custom_layout,
    routing="dynamic", #mock value if crossing metric is used
)

parallel = True
processes = 8 #depending on your hardware
prefix = "../../" #adapt the path depending on where you want to have stored the output
suffix = "test_hc"
best_solution, best_score, best_rep, score_history = hc.run(prefix, suffix, parallel, processes)
print(f"Best solution: {best_solution}")
print(f"Best score: {best_score}")
print(f"To which repetition of the random restarts does the best score belong? {best_rep}")
```

Once one has both the circuit and the dictionary called `layout` defining the labeling, one can translate the circuit `pairs` into the list of gates in terms of locations on the graph.

```{code-cell} ipython3
terminal_pairs = co.translate_layout_circuit(pairs, layout) #let's stick to the simple layout
```

Now one can actually apply the VDP solving:

```{code-cell} ipython3
m, n = 10, 10  # random assignment since g is replaced anyways
router = co.ShortestFirstRouterTGatesDyn(m, n, terminal_pairs, factories, t=2)
router.G = g
vdp_layers = router.find_total_vdp_layers_dyn()
```

Now let's plot all non-empty layers, i.e. all in this example. As before, magic state patches are displayed in pink. Logical data patches are shown in green together with the logical qubit label.

```{code-cell} ipython3
for i, vdp_dict in enumerate(vdp_layers):
    print(f"=====layer = {i}====")
    if len(vdp_dict) != 0:
        plot_lattice_paths(g, vdp_dict, factory_locs=factories, layout=layout, size=size)
```

## Microscopic Snake Example

One can create the stabilizers of the joint codes after the merge between two logical qubits and a snake in between. For instance consider an example for the color code.

```{code-cell} ipython3
import networkx as nx

m = 12
n = 12

g = nx.hexagonal_lattice_graph(m=m, n=n, periodic=False, with_positions=True, create_using=None)

# qubit positions within each patch, must be given in the right order of adjacent patches
positions = [
    [
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (5, 1),
        (5, 2),
        (4, 2),
        (3, 2),
        (2, 2),
        (2, 3),
        (3, 3),
        (4, 3),
        (4, 4),
        (3, 4),
        (2, 4),
        (3, 5),
        (4, 5),
        (3, 6),
        (3, 7),
    ],
    [
        (6, 2),
        (6, 3),
        (6, 4),
        (7, 4),
        (7, 5),
        (6, 5),
        (5, 5),
        (5, 6),
        (6, 6),
        (7, 6),
        (8, 7),
        (7, 7),
        (6, 7),
        (5, 7),
        (4, 8),
        (5, 8),
        (6, 8),
        (7, 8),
        (8, 8),
    ],
    [
        (4, 10),
        (5, 10),
        (6, 10),
        (7, 10),
        (8, 10),
        (8, 11),
        (7, 11),
        (6, 11),
        (5, 11),
        (5, 12),
        (6, 12),
        (7, 12),
        (7, 13),
        (6, 13),
        (5, 13),
        (6, 14),
        (7, 14),
        (6, 15),
        (6, 16),
    ],
    [
        (9, 11),
        (9, 12),
        (9, 13),
        (10, 13),
        (10, 14),
        (9, 14),
        (8, 14),
        (8, 15),
        (9, 15),
        (10, 15),
        (11, 16),
        (10, 16),
        (9, 16),
        (8, 16),
        (7, 17),
        (8, 17),
        (9, 17),
        (10, 17),
        (11, 17),
    ],
]

d = 5 #distance
snake = co.SnakeBuilderSTDW(g, positions, d)

z_plaquettes, x_plaquettes = snake.find_stabilizers()

print("=====X stabilizers if ZZ merge=====")
snake.plot_stabilizers(x_plaquettes)
print("=====Z stabilizers if ZZ merge=====")
snake.plot_stabilizers(z_plaquettes)
```

Furthermore one can specify the subset of stabilizers to be measured to retrieve the logical ZZ result (Fig. 7 in the paper).

```{code-cell} ipython3
# consider the boundary patches to be logical and find the subset of stabilizers to measure the logical ZZ between them
subset_stabs = snake.find_stabilizers_zz()
assert snake.test_zz_stabs(subset_stabs) is True

snake.plot_stabilizers(subset_stabs)
```
