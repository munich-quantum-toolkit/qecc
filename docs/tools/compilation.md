# Logical Compilation

Tools for compiling quantum circuits at the logical level, mapping logical
operations onto the primitives an error-corrected architecture provides.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Code Switching Optimization
:link: ../CodeSwitching
:link-type: doc

Optimized switching between codes whose transversal gate sets complement each
other, recovering universality without magic-state distillation.
:::

:::{grid-item-card} `cococo` Color Code Compilation
:link: ../cococo
:link-type: doc

Lattice-surgery routing for color codes, including compilation that exploits
movable logical qubits on a hexagonal routing graph.
:::

::::

```{toctree}
:hidden:
:maxdepth: 1

../CodeSwitching
../cococo
```
