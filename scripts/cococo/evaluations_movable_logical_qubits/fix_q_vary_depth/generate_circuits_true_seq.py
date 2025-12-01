import cococo.circuit_construction as circuit_construction
import cococo.dag_helper as dag_helper
import json

q = 120
j = 8
d = 320
num_gates = j * d
reps = 20

circuits = []
for _ in range(reps):
    dag, pairs = circuit_construction.create_random_sequential_circuit_dag(
        j, q, num_gates
    )
    circuits.append(pairs)

    # check whether the number of layers you want
    layers = dag_helper.count_cx_gates_per_layer(dag)
    print(layers)
    print(len(layers))


path = f"true_seq_circs_j{j}_q{q}_numgates{num_gates}d{d}_x{reps}.json"

with open(path, "w") as f:
    json.dump(circuits, f)
