import stim
import random


def extract_gates_schedule(schedule):
    """extracts the gates from the schedule in the order of the schedule. ignore 'idle' moves in the vdp dict"""
    gates = []
    for i in range(len(schedule)):
        gates_temp = list(schedule[i]["vdp_dict"].keys())
        gates_temp = [el for el in gates_temp if not isinstance(el, str)]
        gates += gates_temp
    return gates


def extract_gates_schedule_respect_layout(schedule):
    """extracts the gates with the qubit labels (not the positions of quibts) by respecting the changing `layout` dictionaries"""
    gates = []
    for i in range(len(schedule)):
        gates_temp = list(schedule[i]["vdp_dict"].keys())
        gates_temp = [el for el in gates_temp if not isinstance(el, str)]
        layout = schedule[i]["layout"]
        layout_rev = {j: i for i, j in layout.items()}
        # gates_temp = [(layout_rev[el[0]], layout_rev[el[1]]) for el in gates_temp]
        gates_temp_temp = []
        # translate the gates into number labels, not pos on graph
        for el in gates_temp:
            if isinstance(el[0], tuple):
                # cnot
                gates_temp_temp.append((layout_rev[el[0]], layout_rev[el[1]]))
            elif isinstance(el[0], int):
                gates_temp_temp.append(layout_rev[el])
        gates += gates_temp_temp
    return gates


def check_num_gates(terminal_pairs, schedule) -> bool:
    """check that the input `terminal_pairs` has the same number of gates as the resulting schedule."""
    gates_schedule = extract_gates_schedule(schedule)
    if len(gates_schedule) == len(terminal_pairs):
        return True
    else:
        return False


def random_initial_state(n_qubits: int) -> stim.Circuit:
    """creates some random circuit in stim for a random initial state."""
    c = stim.Circuit()
    for q in range(n_qubits):
        # choose a random Pauli: I, X, Y, Z
        p = random.choice(["I", "X", "Y", "Z"])
        if p != "I":
            c.append(p, [q])
    return c


def check_order_dyn_gates_st(terminal_pairs, vdp_layers, layout=None) -> bool:
    """only works for cnot circuits and the standard scheme, i.e. quilt.find_total_vdp_layers_dyn"""

    flattened_pairs = [item for pair in terminal_pairs for item in pair]
    data_qubit_locs = list(set(flattened_pairs))
    n_qubits = len(data_qubit_locs)
    if layout is None:
        layout = {
            i: j for i, j in enumerate(data_qubit_locs)
        }  # random layout, details do not matter here

    initial_state = random_initial_state(n_qubits)

    layout_rev = {j: i for i, j in layout.items()}
    # terminal_pairs_trans = [(layout_rev[el[0]], layout_rev[el[1]]) for el in terminal_pairs]
    terminal_pairs_trans = []
    for el in terminal_pairs:
        # print("el", el)
        if isinstance(el[0], tuple):
            # cnot
            terminal_pairs_trans.append((layout_rev[el[0]], layout_rev[el[1]]))
        elif isinstance(el[0], int):
            terminal_pairs_trans.append(layout_rev[el])
        # print("terminal pairs trans", terminal_pairs_trans)
    gates_schedule = []
    for vdp_dict in vdp_layers:
        gates_schedule += [
            (
                (layout_rev[el[0]], layout_rev[el[1]])
                if isinstance(el[0], tuple)
                else layout_rev[el]
            )
            for el in list(vdp_dict.keys())
        ]
    if not set(terminal_pairs_trans) == set(gates_schedule):
        print("terminal_pairs_trans", len(terminal_pairs_trans), terminal_pairs_trans)
        print("gates_schedule", len(gates_schedule), gates_schedule)
        print(
            "in terminal_pairs_trans but not in gates_schedule",
            set(terminal_pairs_trans) - set(gates_schedule),
        )
        print("vice versa", set(gates_schedule) - set(terminal_pairs_trans))
        if len(set(terminal_pairs_trans) - set(gates_schedule)) != 0:
            idx_lst = [
                terminal_pairs_trans.index(el)
                for el in list(set(terminal_pairs_trans) - set(gates_schedule))
            ]
            missing_gates = [terminal_pairs[idx] for idx in idx_lst]
            print("missing gates geometry:", missing_gates)

        raise ValueError(
            "Something is wrong with the schedule since the gates of terminal pairs and the schedule do not coincide (order irrelevant here)"
        )

    initial_circ_order = initial_state.copy()
    for el in terminal_pairs_trans:
        if isinstance(el, tuple):
            # print("tuple el", el)
            initial_circ_order.append("CNOT", [el[0], el[1]])
        elif isinstance(el, int):
            # print("int el", el)
            initial_circ_order.append("s", el)

    dyn_circ_order = initial_state.copy()
    for el in gates_schedule:
        if isinstance(el, tuple):
            dyn_circ_order.append("CNOT", [el[0], el[1]])
        elif isinstance(el, int):
            dyn_circ_order.append("s", el)

    sim1 = stim.TableauSimulator()
    sim1.do_circuit(initial_circ_order)
    tableau1 = sim1.current_inverse_tableau()

    sim2 = stim.TableauSimulator()
    sim2.do_circuit(dyn_circ_order)
    tableau2 = sim2.current_inverse_tableau()

    return tableau1 == tableau2


def check_order_dyn_gates(terminal_pairs, schedule) -> bool:
    """
    !only works for cnot circuits, for circs with t does not work
    simulates the gate order in the schedule (changed from pushing) and from initial terminal pairs on a random initial state.
    the results must coincide to make sure that the pushing is performed correctly.
    """
    flattened_pairs = [item for pair in terminal_pairs for item in pair]
    data_qubit_locs = list(set(flattened_pairs))
    n_qubits = len(data_qubit_locs)

    gates_schedule = extract_gates_schedule_respect_layout(schedule)
    initial_state = random_initial_state(n_qubits)

    layout = schedule[0]["layout"]
    layout_rev = {j: i for i, j in layout.items()}
    # terminal_pairs_trans = [(layout_rev[el[0]], layout_rev[el[1]]) for el in terminal_pairs]
    terminal_pairs_trans = []
    for el in terminal_pairs:
        if isinstance(el[0], tuple):
            # cnot
            terminal_pairs_trans.append((layout_rev[el[0]], layout_rev[el[1]]))
        elif isinstance(el[0], int):
            terminal_pairs_trans.append(layout_rev[el])

    if not set(terminal_pairs_trans) == set(gates_schedule):
        print(
            "terminal_pairs_trans",
            len(terminal_pairs_trans),
            "gates_schedule",
            len(gates_schedule),
        )
        print(
            "in terminal_pairs_trans but not in gates_schedule",
            set(terminal_pairs_trans) - set(gates_schedule),
        )
        print("vice versa", set(gates_schedule) - set(terminal_pairs_trans))
        if len(set(terminal_pairs_trans) - set(gates_schedule)) != 0:
            idx_lst = [
                terminal_pairs_trans.index(el)
                for el in list(set(terminal_pairs_trans) - set(gates_schedule))
            ]
            missing_gates = [terminal_pairs[idx] for idx in idx_lst]
            print("missing gates geometry:", missing_gates)

        raise ValueError(
            "Something is wrong with the schedule since the gates of terminal pairs and the schedule do not coincide (order irrelevant here)"
        )

    initial_circ_order = initial_state.copy()
    for el in terminal_pairs_trans:
        if isinstance(el, tuple):
            initial_circ_order.append("CNOT", [el[0], el[1]])
        elif isinstance(el, int):
            initial_circ_order.append("s", el)

    dyn_circ_order = initial_state.copy()
    for el in gates_schedule:
        if isinstance(el, tuple):
            dyn_circ_order.append("CNOT", [el[0], el[1]])
        elif isinstance(el, int):
            dyn_circ_order.append("s", el)

    sim1 = stim.TableauSimulator()
    sim1.do_circuit(initial_circ_order)
    tableau1 = sim1.current_inverse_tableau()

    sim2 = stim.TableauSimulator()
    sim2.do_circuit(dyn_circ_order)
    tableau2 = sim2.current_inverse_tableau()

    return tableau1 == tableau2


def check_duplicate_nodes_per_layer_st(vdp_layers):
    """
    checks whether there are duplicate nodes in the paths of one layer.
    this would indicate that there are overlapping paths and is a big problem.
    """
    for i, vdp_dict in enumerate(vdp_layers):
        # collect all used nodes in a list
        all_nodes = []
        for path in vdp_dict.values():
            all_nodes += path
        # check whether there are duplicate items, then problem!!
        seen = set()
        duplicates = [x for x in all_nodes if x in seen or seen.add(x)]

        if len(duplicates) != 0:
            raise ValueError(
                f"There are duplicates in layer {i} !!! The duplicate elements are {duplicates}"
            )
    # if no error was raised print that all is good
    return True


def check_duplicate_nodes_per_layer(schedule):
    """
    checks whether there are duplicate nodes in the paths of one layer.
    this would indicate that there are overlapping paths and is a big problem.
    """
    for i, layer in enumerate(schedule):
        vdp_dict = layer["vdp_dict"]
        steiner = layer["steiner"]
        # collect all used nodes in a list
        all_nodes = []
        for path in vdp_dict.values():
            all_nodes += path
        if steiner is not None:
            for path in steiner.values():
                # do not include path[0] because this is already in vdp_dict
                all_nodes += path[1][
                    1:
                ]  # the very first item is on the path, i.e. it would be duplicate by construction, this is not what we want to catch here
        # check whether there are duplicate items, then problem!!

        seen = set()
        duplicates = [x for x in all_nodes if x in seen or seen.add(x)]

        if len(duplicates) != 0:
            raise ValueError(
                f"There are duplicates in layer {i} !!! The duplicate elements are {duplicates}"
            )
    # if no error was raised print that all is good
    return True


def check_path_on_logical_st(vdp_layers, logical_pos):
    """checks whether the path occupies any logical pos somewhere else than on the end points. this would be an issue!"""
    for i, vdp_dict in enumerate(vdp_layers):
        for t_p, path in vdp_dict.items():
            if any(node in logical_pos for node in path[1:-1]):
                raise ValueError(
                    "There are paths placed on logical pos which cannot be!"
                )
            else:
                pass
    return True


def check_path_on_logical(schedule):
    """checks whether a path / tree occupies any logical pos somewhere else than at the end points."""
    for i, layer in enumerate(schedule):
        vdp_dict = layer["vdp_dict"]
        steiner = layer["steiner"]
        logical_pos = list(layer["layout"].values())
        for path in vdp_dict.values():
            if any(node in logical_pos for node in path[1:-1]):
                raise ValueError(
                    f"In layer {i} there is a path placed on a logical pos :("
                )
        if steiner is not None:
            for tree in steiner.values():
                if any(node in logical_pos for node in tree[0][1:-1]):
                    raise ValueError(
                        f"In layer {i} there is a path (of a tree) placed on a logical pos :("
                    )
                if any(node in logical_pos for node in tree[1]):
                    raise ValueError(
                        f"In layer {i} there is some tree extension placed on a logical pos :("
                    )
    return True


def test_times_t_gates_opt(schedule, t, factories):
    """
    Checks whether the t timestamps make sense in a finished schedule (opt).

    Only the vdp_dict per layer is checked here, the trees are not considered.
    """
    factory_times = {f: t for f in factories}
    for layer in schedule:
        vdp_dict = layer["vdp_dict"]
        # filter out all t factories used in this layer
        for t_gate, path in vdp_dict.items():
            if isinstance(t_gate[0], int):  # only then we have a tgate
                if path[0] in factories:
                    factory = path[0]
                elif path[-1] in factories:
                    factory = path[-1]
                else:
                    raise ValueError(
                        "factories to not add up with the factories in the schedule."
                    )
                if factory_times[factory] > 0:
                    raise ValueError(
                        "Factories are used at forbidden times! (or your input for the test function was wrong)"
                    )
                else:
                    factory_times[factory] = (
                        t + 1
                    )  # s.t. we can directly remove -1 again without case distinction
        factory_times = {f: el - 1 for f, el in factory_times.items()}
    return True


def test_times_t_gates_st(vdp_layers, t, factories):
    """checks whether the t timestamps make sense in a finisehd vdp layeyrs (st)"""
    factory_times = {f: t for f in factories}
    for vdp_dict in vdp_layers:
        for t_gate, path in vdp_dict.items():
            if isinstance(t_gate[0], int):  # only then we have a tgate
                if path[0] in factories:
                    factory = path[0]
                elif path[-1] in factories:
                    factory = path[-1]
                else:
                    raise ValueError(
                        "factories to not add up with the factories in the schedule."
                    )
                if factory_times[factory] > 0:
                    raise ValueError(
                        "Factories are used at forbidden times! (or your input for the test function was wrong)"
                    )
                else:
                    factory_times[factory] = (
                        t + 1
                    )  # s.t. we can directly remove -1 again without case distinction
        factory_times = {f: el - 1 for f, el in factory_times.items()}
    return True
