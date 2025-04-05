import numpy as np
from lib_memory import get_bell_state, get_qubit_state, init_qubit_state

def build_cnot_error_propagation():
    # Define the Pauli multiplication table (modulo global phase)
    pauli_mul = {
        ('C', 'C'): 'C',
        ('C', 'X'): 'X',
        ('C', 'Y'): 'Y',
        ('C', 'Z'): 'Z',
        ('X', 'C'): 'X',
        ('X', 'X'): 'C',
        ('X', 'Y'): 'Z',
        ('X', 'Z'): 'Y',
        ('Y', 'C'): 'Y',
        ('Y', 'X'): 'Z',
        ('Y', 'Y'): 'C',
        ('Y', 'Z'): 'X',
        ('Z', 'C'): 'Z',
        ('Z', 'X'): 'Y',
        ('Z', 'Y'): 'X',
        ('Z', 'Z'): 'C',
    }

    # Function to propagate errors through the CNOT gate
    def propagate_error(E_c, E_t):
        # Initialize output errors
        E_c_out, E_t_out = E_c, E_t

        # Propagate errors based on the CNOT error propagation rules
        # If the control qubit has an X or Y error, it affects the target qubit
        if E_c in ['X', 'Y']:
            E_t_out = pauli_mul[(E_t, 'X')]

        # If the target qubit has a Z or Y error, it affects the control qubit
        if E_t in ['Z', 'Y']:
            E_c_out = pauli_mul[(E_c, 'Z')]

        return (E_c_out, E_t_out)

    # Generate the error mapping for all combinations
    paulis = ['C', 'X', 'Y', 'Z']
    error_propagation = {}
    for E_c in paulis:
        for E_t in paulis:
            input_error = (E_c, E_t)
            output_error = propagate_error(E_c, E_t)
            error_propagation[input_error] = output_error
    return error_propagation

error_propagation_table = build_cnot_error_propagation()

# parity checks
Z = ['Phi+', 'Phi-']    # tells Phi from Psi
X = ['Phi+', 'Psi+']    # tells + from -

# Error propagation through CNOT gate
def apply_cnot(control_error, target_error):
    if control_error in ['E', 'R', 'M'] or target_error in ['E', 'R', 'M']:
        print("Error: Should not get here!")
        return control_error, target_error
    else:
        key = (control_error, target_error)
        (E_control_after, E_target_after) = error_propagation_table.get(key)
        return E_control_after, E_target_after

def purify_one(control_pair, target_pair, measure='TARGET', parity=Z):
    """Perform CNOT and measure."""
    qubit_A1, qubit_B1 = control_pair
    qubit_A2, qubit_B2 = target_pair

    A1_state = get_qubit_state(qubit_A1['m_state'])
    A2_state = get_qubit_state(qubit_A2['m_state'])

    B1_state = get_qubit_state(qubit_B1['m_state'])
    B2_state = get_qubit_state(qubit_B2['m_state'])

    A1_error_after, A2_error_after = apply_cnot(A1_state, A2_state)
    B1_error_after, B2_error_after = apply_cnot(B1_state, B2_state)

    res_parity = None
    if measure == 'TARGET':
        bell_out = get_bell_state(A2_error_after, B2_error_after)
        res_parity = (bell_out in parity)
    elif measure == 'CONTROL':
        bell_out = get_bell_state(A1_error_after, B1_error_after)
        res_parity = (bell_out in parity)
    elif measure == 'DOUBLE':
        bell_out1 = get_bell_state(A1_error_after, B1_error_after)
        bell_out2 = get_bell_state(A2_error_after, B2_error_after)
        res_parity = (bell_out1 in parity[0]) and (bell_out2 in parity[1])
    else:
        return None

    qubit_A1['m_state'] = init_qubit_state(A1_error_after)
    qubit_A2['m_state'] = init_qubit_state(A2_error_after)
    qubit_B1['m_state'] = init_qubit_state(B1_error_after)
    qubit_B2['m_state'] = init_qubit_state(B2_error_after)

    control_after = (qubit_A1, qubit_B1)
    target_after = (qubit_A2, qubit_B2)

    return res_parity, control_after, target_after


def circuit_ssspX(pairs):    # single selection - single error X
    """Purify X error on control pair (uses Z measurement on target)."""
    parityZ, control_after, _ = purify_one(pairs[0], pairs[1], 'TARGET', Z)
    if parityZ:
        return control_after
    return None

def circuit_ssspZ(pairs):    # single selection - single error Z
    """Purify Z error on target pair (uses X measurement on control)."""
    parity, _, target_after = purify_one(pairs[1], pairs[0], 'CONTROL', X)    # reverse order to preserve 1st pair
    if parity:
        return target_after
    return None


def circuit_dsspXZ(pairs):    # double selection single error X
    """Purify X error and verify Z propagation."""
    _, pair0_after1, pair1_after1 = purify_one(pairs[0], pairs[1])   # no measurement
    parity, _, _ = purify_one(pairs[2], pair1_after1, 'DOUBLE', (Z,X))     # purify X and verify Z on measured pair (pair1_after1)
    if parity:
        return pair0_after1
    return None

def circuit_ssdpXZ(pairs):    # single selection double error X and Z
    """Purify X and Z error."""
    parity1, pair0_after1, _ = purify_one(pairs[0], pairs[1], 'TARGET', Z)   # purify X
    parity2, _, pair0_after2 = purify_one(pairs[2], pair0_after1, 'CONTROL', X)    # purify Z
    if parity1 and parity2:
        return pair0_after2
    return None

def circuit_dsdpXZ(pairs):    # double selection double error X and Z
    """Purify X and Z error and verify Z and X propagation."""
    _, pair0_after1, pair1_after1 = purify_one(pairs[0], pairs[1])    # no measurement
    parity2and3, pair2_after2, pair1_after2 = purify_one(pairs[2], pair1_after1, 'DOUBLE', (Z,X))
    _, pair3_after3, pair0_after3 = purify_one(pairs[3], pair0_after1)  # no measurement
    parity4and5, _, _ = purify_one(pair3_after3, pairs[4], 'DOUBLE', (X,Z))
    if parity2and3 and parity4and5:
        return pair0_after3
    return None

circuit_map = {
    'SsSpX': (circuit_ssspX, 2),
    'SsSpZ': (circuit_ssspZ, 2),
    'DsSp': (circuit_dsspXZ, 3),
    'SsDp': (circuit_ssdpXZ, 3),
    'DsDp': (circuit_dsdpXZ, 5)
}

def calculate_total_pairs_needed(circuit_string):
    """
    Calculate the total number of pairs needed based on a purification schedule string.

    Parameters:
        circuit_string (str): The string defining the purification schedule.

    Returns:
        int: Total number of pairs needed.
    """
    total_pairs_needed = 0
    remaining_pairs = 0  # Pairs carried over between rounds
    
    if circuit_string in circuit_map:
        _, input_pairs = circuit_map[circuit_string]
        return input_pairs

    # Split the schedule into rounds
    rounds = circuit_string.split('+')

    for round_index, round_def in enumerate(rounds):
        round_pairs_needed = 0
        parallel_circuits = round_def.split('|')

        for circuit in parallel_circuits:
            if circuit not in ['SsSpX', 'SsSpZ']:
                raise ValueError(f"Circuit not supported for recurrence: {circuit}")

            _, input_pairs = circuit_map[circuit]
            round_pairs_needed += input_pairs

        # Calculate the total pairs needed for this round, considering carried-over pairs
        total_pairs_needed += max(0, round_pairs_needed - remaining_pairs)

        # Update remaining pairs for the next round
        used_pairs = max(round_pairs_needed, remaining_pairs)
        remaining_pairs = sum(1 for circuit in parallel_circuits)

    return total_pairs_needed

def do_purification(circuit_string, pairs):
    """
    Parse the circuit string and perform the specified circuits.

    Args:
        circuit_string (str): The string defining the sequence of circuits.
        pairs (list): The list of provided pairs.

    Returns:
        The last purified pair
    """

    qubit_A1, qubit_B1 = pairs[0]
    id_first_pair = (qubit_A1['id'], qubit_B1['id'])

    # Perform the circuits
    rounds = circuit_string.split('+')
    prev_round_pairs = pairs

    for round_str in rounds:
        used_pairs = 0
        purified_pairs = []
        parallel_circuits = round_str.split('|')
        for circuit in parallel_circuits:
            circuit_func, num_pairs = circuit_map[circuit]
            result_pair = circuit_func(prev_round_pairs[used_pairs:used_pairs+num_pairs])
            if result_pair is not None:
                purified_pairs.append(result_pair)
            else:
                return None
            used_pairs += num_pairs
        purified_pairs.extend(prev_round_pairs[used_pairs:])
        prev_round_pairs = purified_pairs

    if len(prev_round_pairs) == 1:
        # update qubits
        qubit_A1, qubit_B1 = prev_round_pairs[0]
        qubit_A1['purified'] = True
        qubit_B1['purified'] = True

        id_last_pair = (qubit_A1['id'], qubit_B1['id'])
        if id_first_pair != id_last_pair:
            raise ValueError("Purified pair is not the first pair.")

        return (qubit_A1, qubit_B1)
    else:
        raise ValueError("Expected one final pair.")
        return None