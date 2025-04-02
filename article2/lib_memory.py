import numpy as np
from lib_dtmc import MEMORY_STATES

# Mapping from 2 qubit states to Bell state
def get_bell_state(state_A, state_B):
    combined_error = (state_A, state_B)
    error_to_bell_state = {
        ('C', 'C'): 'Phi+',
        ('C', 'Z'): 'Phi-',
        ('Z', 'C'): 'Phi-',
        ('Z', 'Z'): 'Phi+',
        ('C', 'X'): 'Psi+',
        ('X', 'C'): 'Psi+',
        ('X', 'X'): 'Phi+',
        ('X', 'Z'): 'Psi-',
        ('Z', 'X'): 'Psi-',
        ('C', 'Y'): 'Psi-',
        ('Y', 'C'): 'Psi-',
        ('Y', 'Y'): 'Phi+',
        ('X', 'Y'): 'Phi-',
        ('Y', 'X'): 'Phi-',
        ('Z', 'Y'): 'Psi+',
        ('Y', 'Z'): 'Psi+',
    }
    bell_state = error_to_bell_state.get(combined_error, 'Failure')
    return bell_state


class Memory:
    def __init__(self, node_name, num_qubits, tm, t_cutoff):
        """Initialize the Memory object with node name and n qubits."""
        self._node_name = node_name
        self._tm = tm
        self._t_cutoff = t_cutoff
        self._qubits = [{ 'id': f'{node_name}_Qubit_{i}', 'm_state': init_qubit_state('C'), 'age': 0, 'etg': None, 'purified': False } for i in range(num_qubits)]

    @property
    def qubits(self):
        """Get the qubits in memory."""
        return self._qubits

    def update_memory(self):
        for qubit in self._qubits:
            if qubit['etg'] is not None:
                if (qubit['age'] + 1) >= self._t_cutoff:
                    # print(f"Cutoff qubit {qubit['id']}")
                    qubit['etg'] = None
                    qubit['age'] = 0
                    qubit['m_state'] = init_qubit_state('C')
                    qubit['purified'] = False
                else:
                    new_state, state_index = self.apply_memory_transition(qubit['m_state'])
                    if state_index in [MEMORY_STATES['R'], MEMORY_STATES['E'], MEMORY_STATES['M']]:
                        # print(f"Decohered qubit {qubit['id']}")
                        qubit['etg'] = None
                        qubit['age'] = 0
                        qubit['m_state'] = init_qubit_state('C')
                        qubit['purified'] = False
                    else:
                        qubit['m_state'] = new_state
                        qubit['age']+=1

    def get_next_available_qubit(self):
        for index, qubit in enumerate(self._qubits):
            if qubit['etg'] is None:
                # print(f"Free qubit: {qubit['id']}")
                return qubit.copy(), index
        return None, -1

    def replace_qubit(self, qubit_id, new_qubit):
        for i, qubit in enumerate(self._qubits):
            if qubit['id'] == qubit_id:
                self._qubits[i] = new_qubit
                return

    def reset_qubit_id(self, qubit_id):
        for qubit in self._qubits:
            if qubit['id'] == qubit_id:
                qubit['m_state'] = init_qubit_state('C')
                qubit['etg'] = None
                qubit['age'] = 0
                qubit['purified'] = False
                return

    def apply_memory_transition(self, state):
        next_state = self._tm.T @ state
        next_state /= next_state.sum()
        next_state = np.random.choice(range(len(MEMORY_STATES)), p=next_state.flatten())
        new_state = np.zeros((len(MEMORY_STATES), 1))
        new_state[next_state, 0] = 1.0
        return new_state, next_state




def get_pairs_to_purify(node_A, node_B, num=2):
    pairs = []
    for qubit_A in node_A.qubits:
        if qubit_A['etg'] is not None and (not qubit_A['purified']):
            qubit_B = next((q for q in node_B.qubits if q['id'] == qubit_A['etg']), None)
            if qubit_B:
                # print(f"Pair: {qubit_A['id']} - {qubit_B['id']}")
                pairs.append((qubit_A, qubit_B))

    if len(pairs) >= num:
        return pairs[:num]

    return []

def init_qubit_state(state = 'C'):
    vec = np.zeros((len(MEMORY_STATES), 1))
    vec[MEMORY_STATES[state], 0] = 1.0  # init as Clean
    return vec

def project_qubit_vector(photon_vector, qubit_vector):
    # Pauli multiplication table (ignoring global phase)
    multiplication_table = [
        [0, 1, 2, 3],  # C * {C, X, Z, Y}
        [1, 0, 3, 2],  # X * {C, X, Z, Y}
        [2, 3, 0, 1],  # Z * {C, X, Z, Y}
        [3, 2, 1, 0],  # Y * {C, X, Z, Y}
    ]
    
    # Find the indices where the error occurs (index of 1)
    photon_error_index = np.where(photon_vector == 1)[0][0]
    if photon_error_index > 3:
        print("Photo is lost. Should not arrive here!")
        return None

    qubit_error_index = np.where(qubit_vector == 1)[0][0]
    if qubit_error_index > 3:
        print("Qubit is in E,R,M. Should not arrive here!")
        return None

    # Get the updated error index from the multiplication table
    updated_error_index = multiplication_table[photon_error_index][qubit_error_index]

    # Create the updated qubit vector
    updated_qubit_vector = np.zeros((len(MEMORY_STATES), 1))
    updated_qubit_vector[updated_error_index, 0] = 1.0
    
    return updated_qubit_vector

def get_qubit_state(state_vector):
    # Ensure that only one element in the vector is 1
    if np.count_nonzero(state_vector == 1) != 1:
        print("Invalid state vector. Only one state should be active (1).")
        return None

    # Find the index of the active state
    active_state_index = np.where(state_vector == 1)[0][0]

    # Find the corresponding state by its index
    for state, index in MEMORY_STATES.items():
        if index == active_state_index:
            return state

    return None