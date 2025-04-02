import numpy as np
import random
from lib_dtmc import CHANNEL_STATES, MEMORY_STATES, init_memory_transition_matrix, init_channel_transition_matrix, apply_channel_transition
from lib_memory import Memory, get_bell_state, get_qubit_state, project_qubit_vector, get_pairs_to_purify
from lib_purification import calculate_total_pairs_needed, do_purification


def run_simulation(L, PURIF_SCHEDULE, simulation_time=1):
    # Memory errors parameters
    memory_x_err_rate = 0.01            # per second
    memory_y_err_rate = 0.01            # per second
    memory_z_err_rate = 0.01            # per second
    memory_excitation_rate = 0.0        # qubit α|0> + β|0> forced to |1>
    memory_relaxation_rate = 0.0        # qubit loses energy to |0> (Amplitude damping)

    num_qubits = 10                     # Number of qubits per node
    t_cutoff = 500                      # timesteps (simplified T1)

    # Quantum channel parameters (L is in km)
    half_distance = int(L / 2)        # distance from node to BSA in km
    c = 3e8                           # speed of light in vacuum (m/s)
    n_fiber = 1.44                    # refractive index of fiber
    v_fiber = c / n_fiber             # speed in fiber (m/s)
    half_distance_m = half_distance * 1e3  # km -> m

    # Error rates (per km)
    lambda_loss = 0.02  
    lambda_x = 0.01
    lambda_y = 0.01     
    lambda_z = 0.01     

    # BSA parameters
    collection_efficiency = 0.8         
    indistinguishability_window = 1e-9  
    timing_jitter = 50e-12              
    detection_efficiency = 0.99         
    dark_count_probability = 0.01       

    # Source parameters
    emission_success_probability = 1.0  
    emission_x_error_rate = 0.01
    emission_y_error_rate = 0.01
    emission_z_error_rate = 0.01

    f_source = 1e5  # source repetition rate (photons/s)
    t_classical = half_distance_m / v_fiber   # BSA-to-Node ack travel time
    t_photon = half_distance_m / v_fiber      # Node-to-BSA photon travel time
    t_qubit_reset = 1e-6                      # time to reset qubit

    # Per-qubit emission rate (limited by the slowest process)
    per_qubit_emission_rate = min(f_source, int(1 / (t_classical + t_photon + t_qubit_reset)))
    photon_emission_rate = num_qubits * per_qubit_emission_rate
    time_step = 1 / photon_emission_rate
    num_steps = int(simulation_time / time_step)

    # Set up memory and channel transition matrices
    T_memory = init_memory_transition_matrix(
        memory_x_err_rate, memory_z_err_rate, memory_y_err_rate, 
        memory_excitation_rate, memory_relaxation_rate, time_step)
    T_channel = init_channel_transition_matrix(lambda_loss, lambda_x, lambda_z, lambda_y, half_distance)

    # Calculate number of input pairs needed
    if PURIF_SCHEDULE:
        NUM_PAIRS_TO_PURIF = calculate_total_pairs_needed(PURIF_SCHEDULE)
        print(f"  Purification schedule {PURIF_SCHEDULE} requires {NUM_PAIRS_TO_PURIF} pairs")
    
    # Initialize node memories
    node_A = Memory('NodeA', num_qubits, T_memory, t_cutoff)
    node_B = Memory('NodeB', num_qubits, T_memory, t_cutoff)

    # Bell pairs statistics and entanglement counters
    states = {'Phi+': 0, 'Phi-': 0, 'Psi+': 0, 'Psi-': 0}
    dt_entanglements = 0
    entanglement_attempts = 0

    purif_states = {'Phi+': 0, 'Phi-': 0, 'Psi+': 0, 'Psi-': 0}
    successful_purifications = 0
    attempt_purifications = 0

    for step in range(num_steps):
        # Update memory for idle qubits
        node_A.update_memory()    
        node_B.update_memory()

        qubit_A, qubit_A_index = node_A.get_next_available_qubit()
        qubit_B, qubit_B_index = node_B.get_next_available_qubit()
        if qubit_A is None or qubit_B is None:
            continue

        # Apply memory errors during the time step (age=1)
        qubit_A['m_state'], state_index_A = node_A.apply_memory_transition(qubit_A['m_state'])
        qubit_B['m_state'], state_index_B = node_B.apply_memory_transition(qubit_B['m_state'])

        # Simulate photon emission from nodes A and B
        emission_success_A = random.random() < emission_success_probability
        emission_success_B = random.random() < emission_success_probability

        emission_time_A = step * time_step + random.gauss(0, timing_jitter)
        emission_time_B = step * time_step + random.gauss(0, timing_jitter)

        entanglement_attempts += 1

        # Photon simulation for Node A
        if emission_success_A:
            photon_A = np.zeros((len(CHANNEL_STATES), 1))
            # initial clean state with error applied
            error_rates = [emission_x_error_rate, emission_z_error_rate, emission_y_error_rate]
            error_state = np.random.choice(
                [CHANNEL_STATES['X'], CHANNEL_STATES['Z'], CHANNEL_STATES['Y'], CHANNEL_STATES['C']],
                p=[*error_rates, 1 - sum(error_rates)]
            )
            photon_A[error_state, 0] = 1.0
            photon_A, _ = apply_channel_transition(photon_A, T_channel)
            photon_state_A = np.argmax(photon_A)  # index representing state
        else:
            photon_A = None
            photon_state_A = CHANNEL_STATES['L']

        # Photon simulation for Node B
        if emission_success_B:
            photon_B = np.zeros((len(CHANNEL_STATES), 1))
            error_rates = [emission_x_error_rate, emission_z_error_rate, emission_y_error_rate]
            error_state = np.random.choice(
                [CHANNEL_STATES['X'], CHANNEL_STATES['Z'], CHANNEL_STATES['Y'], CHANNEL_STATES['C']],
                p=[*error_rates, 1 - sum(error_rates)]
            )
            photon_B[error_state, 0] = 1.0
            photon_B, _ = apply_channel_transition(photon_B, T_channel)
            photon_state_B = np.argmax(photon_B)
        else:
            photon_B = None
            photon_state_B = CHANNEL_STATES['L']

        # BSA measurement: check if both photons are present and not lost
        if photon_state_A != CHANNEL_STATES['L'] and photon_state_B != CHANNEL_STATES['L']:
            collected_A = random.random() < collection_efficiency
            arrival_time_A = emission_time_A + half_distance_m / v_fiber

            collected_B = random.random() < collection_efficiency
            arrival_time_B = emission_time_B + half_distance_m / v_fiber

            if collected_A and collected_B:
                indistinguishable = abs(arrival_time_A - arrival_time_B) < indistinguishability_window
            else:
                indistinguishable = False

            if indistinguishable:
                # Check if qubits are still in valid states
                if (state_index_A in [MEMORY_STATES['C'], MEMORY_STATES['X'], MEMORY_STATES['Y'], MEMORY_STATES['Z']] and
                    state_index_B in [MEMORY_STATES['C'], MEMORY_STATES['X'], MEMORY_STATES['Y'], MEMORY_STATES['Z']]):
                    if random.random() < detection_efficiency:
                        if random.random() >= dark_count_probability:
                            dt_entanglements += 1

                            # Project errors from the photons to the memories
                            m_new_qubit_A = project_qubit_vector(photon_A, qubit_A['m_state'])
                            m_new_qubit_B = project_qubit_vector(photon_B, qubit_B['m_state'])
                            state = get_bell_state(get_qubit_state(m_new_qubit_A), get_qubit_state(m_new_qubit_B))

                            if state in states:
                                states[state] += 1

                            # Note: optics-based BSA can only distinguish Phi and Psi
                            # Update memories
                            qubit_A['m_state'] = m_new_qubit_A
                            qubit_A['age'] = 1
                            qubit_A['etg'] = qubit_B['id']
                            node_A.replace_qubit(qubit_A['id'], qubit_A)
                            
                            qubit_B['m_state'] = m_new_qubit_B
                            qubit_B['age'] = 1
                            qubit_B['etg'] = qubit_A['id']
                            node_B.replace_qubit(qubit_B['id'], qubit_B)

                            if PURIF_SCHEDULE is None:
                                continue
                            # If purification: Check if we have enough pairs for the circuit
                            # We assume purif is done in parralel to ONE entanglement attempt (= one timestep)
                            pairs_to_purify = get_pairs_to_purify(node_A, node_B, NUM_PAIRS_TO_PURIF)
                            if pairs_to_purify:
                                attempt_purifications += 1
                                purified_pair = do_purification(PURIF_SCHEDULE, pairs_to_purify)
                                if purified_pair is not None:
                                    successful_purifications += 1
                                    (p_qubit_A1, p_qubit_B1) = purified_pair
                                    node_A.replace_qubit(p_qubit_A1['id'], p_qubit_A1)
                                    node_B.replace_qubit(p_qubit_B1['id'], p_qubit_B1)
                                    for pair in pairs_to_purify[1:]:    # the 1st pair is always the purified one
                                        (qubit_A, qubit_B) = pair
                                        node_A.reset_qubit_id(qubit_A['id'])
                                        node_B.reset_qubit_id(qubit_B['id'])

                                    state = get_bell_state(get_qubit_state(p_qubit_A1['m_state']), get_qubit_state(p_qubit_B1['m_state']))
                                    purif_states[state]+=1
                                else:
                                    for pair in pairs_to_purify:
                                        (qubit_A, qubit_B) = pair
                                        node_A.reset_qubit_id(qubit_A['id'])
                                        node_B.reset_qubit_id(qubit_B['id'])
                            
    etg_rate = dt_entanglements / simulation_time
    entanglement_proba = dt_entanglements / entanglement_attempts

    total_count = sum(states.values())
    if total_count > 0:
        percentages = {state: (count / total_count) * 100 for state, count in states.items()}
    else:
        percentages = {state: 0 for state in states}
        
    purif_proba = successful_purifications / attempt_purifications if attempt_purifications else 0
    purif_total_count = sum(purif_states.values())
    if purif_total_count > 0:
        purif_percentages = {state: (count / purif_total_count) * 100 for state, count in purif_states.items()}
    else:
        purif_percentages = {state: 0 for state in purif_states}

    return etg_rate, entanglement_proba, purif_proba, percentages, purif_percentages


if __name__ == "__main__":
    etg_rate, entanglement_proba, purif_proba, percentages, purif_percentages = run_simulation(10, None)
    print()
    print(f"Entanglement Generation Rate: {etg_rate} pairs per second")
    print(f"Entanglement Generation Probability: {entanglement_proba}")
    print(f"Generated Bell states:")
    print(percentages)
    print()
    print(f"Purification Success Probability: {purif_proba}")
    print(f"Purified Bell states:")
    print(purif_percentages)