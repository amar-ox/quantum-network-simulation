import numpy as np
import random
import matplotlib.pyplot as plt
from lib_dtmc import CHANNEL_STATES, MEMORY_STATES, init_memory_transition_matrix, init_channel_transition_matrix, apply_channel_transition
from lib_memory import Memory, get_bell_state, get_qubit_state, project_qubit_vector

def run_simulation(L, simulation_time=1):
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

    # Initialize node memories
    node_A = Memory('NodeA', num_qubits, T_memory, t_cutoff)
    node_B = Memory('NodeB', num_qubits, T_memory, t_cutoff)

    # Bell pairs statistics and entanglement counters
    states = {'Phi+': 0, 'Phi-': 0, 'Psi+': 0, 'Psi-': 0}
    dt_entanglements = 0
    entanglement_attempts = 0

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
                            
    etg_rate = dt_entanglements / simulation_time
    entanglement_proba = dt_entanglements / entanglement_attempts

    total_count = sum(states.values())
    if total_count > 0:
        percentages = {state: (count / total_count) * 100 for state, count in states.items()}
    else:
        percentages = {state: 0 for state in states}
    return etg_rate, entanglement_proba, percentages

# Define a range of distances (in km)
distances = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Store results for each distance
etg_rates = []
etg_probabilities = []
phi_fidelities = []
bell_ratios = {state: [] for state in ['Phi+', 'Phi-', 'Psi+', 'Psi-']}

# Run simulation for each distance
for L in distances:
    print(f"Simulating for L = {L} km...")
    etg_rate, entanglement_proba, percentages = run_simulation(L, simulation_time=1)
    etg_rates.append(etg_rate)
    etg_probabilities.append(entanglement_proba)
    phi_fidelities.append(percentages['Phi+'])
    for state in bell_ratios:
        bell_ratios[state].append(percentages[state])

plt.figure(figsize=(10, 8))

# Plot Fidelity vs. Distance
plt.subplot(2, 2, 1)
plt.plot(distances, phi_fidelities, marker='o', linestyle='-', color='b')
plt.xlabel('Total Distance L (km)')
plt.ylabel('Fidelity (% of Phi+ state)')
plt.title('Fidelity vs. Distance')
plt.grid(True)

# Plot Bell State Ratios vs. Distance
plt.subplot(2, 2, 2)
for state, ratios in bell_ratios.items():
    plt.plot(distances, ratios, marker='o', label=state)
plt.xlabel('Total Distance L (km)')
plt.ylabel('Bell State Ratio (%)')
plt.title('Bell States Ratios vs. Distance')
plt.legend()
plt.grid(True)

# Plot Entanglement Generation Rate vs. Distance
plt.subplot(2, 2, 3)
plt.plot(distances, etg_rates, marker='o', linestyle='-', color='g')
plt.xlabel('Total Distance L (km)')
plt.ylabel('Generation Rate (pairs/s)')
plt.title('Entanglement Generation Rate vs. Distance')
plt.grid(True)

# Plot Entanglement Generation Probability vs. Distance
plt.subplot(2, 2, 4)
plt.plot(distances, etg_probabilities, marker='o', linestyle='-', color='r')
plt.xlabel('Total Distance L (km)')
plt.ylabel('Generation Probability')
plt.title('Entanglement Generation Probability vs. Distance')
plt.grid(True)

plt.tight_layout()
plt.show()