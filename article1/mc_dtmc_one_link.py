import numpy as np
import random
from lib_dtmc import CHANNEL_STATES, MEMORY_STATES, init_memory_transition_matrix, init_channel_transition_matrix, apply_channel_transition
from lib_memory import Memory, get_bell_state, get_qubit_state, project_qubit_vector


# Memory errors parameters
memory_x_err_rate = 0.01            # per second
memory_y_err_rate = 0.01            # per second
memory_z_err_rate = 0.01            # per second
memory_excitation_rate = 0.0        # qubit α|0> + β|0> is forced to make a transition to |1> state due to external energy absorbed from the environment
memory_relaxation_rate = 0.0        # the qubit unintentionally loses energy and is forced to move to the ground state |0>

num_qubits = 10                     # Number of qubits per node
t_cutoff = 500                      # We could use time instead of timesteps

# Quantum channel parameters
L = 10                          # Total length in km
half_distance = int(L / 2)      # Distance from node to BSA in km
c = 3e8                         # Speed of light in vacuum (m/s)
n_fiber = 1.44                  # Refractive index of fiber
v_fiber = c / n_fiber           # Speed of light in fiber (m/s)
half_distance_m = half_distance * 1e3  # Convert km to meters

# Error rates (per km)
lambda_loss = 0.02  # Loss rate per km
lambda_x = 0.01     # X error rate per km
lambda_y = 0.01     # Y error rate per km
lambda_z = 0.01     # Z error rate per km


# BSA parameters
collection_efficiency = 0.8          # Fraction of photons collected (0 <= value <= 1)
indistinguishability_window = 1e-9   # Indistinguishability window in seconds (e.g., 1 ns)
timing_jitter = 50e-12               # Timing jitter in seconds (e.g., 50 ps)
detection_efficiency = 0.99          # Probability of detecting an arriving photon. Excluding the 50% linear optics limitation
dark_count_probability = 0.01        # Probability of a dark count per detection event


# Source parameters
# Emission
emission_success_probability = 1.0    # emitted, collected, coupled into the fiber
emission_x_error_rate = 0.01
emission_y_error_rate = 0.01
emission_z_error_rate = 0.01

# Emission rate
f_source = 1e6                                          # Repetition rate of the photon source (photons/second).
t_classical = half_distance_m / v_fiber                 # BSA-to-Node ack (seconds).
t_photon = half_distance_m / v_fiber                    # Node-to-BSA photons (seconds).
t_qubit_reset = 1e-6                                    # Time to reset or prepare the qubit for another emission (seconds).

# Per-qubit emission rate
per_qubit_emission_rate = min(f_source, int(1 / (t_classical + t_photon + t_qubit_reset)))       # ~ 1 round of BK protocol


# Total photon emission rate
photon_emission_rate = num_qubits * per_qubit_emission_rate     # correspond to attempts rate
print(f"Attempt rate: {photon_emission_rate} attempts/second")

time_step = 1 / photon_emission_rate    # Time between emissions
print(f"Timestep: {time_step} s ({time_step * 1e3} ms)")

simulation_time = 1                 # Total simulation time in seconds
print(f"Simulate: {simulation_time} seconds")

print(f"Memory t_cutoff: {t_cutoff * time_step * 1e3} ms ({t_cutoff} X timestep)")
print()

num_steps = int(simulation_time / time_step)

T_memory = init_memory_transition_matrix(
    memory_x_err_rate, memory_z_err_rate, memory_y_err_rate, 
    memory_excitation_rate, memory_relaxation_rate, 
    time_step)
T_channel = init_channel_transition_matrix(lambda_loss, lambda_x, lambda_z, lambda_y, half_distance)

# Initialize qubit at nodes (simulation vectors and IDs)
node_A = Memory('NodeA', num_qubits, T_memory, t_cutoff)
node_B = Memory('NodeB', num_qubits, T_memory, t_cutoff)

# Bell pairs stats
pairs_trace = []

states = {
    'Phi+': 0,
    'Phi-': 0,
    'Psi+': 0,
    'Psi-': 0
}

# entanglements stats
entanglement_attempts = 0
dt_entanglements = 0

print("Started...")

for step in range(num_steps):

    # simulate passing time for other entangled qubits
    node_A.update_memory()    
    node_B.update_memory()

    qubit_A, qubit_A_index = node_A.get_next_available_qubit()
    qubit_B, qubit_B_index = node_B.get_next_available_qubit()

    # if no qubit available -> advance time in memory and continue
    if qubit_A is None or qubit_B is None:
        # print("No free qubits")
        continue

    # print(f"Try to entangle qubits: {qubit_A['id']} x {qubit_B['id']}")

    # For this newly photon-qubit entanglemenet: apply memory errors for the duration of the attempt (one timestep -> age=1)
    qubit_A['m_state'], state_index_A = node_A.apply_memory_transition(qubit_A['m_state'])
    qubit_B['m_state'], state_index_B = node_B.apply_memory_transition(qubit_B['m_state'])

    # Nodes attempt to emit photons
    emission_success_A = random.random() < emission_success_probability
    emission_success_B = random.random() < emission_success_probability

    # Assign emission times with timing jitter
    emission_time_A = step * time_step + random.gauss(0, timing_jitter)
    emission_time_B = step * time_step + random.gauss(0, timing_jitter)

    entanglement_attempts += 1

    if emission_success_A:
        # Photon A simulation vector
        photon_A = np.zeros((len(CHANNEL_STATES), 1))
        photon_A[CHANNEL_STATES['C'], 0] = 1.0  # Start as Clean

        # Apply emission errors (instantaneous)
        error_rates = [emission_x_error_rate, emission_z_error_rate, emission_y_error_rate]
        error_state = np.random.choice(
            [CHANNEL_STATES['X'], CHANNEL_STATES['Z'], CHANNEL_STATES['Y'], CHANNEL_STATES['C']],
            p=[*error_rates, 1 - sum(error_rates)]
        )
        photon_A = np.zeros((len(CHANNEL_STATES), 1))
        photon_A[error_state, 0] = 1.0

        # Photons travel through the channel
        photon_A, photon_state_A = apply_channel_transition(photon_A, T_channel)
    else:
        photon_A = None  # No photon emitted
        photon_state_A = CHANNEL_STATES['L']

    if emission_success_B:
        # Photon B simulation vector
        photon_B = np.zeros((len(CHANNEL_STATES), 1))
        photon_B[CHANNEL_STATES['C'], 0] = 1.0  # Start as Clean

        # Apply emission errors (instantaneous)
        error_rates = [emission_x_error_rate, emission_z_error_rate, emission_y_error_rate]
        error_state = np.random.choice(
            [CHANNEL_STATES['X'], CHANNEL_STATES['Z'], CHANNEL_STATES['Y'], CHANNEL_STATES['C']],
            p=[*error_rates, 1 - sum(error_rates)]
        )
        photon_B = np.zeros((len(CHANNEL_STATES), 1))
        photon_B[error_state, 0] = 1.0

        # Photons travel through the channel
        photon_B, photon_state_B = apply_channel_transition(photon_B,T_channel)
    else:
        photon_B = None  # No photon emitted
        photon_state_B = CHANNEL_STATES['L']


    # BSA measurement
    if photon_state_A != CHANNEL_STATES['L'] and photon_state_B != CHANNEL_STATES['L']:

        # Apply collection efficiency to photon_A
        collected_A = random.random() < collection_efficiency
        # Calculate arrival time at BSA for photon_A
        arrival_time_A = emission_time_A + half_distance_m / v_fiber

        # Apply collection efficiency to photon_B
        collected_B = random.random() < collection_efficiency
        # Calculate arrival time at BSA for photon_B
        arrival_time_B = emission_time_B + half_distance_m / v_fiber

        # Check if photons are collected
        if collected_A and collected_B:
        # Check if photons are within the indistinguishability window
            indistinguishable = abs(arrival_time_A - arrival_time_B) < indistinguishability_window
        else:
            indistinguishable = False

        # BSA measurement
        if indistinguishable:
            # Check if photons are still entangled with their qubits
            if (state_index_A in [MEMORY_STATES['C'], MEMORY_STATES['X'], MEMORY_STATES['Y'], MEMORY_STATES['Z']]
                and state_index_B in [MEMORY_STATES['C'], MEMORY_STATES['X'], MEMORY_STATES['Y'], MEMORY_STATES['Z']]):
                # Simulate BSA detection efficiency
                detection_success = random.random() < detection_efficiency
                if detection_success:
                    # Simulate dark counts
                    dark_count = random.random() < dark_count_probability
                    if not dark_count:

                        # Deduce projected state at BSA
                        state_at_bsa = get_bell_state(get_qubit_state(photon_A), get_qubit_state(photon_B))

                        # NOTE: BSA can only distinguish 2 out of 4 Bell states
                        if state_at_bsa in ['Phi+', 'Phi-', 'Psi+', 'Psi-']:
                            # Increment detected entanglements
                            dt_entanglements+=1

                            # Project photon errors to respective qubits
                            m_new_qubit_A = project_qubit_vector(photon_A, qubit_A['m_state'])
                            m_new_qubit_B = project_qubit_vector(photon_B, qubit_B['m_state'])

                            state = get_bell_state(get_qubit_state(m_new_qubit_A), get_qubit_state(m_new_qubit_B))
                            states[state]+=1

                            # Update qubit A
                            qubit_A['m_state'] = m_new_qubit_A
                            qubit_A['age'] = 1
                            qubit_A['etg'] = qubit_B['id']
                            node_A.replace_qubit(qubit_A['id'], qubit_A)
                            # print(f"Node A qubits now: {node_A_qubits}")

                            # Update qubit B                       
                            qubit_B['m_state'] = m_new_qubit_B
                            qubit_B['age'] = 1
                            qubit_B['etg'] = qubit_A['id']
                            node_B.replace_qubit(qubit_B['id'], qubit_B)
                            # print(f"Node B qubits now: {node_B_qubits}")

                            pairs_trace.append((get_qubit_state(qubit_A['m_state']), get_qubit_state(qubit_B['m_state'])))

print()

etg_rate = dt_entanglements / simulation_time
print(f"Entanglement Generation Rate: {etg_rate} pairs per second")

entanglement_proba = dt_entanglements / entanglement_attempts
print(f"Entanglement Generation Probability: {entanglement_proba}")

total_count = sum(states.values())
if total_count != 0:
    percentages = {state: (count / total_count) * 100 for state, count in states.items()}
    print(f"Generated Bell states from {total_count} pairs:")
    print(percentages)
