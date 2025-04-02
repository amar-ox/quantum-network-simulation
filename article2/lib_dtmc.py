import numpy as np
import random
from scipy.linalg import expm


# Channel state indices
CHANNEL_STATES = {
    'C': 0,
    'X': 1,
    'Z': 2,
    'Y': 3,
    'L': 4
}

# Memory state indices
MEMORY_STATES = {
    'C': CHANNEL_STATES['C'],
    'X': CHANNEL_STATES['X'],
    'Z': CHANNEL_STATES['Z'],
    'Y': CHANNEL_STATES['Y'],
    'E': 4,
    'R': 5,
    'M': 6
}


def init_memory_transition_matrix(x_err_rate, z_err_rate, y_err_rate, excitation_rate, relaxation_rate, time_step):
    # Construct the memory transition matrix
    T_memory = np.zeros((7, 7))

    # From Clean state
    Sigma_0 = x_err_rate + z_err_rate + y_err_rate + excitation_rate + relaxation_rate
    T_memory[MEMORY_STATES['C'], MEMORY_STATES['C']] = 1 - Sigma_0
    T_memory[MEMORY_STATES['C'], MEMORY_STATES['X']] = x_err_rate
    T_memory[MEMORY_STATES['C'], MEMORY_STATES['Z']] = z_err_rate
    T_memory[MEMORY_STATES['C'], MEMORY_STATES['Y']] = y_err_rate
    T_memory[MEMORY_STATES['C'], MEMORY_STATES['E']] = excitation_rate
    T_memory[MEMORY_STATES['C'], MEMORY_STATES['R']] = relaxation_rate

    # From Xerror state
    Sigma_1 = y_err_rate + z_err_rate + excitation_rate + relaxation_rate
    T_memory[MEMORY_STATES['X'], MEMORY_STATES['X']] = 1 - Sigma_1
    T_memory[MEMORY_STATES['X'], MEMORY_STATES['Z']] = y_err_rate
    T_memory[MEMORY_STATES['X'], MEMORY_STATES['Y']] = z_err_rate
    T_memory[MEMORY_STATES['X'], MEMORY_STATES['E']] = excitation_rate
    T_memory[MEMORY_STATES['X'], MEMORY_STATES['R']] = relaxation_rate

    # From Zerror state
    Sigma_2 = y_err_rate + x_err_rate + excitation_rate + relaxation_rate
    T_memory[MEMORY_STATES['Z'], MEMORY_STATES['X']] = y_err_rate
    T_memory[MEMORY_STATES['Z'], MEMORY_STATES['Z']] = 1 - Sigma_2
    T_memory[MEMORY_STATES['Z'], MEMORY_STATES['Y']] = x_err_rate
    T_memory[MEMORY_STATES['Z'], MEMORY_STATES['E']] = excitation_rate
    T_memory[MEMORY_STATES['Z'], MEMORY_STATES['R']] = relaxation_rate

    # From Yerror state
    Sigma_3 = z_err_rate + x_err_rate + excitation_rate + relaxation_rate
    T_memory[MEMORY_STATES['Y'], MEMORY_STATES['X']] = z_err_rate
    T_memory[MEMORY_STATES['Y'], MEMORY_STATES['Z']] = x_err_rate
    T_memory[MEMORY_STATES['Y'], MEMORY_STATES['Y']] = 1 - Sigma_3
    T_memory[MEMORY_STATES['Y'], MEMORY_STATES['E']] = excitation_rate
    T_memory[MEMORY_STATES['Y'], MEMORY_STATES['R']] = relaxation_rate

    # From Excited state
    Sigma_4 = relaxation_rate
    T_memory[MEMORY_STATES['E'], MEMORY_STATES['E']] = 1 - Sigma_4
    T_memory[MEMORY_STATES['E'], MEMORY_STATES['R']] = relaxation_rate

    # From Relaxed state
    Sigma_5 = excitation_rate
    T_memory[MEMORY_STATES['R'], MEMORY_STATES['E']] = excitation_rate
    T_memory[MEMORY_STATES['R'], MEMORY_STATES['R']] = 1 - Sigma_5

    # From Mixed state
    Sigma_6 = excitation_rate + relaxation_rate
    T_memory[MEMORY_STATES['M'], MEMORY_STATES['E']] = excitation_rate
    T_memory[MEMORY_STATES['M'], MEMORY_STATES['R']] = relaxation_rate
    T_memory[MEMORY_STATES['M'], MEMORY_STATES['M']] = 1 - Sigma_6

    for i in range(7):
        row_sum = np.sum(T_memory[i, :])
        if not np.isclose(row_sum, 1.0):
            print(f"Warning: Row {i} in T_memory does not sum to 1 (sum = {row_sum})")

    T_memory[i, :] /= row_sum

    # Compute the cumulative transition matrix
    T_cumulative = expm(T_memory * time_step)
    return T_cumulative


def init_channel_transition_matrix(lambda_loss, lambda_x, lambda_z, lambda_y, distance):
    # Construct the channel transition matrix
    T_unit = np.zeros((5, 5))

    # From Clean state
    Sigma_0 = lambda_x + lambda_z + lambda_y + lambda_loss
    T_unit[CHANNEL_STATES['C'], CHANNEL_STATES['C']] = 1 - Sigma_0
    T_unit[CHANNEL_STATES['C'], CHANNEL_STATES['X']] = lambda_x
    T_unit[CHANNEL_STATES['C'], CHANNEL_STATES['Z']] = lambda_z
    T_unit[CHANNEL_STATES['C'], CHANNEL_STATES['Y']] = lambda_y
    T_unit[CHANNEL_STATES['C'], CHANNEL_STATES['L']] = lambda_loss

    # From Xerror state
    Sigma_1 = lambda_y + lambda_z + lambda_loss
    T_unit[CHANNEL_STATES['X'], CHANNEL_STATES['X']] = 1 - Sigma_1
    T_unit[CHANNEL_STATES['X'], CHANNEL_STATES['Z']] = lambda_y
    T_unit[CHANNEL_STATES['X'], CHANNEL_STATES['Y']] = lambda_z
    T_unit[CHANNEL_STATES['X'], CHANNEL_STATES['L']] = lambda_loss

    # From Zerror state
    Sigma_2 = lambda_y + lambda_x + lambda_loss
    T_unit[CHANNEL_STATES['Z'], CHANNEL_STATES['X']] = lambda_y
    T_unit[CHANNEL_STATES['Z'], CHANNEL_STATES['Z']] = 1 - Sigma_2
    T_unit[CHANNEL_STATES['Z'], CHANNEL_STATES['Y']] = lambda_x
    T_unit[CHANNEL_STATES['Z'], CHANNEL_STATES['L']] = lambda_loss

    # From Yerror state
    Sigma_3 = lambda_z + lambda_x + lambda_loss
    T_unit[CHANNEL_STATES['Y'], CHANNEL_STATES['X']] = lambda_z
    T_unit[CHANNEL_STATES['Y'], CHANNEL_STATES['Z']] = lambda_x
    T_unit[CHANNEL_STATES['Y'], CHANNEL_STATES['Y']] = 1 - Sigma_3
    T_unit[CHANNEL_STATES['Y'], CHANNEL_STATES['L']] = lambda_loss

    # From Lost state (absorbing state)
    T_unit[CHANNEL_STATES['L'], CHANNEL_STATES['L']] = 1.0

    for i in range(5):
        row_sum = np.sum(T_unit[i, :])
        if not np.isclose(row_sum, 1.0):
            T_unit[i, :] /= row_sum
            print(f"Warning: Row {i} in T_unit did not sum to 1. Normalized the row.")

    # Compute the cumulative transition matrix
    T_cumulative = np.linalg.matrix_power(T_unit, distance)
    return T_cumulative


# Function to apply channel transition
def apply_channel_transition(state, T_matrix):
    next_state = T_matrix.T @ state
    next_state /= next_state.sum()
    next_state = np.random.choice(range(len(CHANNEL_STATES)), p=next_state.flatten())
    new_state = np.zeros((len(CHANNEL_STATES), 1))
    new_state[next_state, 0] = 1.0
    return new_state, next_state
