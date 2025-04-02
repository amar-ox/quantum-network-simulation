import numpy as np
import random
import matplotlib.pyplot as plt

from lib_purification import calculate_total_pairs_needed, do_purification
from mc_dtmc_one_link_purif import run_simulation


# Distances (in km)
distances = [10, 20, 30, 40, 50]

# Purification schedules to test
# `None` corresponds to no purification
purif_schedules = [None, 'SsSpX', 'DsSp', 'SsDp', 'DsDp', 'SsSpX|SsSpX+SsSpZ']

n_runs = 5

fidelities_before = {sched: [] for sched in purif_schedules}
fidelities_after = {sched: [] for sched in purif_schedules}

for sched in purif_schedules:
    print(f"Testing purification schedule: {sched}")
    for L in distances:
        print(f"  Distance: {L} km")
        before_runs = []
        after_runs = []
        for i in range(n_runs):
            print(f"    Simulation run {i+1}")
            _, _, _, percentages, purif_percentages = run_simulation(L, sched)
            before_runs.append(percentages['Phi+'])
            after_runs.append(purif_percentages['Phi+'])

        mean_before = np.mean(before_runs)
        mean_after = np.mean(after_runs)
        
        fidelities_before[sched].append(mean_before)
        fidelities_after[sched].append(mean_after)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(distances, fidelities_before[None], marker='o', linestyle='--', label='No Purification')
for sched in purif_schedules:
    if sched is not None:
        plt.plot(distances, fidelities_after[sched], marker='x', linestyle='-', label=f'{sched}')

plt.xlabel('Total Distance L (km)')
plt.ylabel('Mean Fidelity (% Phi+)')
plt.title('Fidelity vs Distance: No Purification vs. Purification Circuits')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()