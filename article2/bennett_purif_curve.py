import numpy as np
import matplotlib.pyplot as plt

def F_prime(F):
    numerator = F**2 + (1/9) * (1 - F)**2
    denominator = F**2 + (2/3) * F * (1 - F) + (5/9) * (1 - F)**2
    return numerator / denominator

F_values = np.linspace(0, 1, 500)
F_prime_values = F_prime(F_values)

plt.plot(F_values, F_prime_values, label="$F'$")
plt.plot(F_values, F_values, '--', label="$F = F'$")
plt.axvline(x=0.5, color='red', linestyle=':', label="$F=0.5$")

# Labels and legend
plt.xlabel("Input Fidelity $F$")
plt.ylabel("Output Fidelity $F'$")
plt.title("Purification Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
