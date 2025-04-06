# Simulating Quantum Communications: A Beginner's Tutorial

<p align="center">
  <img src="quantum_link.gif" alt="Purification demo" />
</p>

This repository contains a series of quantum entanglement simulations using Markov processes for tracking errors. Each folder corresponds to an article in our tutorial series on simulating quantum entanglements for beginners.

We currently have two articles. See below.

## Overview

The simulations model entanglement distribution between two quantum nodes using noisy quantum memories and a lossy quantum channel. Key features include:

- **Bell State Generation:** Simulation of photon emissions, channel losses, memory errors, and Bell state measurement.
- **Distance Variation:** Investigation of how increasing channel distances affect the fidelity of the generated (Phi+) state.
- **Performance Metrics:** Analysis and plotting of fidelity, full Bell state ratios, entanglement generation rate, and entanglement generation probability.
- **Entanglement Purification:** Various circuits and schedules for purification.

## Repository Contents

Each folder corresponds to an article in the tutorial series:

- **`article1`**  
  Contains a simulation demonstrating how increasing channel distances impact the fidelity of generated Bell states, as well as the entanglement generation rate and probability.

- **`article2`**  
  Contains a simulation demonstrating how various purification circuits impact the fidelity of generated Bell states over increasing channel distances.

*(Additional articles will be added in separate folders as the series progresses.)*

## Running the Simulations

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/quantum-entanglement-simulation.git
   cd quantum-entanglement-simulation
   ```

2. **Install Dependencies:**

   The simulation requires Python 3 and the following packages:
   - `numpy`
   - `matplotlib`

   Install the required packages using pip:

   ```bash
   pip install numpy matplotlib
   ```

## Usage

To run the simulation and generate the plots for the desired article (e.g., article1):

1. **Navigate to the Article Folder:**

   ```bash
   cd article1
   ```

2. **Run the Simulation Script:**

   ```bash
   python simulations.py
   ```

## Blog Tutorial

For a detailed explanation of the simulation methodology, including the underlying quantum communication principles and a step-by-step code walkthrough, check out the full tutorial series:  
- [Simulating Quantum Communications: A Beginner's Tutorial – Part I](https://medium.com/@amar.abane.phd/simulating-quantum-communications-a-beginners-tutorial-part-i-03224c2a4108)
- [Simulating Quantum Communications: A Beginner’s Tutorial — Part II (Entanglement Purification)](https://medium.com/@amar.abane.phd/simulating-quantum-communications-a-beginners-tutorial-part-ii-entanglement-purification-bedec75bf5a5)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have suggestions, improvements, or bug fixes.