 # DCQSF Simulator

This repository contains a Python implementation of the Dynamic Contextual Quantum Semantic Framework (DCQSF) simulator. It provides tools for representing entities using quantum states, integrating contextual information, calculating relevance using projection and fidelity, and simulating user state evolution.

## Features

- **Entity Representation:** Supports both pure and mixed state representations of entities using density matrices.
- **Context Integration:** Combines semantic and contextual states using tensor products.
- **Relevance Calculation:** Implements relevance calculation using projection operators and quantum fidelity.
- **State Evolution:** Simulates user state evolution using quantum operations (Kraus operators).
- **Visualization:** Includes visualization tools for density matrices and Bloch spheres using Qiskit and Matplotlib.

## Prerequisites

- Python 3.6+
- NumPy
- Matplotlib
- Qiskit (for visualization)

You can install the required packages using pip:

```bash
pip install numpy matplotlib qiskit
