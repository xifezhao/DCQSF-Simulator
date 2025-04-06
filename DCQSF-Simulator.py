# DCQSF Simulator - Complete Code 

import cirq
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt # For general plotting
# Import Qiskit visualization tools
from qiskit.visualization import plot_state_city, plot_bloch_multivector
# Import Qiskit's DensityMatrix, using an alias to avoid potential conflicts
from qiskit.quantum_info import DensityMatrix as qkDensityMatrix

# --- Helper Functions ---

def state_vector_to_density_matrix(state_vector: np.ndarray) -> np.ndarray:
    """Converts a state vector to a density matrix rho = |psi><psi|"""
    psi = np.array(state_vector, dtype=complex)
    if len(psi.shape) == 1:
        psi = psi[:, np.newaxis] # Convert to column vector (n, 1)
    elif psi.shape[1] != 1:
        raise ValueError("Input must be a 1D array or a column vector")
    return psi @ psi.conj().T # @ is matrix multiplication, .conj().T is conjugate transpose (dagger)

def create_projector_from_state_vector(state_vector: np.ndarray) -> np.ndarray:
    """Creates a projection operator P = |psi><psi| from a state vector"""
    return state_vector_to_density_matrix(state_vector)

def normalize_state_vector(state_vector: np.ndarray) -> np.ndarray:
    """Normalizes a state vector"""
    norm = np.linalg.norm(state_vector)
    if norm == 0:
        raise ValueError("Cannot normalize the zero vector")
    return state_vector / norm

# --- DCQSF Simulator Class ---

class DCQSF_Simulator:
    """
    Simulates the core concepts of the Dynamic Contextual Quantum Semantic Framework (DCQSF).
    """

    def __init__(self):
        """Initializes the simulator."""
        # Note: Current time is Saturday, April 5, 2025 at 5:08:57 PM CST.
        print("DCQSF Simulator Initialized.")

    def represent_entity_pure(self, state_vector: np.ndarray) -> np.ndarray:
        """Represents an entity with a pure state density matrix."""
        print(f"Representing entity (Pure State): Vector = {state_vector.flatten()}")
        normalized_sv = normalize_state_vector(state_vector)
        rho = state_vector_to_density_matrix(normalized_sv)
        print(f"  -> Density Matrix rho:\n{rho}\n")
        return rho

    def represent_entity_mixed(self, states_and_probabilities: List[Tuple[np.ndarray, float]]) -> np.ndarray:
        """Represents an entity with a mixed state density matrix."""
        print(f"Representing entity (Mixed State): {len(states_and_probabilities)} components")
        rho = None
        total_prob = 0.0
        dim = -1
        for i, (sv, prob) in enumerate(states_and_probabilities):
            if prob < 0 or prob > 1: raise ValueError(f"Probability p_{i}={prob} must be between [0, 1]")
            total_prob += prob
            normalized_sv = normalize_state_vector(sv)
            current_dim = normalized_sv.shape[0]
            if rho is None: # First component
                 dim = current_dim
                 # Check if dimension is a power of 2
                 if dim == 0 or (dim > 0 and (dim & (dim - 1) != 0)):
                     raise ValueError(f"State vector dimension {dim} must be a power of 2")
                 rho = np.zeros((dim, dim), dtype=complex) # Initialize density matrix
            elif current_dim != dim:
                 raise ValueError(f"All state vectors in a mixed state must have the same dimension ({dim})")
            rho_i = state_vector_to_density_matrix(normalized_sv)
            rho += prob * rho_i
            print(f"  Component {i}: Vector = {sv.flatten()}, Probability = {prob}")

        # Check if probabilities sum to 1
        if not np.isclose(total_prob, 1.0):
            print(f"Warning: Sum of probabilities {total_prob:.4f} is not close to 1. Normalizing.")
            if total_prob > 1e-9: # Avoid division by zero
               rho /= total_prob
            else:
               print("Warning: Total probability is close to zero, cannot normalize.")

        # Verify trace of the final density matrix is 1
        trace_rho = np.trace(rho)
        if not np.isclose(trace_rho, 1.0):
             print(f"Warning: Trace of final mixed state density matrix {trace_rho.real:.4f} is not close to 1.")
             # Optionally force trace normalization: rho /= trace_rho

        print(f"  -> Mixed Density Matrix rho:\n{rho}\n")
        return rho

    def integrate_context(self, rho_semantic: np.ndarray, rho_context: np.ndarray) -> np.ndarray:
        """Combines semantic and context states using tensor product."""
        print("Context Integration (Tensor Product):")
        print(f"  Semantic rho_semantic (Dimension {rho_semantic.shape[0]}):\n{rho_semantic}")
        print(f"  Context rho_context (Dimension {rho_context.shape[0]}):\n{rho_context}")
        rho_combined = np.kron(rho_semantic, rho_context)
        print(f"  -> Combined rho_combined (Dimension {rho_combined.shape[0]}):\n{rho_combined}\n")
        return rho_combined

    def calculate_relevance_projection(self, rho_doc: np.ndarray, P_query: np.ndarray) -> float:
        """Calculates relevance using projection operator."""
        print("Calculating Relevance (Projection Operator):")
        print(f"  Document rho_d:\n{rho_doc}")
        print(f"  Query Projector P_q:\n{P_query}")
        if rho_doc.shape != P_query.shape: raise ValueError("Dimensions do not match")
        # Relevance = Tr(P_q * rho_d)
        relevance = np.trace(P_query @ rho_doc).real # Result should be real
        # Clamp to [0, 1] to handle potential numerical errors
        relevance = max(0.0, min(1.0, relevance))
        print(f"  -> Relevance Tr(P_q * rho_d) = {relevance:.4f}\n")
        return relevance

    def calculate_relevance_fidelity(self, rho_doc: np.ndarray, rho_query: np.ndarray) -> float:
        """
        Calculates relevance using quantum fidelity.
        Implemented directly using NumPy for the standard definition:
        F(rho, sigma) = (Tr[sqrt(sqrt(rho) * sigma * sqrt(rho))])^2.
        Includes optimizations for pure states.
        """
        print("Calculating Relevance (Fidelity):")
        print(f"  Document rho_d (Dimension {rho_doc.shape}):\n{rho_doc}")
        print(f"  Query rho_q (Dimension {rho_query.shape}):\n{rho_query}")

        if rho_doc.shape != rho_query.shape:
            raise ValueError("Dimensions do not match")

        # Ensure matrices are Hermitian for numerical stability
        rho_doc = (rho_doc + rho_doc.conj().T) / 2
        rho_query = (rho_query + rho_query.conj().T) / 2

        try:
            # Check for pure states using purity Tr(rho^2) == 1
            # Use np.isclose for floating point comparisons
            is_pure_doc = np.isclose(np.trace(rho_doc @ rho_doc).real, 1.0)
            is_pure_query = np.isclose(np.trace(rho_query @ rho_query).real, 1.0)

            # If both are pure states, use simpler calculation F = Tr(rho_doc * rho_query)
            if is_pure_doc and is_pure_query:
                print("  Detected both states are pure, using simplified calculation F = Tr(rho_doc * rho_query)")
                # Note: For pure states Tr(rho_doc @ rho_query) = |<psi_doc|psi_query>|^2
                fidelity = np.trace(rho_doc @ rho_query).real
            # If one is pure and one is mixed, use F = Tr(rho_pure * rho_mix)
            elif (is_pure_query and not is_pure_doc) or (not is_pure_query and is_pure_doc):
                 print("  Detected one pure and one mixed state, using simplified calculation F = Tr(rho_query * rho_doc)")
                 # This works because F(pure, mix) = <psi_pure| rho_mix |psi_pure> = Tr(|psi><psi| rho_mix)
                 fidelity = np.trace(rho_query @ rho_doc).real
            # If both are mixed states, use the general formula
            else:
                print("  Detected both states are mixed (or numerically impure), using general fidelity formula")
                # Calculate sqrt(rho_query) using eigenvalue decomposition
                evals_q, evecs_q = np.linalg.eigh(rho_query)
                # Ensure eigenvalues are non-negative due to potential numerical errors
                sqrt_evals_q = np.sqrt(np.maximum(evals_q, 0))
                sqrt_rho_q = evecs_q @ np.diag(sqrt_evals_q) @ evecs_q.conj().T
                # Ensure result is Hermitian
                sqrt_rho_q = (sqrt_rho_q + sqrt_rho_q.conj().T) / 2

                # Calculate M = sqrt(rho_query) * rho_doc * sqrt(rho_query)
                M = sqrt_rho_q @ rho_doc @ sqrt_rho_q
                # Ensure M is Hermitian
                M = (M + M.conj().T) / 2

                # Calculate the trace of the square root of M
                # using the sum of the square roots of M's eigenvalues
                evals_M = np.linalg.eigvalsh(M) # Eigenvalues of Hermitian matrix M
                 # Ensure non-negative eigenvalues
                sqrt_evals_M = np.sqrt(np.maximum(evals_M, 0))
                # Fidelity is the square of the sum of sqrt(eigenvalues)
                fidelity = (np.sum(sqrt_evals_M))**2

            # Ensure result is real and clamped to [0, 1]
            fidelity = max(0.0, min(1.0, fidelity.real))
            print(f"  -> Relevance (Fidelity) F(rho_q, rho_d) = {fidelity:.4f}\n")
            return fidelity

        except np.linalg.LinAlgError as lae:
            print(f"  Linear algebra error during fidelity calculation: {lae}")
            print("  This might be due to numerical instability (e.g., matrix square root).")
            print(f"  -> Relevance (Fidelity) F(rho_q, rho_d) = Calculation Failed\n")
            return -1.0
        except Exception as e:
            print(f"  Unknown error during fidelity calculation: {e}")
            print(f"  -> Relevance (Fidelity) F(rho_q, rho_d) = Calculation Failed\n")
            return -1.0


    def evolve_state(self, rho_initial: np.ndarray, kraus_operators: List[np.ndarray]) -> np.ndarray:
        """Evolves a state using quantum operations (Kraus operators)."""
        print("State Evolution (Quantum Operation):")
        print(f"  Initial state rho_initial:\n{rho_initial}")
        print(f"  Kraus Operators ({len(kraus_operators)}):")
        sum_Ak_dagger_Ak = None # For trace-preserving check

        for i, A_k in enumerate(kraus_operators):
            print(f"    A_{i}:\n{A_k}")
            if A_k.shape[1] != rho_initial.shape[0]:
                 raise ValueError(f"Dimension of Kraus operator A_{i} ({A_k.shape}) incompatible with rho ({rho_initial.shape})")
            if A_k.shape[0] != rho_initial.shape[0]:
                 # This check is also usually needed; Kraus operators are often square
                 # or at least the output dimension should match if the space doesn't change.
                 print(f"Warning: Output dimension of Kraus operator A_{i} ({A_k.shape[0]}) does not match rho ({rho_initial.shape[0]}).")

            # Accumulate A_k^dagger * A_k for verification
            if sum_Ak_dagger_Ak is None:
                 sum_Ak_dagger_Ak = np.zeros((A_k.shape[1], A_k.shape[1]), dtype=complex)
            sum_Ak_dagger_Ak += A_k.conj().T @ A_k

        # Verify trace-preserving condition sum_k A_k^dagger * A_k = I (optional)
        dim = rho_initial.shape[0]
        identity = np.eye(dim, dtype=complex)
        is_trace_preserving = False
        if sum_Ak_dagger_Ak is not None:
            is_trace_preserving = np.allclose(sum_Ak_dagger_Ak, identity)
            if not is_trace_preserving:
                print("Warning: Kraus operators do not satisfy trace-preserving condition sum(A_k^dagger * A_k) = I.")
                print(f"  sum(A_k^dagger * A_k) =\n{sum_Ak_dagger_Ak}")
                print("  This indicates the operation might not be a physically valid, completely described quantum channel (could be non-trace-preserving or part of a measurement).")

        # Apply evolution formula: rho_final = sum_k A_k * rho_initial * A_k^dagger
        # Output dimension is determined by the number of rows in A_k
        output_dim = kraus_operators[0].shape[0] if kraus_operators else dim
        rho_final = np.zeros((output_dim, output_dim), dtype=complex)
        for A_k in kraus_operators:
            rho_final += A_k @ rho_initial @ A_k.conj().T

        # Verify trace after evolution (should remain 1 if operation is trace-preserving and started at 1)
        trace_final = np.trace(rho_final).real
        print(f"  Evolved state rho_final:\n{rho_final}")
        print(f"  Trace of evolved state Tr(rho_final) = {trace_final:.4f}\n")

        initial_trace_is_one = np.isclose(np.trace(rho_initial).real, 1.0)
        if is_trace_preserving and initial_trace_is_one and not np.isclose(trace_final, 1.0):
             print(f"Warning: After trace-preserving operation, final trace {trace_final:.4f} is not close to 1 (potential numerical error).")
             # Optionally force trace normalization: rho_final /= trace_final

        return rho_final

# --- Visualization Helper Functions ---

def visualize_density_matrix_city(rho: np.ndarray, title: str):
    """Visualizes a density matrix using Qiskit's plot_state_city"""
    try:
        # Convert NumPy array to Qiskit DensityMatrix object
        qiskit_rho = qkDensityMatrix(rho)
        fig = plot_state_city(qiskit_rho, title=title, figsize=(5, 5))
        plt.show() # Display plot
    except ImportError:
        print("Warning: Qiskit not installed or could not be imported. Cannot visualize density matrix. Please run 'pip install qiskit'")
    except Exception as e:
        print(f"Error visualizing density matrix '{title}': {e}")

def visualize_bloch_sphere(rho: np.ndarray, title: str):
    """Visualizes a single-qubit density matrix using Qiskit's plot_bloch_multivector"""
    if rho.shape != (2, 2):
        print(f"Warning: Bloch sphere visualization only applicable for single qubits (2x2 density matrix). Skipping '{title}'.")
        return
    try:
        qiskit_rho = qkDensityMatrix(rho)
        fig = plot_bloch_multivector(qiskit_rho, title=title)
        plt.show() # Display plot
    except ImportError:
        print("Warning: Qiskit not installed or could not be imported. Cannot visualize Bloch sphere. Please run 'pip install qiskit'")
    except Exception as e:
        print(f"Error visualizing Bloch sphere '{title}': {e}")


# --- Example Usage ---

if __name__ == "__main__":
    # Ensure Matplotlib and Qiskit are installed: pip install matplotlib qiskit
    print("Running DCQSF Simulation with Visualization...")
    simulator = DCQSF_Simulator()

    # --- 1. Entity Representation ---
    print("\n--- 1. Entity Representation ---")
    # Define some basic state vectors |0> and |1>
    q0 = np.array([1, 0], dtype=complex) # |0>
    q1 = np.array([0, 1], dtype=complex) # |1>
    # Define a superposition state |+> = (|0> + |1>) / sqrt(2)
    q_plus = normalize_state_vector(q0 + q1)
    # Define a mixed state representing ambiguity or uncertainty
    # 50% probability |0>, 50% probability |1> (maximally mixed state I/2)
    mixed_state_def = [(q0, 0.5), (q1, 0.5)]
    rho_mixed = simulator.represent_entity_mixed(mixed_state_def) # I/2

    # Query: Assume query explicitly points to the |+> state
    query_vector = q_plus
    rho_query = simulator.represent_entity_pure(query_vector) # |+><+|

    # Document 1: Assumed highly relevant, state close to |+> but slightly perturbed
    doc1_vector = normalize_state_vector(0.9 * q0 + 1.1 * q1) # Superposition slightly biased to |1>
    rho_doc1 = simulator.represent_entity_pure(doc1_vector)

    # Document 2: Assumed less relevant, state closer to |0>
    doc2_vector = normalize_state_vector(0.95 * q0 + 0.05 * q1) # Close to |0>
    rho_doc2 = simulator.represent_entity_pure(doc2_vector)

    # Document 3: Assumed partially relevant, represented as a mixed state
    doc3_mixed_def = [(q_plus, 0.6), (q0, 0.4)] # 60% like query |+>, 40% like |0>
    rho_doc3 = simulator.represent_entity_mixed(doc3_mixed_def)

    # Initial User State: Assume user interest is initially vague (maximally mixed state I/2)
    user_initial_vector_def = [(q0, 0.5), (q1, 0.5)]
    rho_user_initial = simulator.represent_entity_mixed(user_initial_vector_def) # I/2

    # --- 1.1 Visualization: Density Matrix Representations ---
    print("\n--- 1.1 Visualization: Density Matrix Representations (Qiskit State City Plot) ---")
    print("Description: Shows the structure of different state types.")
    print(" - Diagonal elements represent populations in basis states.")
    print(" - Off-diagonal elements (real and imaginary parts) represent coherences.")
    visualize_density_matrix_city(rho_query, "Query (Pure State |+>)")
    visualize_density_matrix_city(rho_doc3, "Document 3 (Mixed State 0.6|+><+| + 0.4|0><0|)")
    visualize_density_matrix_city(rho_user_initial, "Initial User (Maximally Mixed State I/2)")

    # --- 2. Context Integration ---
    print("\n--- 2. Context Integration ---")
    # Combine Query rho_query and Initial User State rho_user_initial
    # Note: Space dimension increases (1 qubit + 1 qubit = 2 qubits = 4 dimensions)
    rho_query_user_context = simulator.integrate_context(rho_query, rho_user_initial)
    # Optional: Visualize the combined 4x4 matrix (more complex to interpret)
    # visualize_density_matrix_city(rho_query_user_context, "Combined Query-User State")

    # --- 3. Relevance Determination ---
    print("\n--- 3. Relevance Determination ---")
    # Method A: Projection Operator
    # Query Projector P_q = |psi_q><psi_q|
    P_query = create_projector_from_state_vector(query_vector)
    print("\n--- Relevance Calculation (Projection) ---")
    relevance_doc1_proj = simulator.calculate_relevance_projection(rho_doc1, P_query)
    relevance_doc2_proj = simulator.calculate_relevance_projection(rho_doc2, P_query)
    relevance_doc3_proj = simulator.calculate_relevance_projection(rho_doc3, P_query)

    # Method B: Fidelity (using NumPy implementation for standard definition)
    print("\n--- Relevance Calculation (Fidelity - NumPy Impl.) ---")
    relevance_doc1_fid = simulator.calculate_relevance_fidelity(rho_doc1, rho_query)
    relevance_doc2_fid = simulator.calculate_relevance_fidelity(rho_doc2, rho_query)
    relevance_doc3_fid = simulator.calculate_relevance_fidelity(rho_doc3, rho_query)

    print("\n--- Relevance Score Summary ---")
    print(f"Document 1: Projection Relevance={relevance_doc1_proj:.4f}, Fidelity={relevance_doc1_fid:.4f}")
    print(f"Document 2: Projection Relevance={relevance_doc2_proj:.4f}, Fidelity={relevance_doc2_fid:.4f}")
    print(f"Document 3: Projection Relevance={relevance_doc3_proj:.4f}, Fidelity={relevance_doc3_fid:.4f}")
    # Expect Fidelity == Projection for pure-pure and pure-mixed cases now

    # --- 3.1 Visualization: Relevance Score Comparison ---
    print("\n--- 3.1 Visualization: Relevance Score Comparison (Bar Chart) ---")
    doc_labels = ['Document 1', 'Document 2', 'Document 3']
    proj_scores = [relevance_doc1_proj, relevance_doc2_proj, relevance_doc3_proj]
    # Use -1 for failed calculations if fidelity returned -1.0
    fid_scores = [f if f >= 0 else 0 for f in [relevance_doc1_fid, relevance_doc2_fid, relevance_doc3_fid]] # Replace -1 with 0 for plotting

    x = np.arange(len(doc_labels)) # label locations
    width = 0.35 # width of the bars

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, proj_scores, width, label='Projection Tr(Pq * ρd)')
    rects2 = ax.bar(x + width/2, fid_scores, width, label='Fidelity F(ρq, ρd)')

    # Add some text for labels, title and axes ticks
    ax.set_ylabel('Relevance Score')
    ax.set_title('Relevance Score Comparison for Documents vs. Query')
    ax.set_xticks(x)
    ax.set_xticklabels(doc_labels)
    ax.legend()
    ax.bar_label(rects1, padding=3, fmt='%.4f')
    ax.bar_label(rects2, padding=3, fmt='%.4f')
    ax.set_ylim(0, 1.1) # Set Y-axis limits for better label visibility

    fig.tight_layout()
    plt.show()


    # --- 4. Dynamic User State Evolution ---
    print("\n--- 4. Dynamic User State Evolution ---")
    # Assume user interacts with a "relevant" document (concept represented by query state)

    # Example 1: Depolarizing Channel
    p_depol = 0.2 # Depolarization probability
    dim_user = rho_user_initial.shape[0]
    I_user = np.eye(dim_user, dtype=complex) # Ensure I is complex
    # Pauli matrices (ensure complex type)
    X_user = cirq.unitary(cirq.X).astype(complex)
    Y_user = cirq.unitary(cirq.Y).astype(complex)
    Z_user = cirq.unitary(cirq.Z).astype(complex)

    kraus_depol = [
        np.sqrt(1 - 3*p_depol/4) * I_user,
        np.sqrt(p_depol/4) * X_user,
        np.sqrt(p_depol/4) * Y_user,
        np.sqrt(p_depol/4) * Z_user
    ]
    print("\n--- User State Evolution (Example: Depolarizing Channel) ---")
    rho_user_evolved_depol = simulator.evolve_state(rho_user_initial, kraus_depol) # Result should still be I/2

    # Example 2: Unitary Rotation
    theta = np.pi / 8 # Rotation angle
    U_interaction = cirq.unitary(cirq.Ry(rads=theta)).astype(complex)
    # Single unitary evolution is a special quantum operation with one Kraus operator A0 = U
    kraus_rotation = [U_interaction]
    print("\n--- User State Evolution (Example: Unitary Rotation) ---")
    rho_user_evolved_rot = simulator.evolve_state(rho_user_initial, kraus_rotation) # Result should still be I/2

    # --- Fidelity of Evolved User States vs. Query ---
    print("\n--- Fidelity of Evolved User States vs. Query ---")
    # Note: Comparing user state rho_user (I/2 before/after) with query rho_query (|+><+|)
    fid_initial_user_query = simulator.calculate_relevance_fidelity(rho_user_initial, rho_query)
    fid_evolved_depol_query = simulator.calculate_relevance_fidelity(rho_user_evolved_depol, rho_query)
    fid_evolved_rot_query = simulator.calculate_relevance_fidelity(rho_user_evolved_rot, rho_query)

    print("\n--- User State Evolution Summary ---")
    # Expect these three values to be ~0.5, since F(I/2, pure_state) = 0.5
    print(f"Initial User State vs Query: Fidelity = {fid_initial_user_query:.4f}")
    print(f"User State after Depolarizing Channel vs Query: Fidelity = {fid_evolved_depol_query:.4f}")
    print(f"User State after Unitary Rotation vs Query: Fidelity = {fid_evolved_rot_query:.4f}")


    # --- 4.1 Visualization: User State Evolution (Bloch Sphere) ---
    print("\n--- 4.1 Visualization: User State Evolution (Bloch Sphere) ---")
    print("Description: Shows the user state on the Bloch sphere and its evolution.")
    print(" - Center of the sphere represents the maximally mixed state (I/2).")
    print(" - Surface of the sphere represents pure states.")
    visualize_bloch_sphere(rho_user_initial, "Initial User State (I/2)")
    visualize_bloch_sphere(rho_user_evolved_depol, "User State (After Depolarizing Channel)")
    visualize_bloch_sphere(rho_user_evolved_rot, "User State (After Unitary Rotation)")

    print("\n--- Simulation End ---")
