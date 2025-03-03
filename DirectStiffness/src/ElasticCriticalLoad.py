import numpy as np
from scipy.linalg import eig
from PostProcess import compute_local_forces_moments

def compute_geometric_stiffness_matrix(element, local_forces_moments):
    """
    Compute the geometric stiffness matrix for a given element.
    """
    # Extract forces and moments
    Fx1, Fy1, Fz1, Mx1, My1, Mz1, Fx2, Fy2, Fz2, Mx2, My2, Mz2 = local_forces_moments

    # Compute geometric stiffness matrix (simplified for brevity)
    L = element.L
    k_geometric_local = np.zeros((12, 12))

    # Fill in the matrix based on DSM theory
    # Example: Axial force contribution
    k_geometric_local[0, 0] = Fx1 / L
    k_geometric_local[6, 6] = Fx2 / L
    k_geometric_local[0, 6] = -Fx1 / L
    k_geometric_local[6, 0] = -Fx2 / L

    return k_geometric_local

def assemble_global_geometric_stiffness_matrix(structure, local_forces_moments):
    """
    Assemble the global geometric stiffness matrix.
    """
    num_nodes = len(structure.nodes)
    num_dofs = num_nodes * 6
    K_geometric_global = np.zeros((num_dofs, num_dofs))

    for element, forces_moments in zip(structure.elements, local_forces_moments):
        k_geometric_local = compute_geometric_stiffness_matrix(element, forces_moments)
        T = element.transformation_matrix()
        k_geometric_global = T.T @ k_geometric_local @ T

        # Get DOF indices for node1 and node2
        dofs_node1 = [element.node1.id * 6 + dof for dof in element.node1.dofs]
        dofs_node2 = [element.node2.id * 6 + dof for dof in element.node2.dofs]
        dofs = dofs_node1 + dofs_node2

        # Add to global geometric stiffness matrix
        for i, dof_i in enumerate(dofs):
            for j, dof_j in enumerate(dofs):
                K_geometric_global[dof_i, dof_j] += k_geometric_global[i, j]

    return K_geometric_global

def elastic_critical_load_analysis(structure, displacements):
    """
    Perform elastic critical load analysis.
    """
    # Compute local forces and moments for each element
    local_forces_moments = []
    for element in structure.elements:
        local_forces_moments.append(compute_local_forces_moments(element, displacements))

    # Assemble global geometric stiffness matrix
    K_geometric_global = assemble_global_geometric_stiffness_matrix(structure, local_forces_moments)

    # Assemble global stiffness matrix
    K_global = structure.assemble_global_stiffness_matrix()

    # Solve the eigenvalue problem
    eigenvalues, eigenvectors = eig(K_global, -K_geometric_global)

    # Find the smallest positive eigenvalue (critical load factor)
    critical_load_factor = np.min(eigenvalues[eigenvalues > 0])

    # Extract the corresponding buckling mode
    buckling_mode_index = np.argmin(eigenvalues[eigenvalues > 0])
    buckling_mode = eigenvectors[:, buckling_mode_index]

    return critical_load_factor, buckling_mode