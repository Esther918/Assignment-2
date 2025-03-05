import numpy as np
from scipy.linalg import eig
from math_utils import local_geometric_stiffness_matrix_3D_beam

class ElasticCriticalLoad:
    def __init__(self, structure):
        self.structure = structure

    def compute_local_forces_moments(self, element, displacements):
        displacements_flat = displacements.flatten()
        dofs_node1 = [element.node1.id * 6 + dof for dof in element.node1.dofs]
        dofs_node2 = [element.node2.id * 6 + dof for dof in element.node2.dofs]
        dofs = dofs_node1 + dofs_node2

        global_displacements = displacements_flat[dofs]
        T = element.transformation_matrix()
        local_displacements = T @ global_displacements

        k_local = element.local_stiffness_matrix()
        local_forces_moments = k_local @ local_displacements

        return local_forces_moments

    def compute_geometric_stiffness_matrix(self, element, local_forces_moments):
        """Compute the geometric stiffness matrix using math_utils implementation."""
        Fx1, Fy1, Fz1, Mx1, My1, Mz1, Fx2, Fy2, Fz2, Mx2, My2, Mz2 = local_forces_moments
        L = element.L
        A = element.A
        I_rho = element.J 

        k_geometric_local = local_geometric_stiffness_matrix_3D_beam(
            L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2
        )

        return k_geometric_local

    def assemble_global_geometric_stiffness_matrix(self, displacements):
        num_nodes = len(self.structure.nodes)
        num_dofs = num_nodes * 6
        K_geometric_global = np.zeros((num_dofs, num_dofs))

        local_forces_moments = [self.compute_local_forces_moments(element, displacements) 
                              for element in self.structure.elements]

        for element, forces_moments in zip(self.structure.elements, local_forces_moments):
            k_geometric_local = self.compute_geometric_stiffness_matrix(element, forces_moments)
            T = element.transformation_matrix()
            k_geometric_global = T.T @ k_geometric_local @ T

            dofs_node1 = [element.node1.id * 6 + dof for dof in element.node1.dofs]
            dofs_node2 = [element.node2.id * 6 + dof for dof in element.node2.dofs]
            dofs = dofs_node1 + dofs_node2

            for i, dof_i in enumerate(dofs):
                for j, dof_j in enumerate(dofs):
                    K_geometric_global[dof_i, dof_j] += k_geometric_global[i, j]

        return K_geometric_global

    def elastic_critical_load_analysis(self, displacements):
        K_geometric_global = self.assemble_global_geometric_stiffness_matrix(displacements)
        K_global = self.structure.assemble_global_stiffness_matrix()

        try:
            eigenvalues, eigenvectors = eig(K_global, -K_geometric_global)
            positive_eigenvalues = eigenvalues[eigenvalues > 0].real
            if len(positive_eigenvalues) == 0:
                raise ValueError("No positive eigenvalues found, check model stability.")
            
            critical_load_factor = np.min(positive_eigenvalues)
            buckling_mode_index = np.argmin(eigenvalues[eigenvalues > 0])
            buckling_mode = eigenvectors[:, buckling_mode_index].real

            return critical_load_factor, buckling_mode
        except np.linalg.LinAlgError as e:
            print(f"Linear algebra error during eig computation: {e}")
            raise