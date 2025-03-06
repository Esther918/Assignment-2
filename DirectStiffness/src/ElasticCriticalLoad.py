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

            dofs_node1 = [element.node1.id * 6 + dof for dof in element.node1.dofs]
            dofs_node2 = [element.node2.id * 6 + dof for dof in element.node2.dofs]
            dofs = dofs_node1 + dofs_node2

            k_geometric_global = T.T @ k_geometric_local @ T

            for i, dof_i in enumerate(dofs):
                for j, dof_j in enumerate(dofs):
                    K_geometric_global[dof_i, dof_j] += k_geometric_global[i, j]

        return K_geometric_global
    
    def apply_boundary_conditions(self, K):
        """Apply boundary conditions to a stiffness matrix."""
        constrained_dofs = []
        for node in self.structure.nodes.values():
            for dof, is_fixed in enumerate(node.boundary_conditions):
                if is_fixed:
                    constrained_dofs.append(node.id * 6 + dof)

        free_dofs = sorted(set(range(K.shape[0])) - set(constrained_dofs))
        K_bc = K[np.ix_(free_dofs, free_dofs)]

        return K_bc, free_dofs

    def elastic_critical_load_analysis(self, displacements):
        K_geometric_global = self.assemble_global_geometric_stiffness_matrix(displacements)
        K_global = self.structure.assemble_global_stiffness_matrix()

        K_geometric_global_bc, free_dofs = self.apply_boundary_conditions(K_geometric_global)
        K_global_bc, _ = self.apply_boundary_conditions(K_global)

        try:
            eigenvalues, eigenvectors = eig(K_global_bc, -K_geometric_global_bc)
            # print("Eigenvalues:", eigenvalues)  
            # print("Eigenvectors shape:", eigenvectors.shape)

            positive_eigenvalues = eigenvalues[eigenvalues > 0].real
            if len(positive_eigenvalues) == 0:
                raise ValueError("No positive eigenvalues found, check model stability.")

            critical_load_factor = np.min(positive_eigenvalues)
            buckling_mode_index = np.argmin(positive_eigenvalues)
            buckling_mode_bc = eigenvectors[:, buckling_mode_index].real

            buckling_mode = np.zeros(len(self.structure.nodes) * 6)
            for i, dof in enumerate(free_dofs):
                buckling_mode[dof] = buckling_mode_bc[i]

            buckling_mode_normalized = buckling_mode / np.max(np.abs(buckling_mode))

            for node_id, node in self.structure.nodes.items():
                for local_dof in range(6):
                    global_dof = node_id * 6 + local_dof
                    node.geo_disp[local_dof] = buckling_mode_normalized[global_dof]

            return critical_load_factor, buckling_mode_normalized

        except np.linalg.LinAlgError as e:
            print(f"Linear algebra error during eig computation: {e}")
            raise