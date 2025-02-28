import numpy as np
from math_utils import (
    local_elastic_stiffness_matrix_3D_beam,
    rotation_matrix_3D,
    transformation_matrix_3D,
)

class Node:
    def __init__(self, id, x, y, z):
        """
        id: node name.
        x, y, z: Coordinates of the node.
        """
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.dofs = [0, 1, 2, 3, 4, 5]  # 6 DOFs per node
        self.forces = np.zeros(6)  # [Fx, Fy, Fz, Mx, My, Mz]
        self.boundary_conditions = [False] * 6  # [Ux, Uy, Uz, Rx, Ry, Rz] (False = free, True = constrained)

    def apply_force(self, forces):
        """
        Apply forces and moments to the node.
        force_vector: List of 6 elements [Fx, Fy, Fz, Mx, My, Mz].
        """
        self.forces = np.array(forces)

    def set_boundary_condition(self, conditions):
        """
        Apply boundary conditions to the node.
        bc_vector: List of 6 boolean elements [Ux, Uy, Uz, Rx, Ry, Rz].
        """
        self.boundary_conditions = conditions

class BeamElement3D:
    def __init__(self, id, node1, node2, E, nu, A, Iy, Iz, J, nodes):
        """
        E: Young's modulus
        nu: Poisson's ratio
        A: Cross-sectional area
        L: Length of the beam
        Iy: Second moment of area about y-axis
        Iz: Second moment of area about z-axis
        J: Torsional moment of inertia
        """
        self.id = id
        self.node1 = nodes[node1]
        self.node2 = nodes[node2]
        self.E = E
        self.nu = nu
        self.A = A
        self.Iy = Iy
        self.Iz = Iz
        self.J = J
        self.L = np.linalg.norm([
            self.node2.x - self.node1.x,
            self.node2.y - self.node1.y,
            self.node2.z - self.node1.z,
        ])

    def local_stiffness_matrix(self):
        """Compute the local stiffness matrix for the element."""
        return local_elastic_stiffness_matrix_3D_beam(
            self.E, self.nu, self.A, self.L, self.Iy, self.Iz, self.J
        )

    def transformation_matrix(self):
        """Compute the transformation matrix for the element."""
        gamma = rotation_matrix_3D(
            self.node1.x, self.node1.y, self.node1.z,
            self.node2.x, self.node2.y, self.node2.z
        )
        return transformation_matrix_3D(gamma)

class Function:
    def __init__(self):
        self.nodes = {}
        self.elements = []

    def add_node(self, node):
        """Add a node to the structure."""
        self.nodes[node.id] = node

    def add_element(self, element):
        """Add an element to the structure."""
        self.elements.append(element)

    def assemble_global_stiffness_matrix(self):
        """Assemble the global stiffness matrix."""
        num_nodes = len(self.nodes)
        num_dofs = num_nodes * 6
        K_global = np.zeros((num_dofs, num_dofs))

        for element in self.elements:
            k_local = element.local_stiffness_matrix()
            T = element.transformation_matrix()
            k_global = T.T @ k_local @ T

            # Get DOF indices for node1 and node2
            dofs_node1 = [element.node1.id * 6 + dof for dof in element.node1.dofs]
            dofs_node2 = [element.node2.id * 6 + dof for dof in element.node2.dofs]
            dofs = dofs_node1 + dofs_node2

            # Add to global stiffness matrix
            for i, dof_i in enumerate(dofs):
                for j, dof_j in enumerate(dofs):
                    K_global[dof_i, dof_j] += k_global[i, j]

        return K_global

    def assemble_force_vector(self):
        """Assemble the global force vector."""
        num_nodes = len(self.nodes)
        F = np.zeros(num_nodes * 6)

        for node in self.nodes.values():
            dofs = [node.id * 6 + dof for dof in node.dofs]
            F[dofs] = node.forces

        return F

    def apply_boundary_conditions(self, K_global, F):
        """Apply boundary conditions to the global stiffness matrix and force vector."""
        constrained_dofs = []

        for node in self.nodes.values():
            for dof, is_fixed in enumerate(node.boundary_conditions):
                if is_fixed:
                    constrained_dofs.append(node.id * 6 + dof)

        # Free DOFs are all DOFs not in constrained_dofs
        free_dofs = sorted(set(range(K_global.shape[0])) - set(constrained_dofs))
        supported_dofs = sorted(constrained_dofs)

        # Copy the global stiffness matrix and force vector
        dof_order = free_dofs + supported_dofs
        K_copy = K_global[dof_order, :][:, dof_order]
        F_copy = F[dof_order]

        # Partition the reordered stiffness matrix and force vector
        m = len(free_dofs)  
        K_ff = K_copy[:m, :m]
        K_fs = K_copy[:m, m:]
        K_sf = K_copy[m:, :m]
        K_ss = K_copy[m:, m:]

        F_f = F_copy[:m]
        F_s = F_copy[m:]

        return K_ff, K_fs, K_sf, K_ss, F_f, F_s

    def solve(self):
        """Solve for displacements and reaction forces."""
        # Assemble global stiffness matrix and force vector
        K_global = self.assemble_global_stiffness_matrix()
        F = self.assemble_force_vector()

        # Apply boundary conditions and partition the global stiffness matrix
        K_ff, K_fs, K_sf, K_ss, F_f, F_s = self.apply_boundary_conditions(K_global, F)

        # Solve for free displacements
        U_f = np.linalg.solve(K_ff, F_f)

        # Solve for support reactions
        F_s = K_sf @ U_f

        # Combine free and supported displacements
        num_nodes = len(self.nodes)
        displacements = np.zeros(num_nodes * 6)
        
        # Get free DOFs
        constrained_dofs = []
        for node in self.nodes.values():
            for dof, is_fixed in enumerate(node.boundary_conditions):
                if is_fixed:
                    constrained_dofs.append(node.id * 6 + dof)
        free_dofs = sorted(set(range(K_global.shape[0])) - set(constrained_dofs))

        # Assign free displacements
        displacements[list(free_dofs)] = U_f
        
        # Displacements output
        displacements = displacements.reshape(-1, 6)

        # Reshape reactions for output
        num_reaction_nodes = len(constrained_dofs) // 6  
        if len(constrained_dofs) % 6 != 0:
            num_reaction_nodes += 1  

        # Make F_s length a multiple of 6
        if len(F_s) % 6 != 0:
            padding_length = 6 - (len(F_s) % 6)
            F_s = np.pad(F_s, (0, padding_length), mode='constant')

        # Reaction force output
        reactions = F_s.reshape(-1, 6)

        return displacements, reactions