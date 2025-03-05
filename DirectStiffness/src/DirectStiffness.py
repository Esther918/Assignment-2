import numpy as np
from math_utils import (
    local_elastic_stiffness_matrix_3D_beam,
    rotation_matrix_3D,
    transformation_matrix_3D,
)

class Node:
    def __init__(self, id, x, y, z):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.dofs = [0, 1, 2, 3, 4, 5] 
        self.forces = np.zeros(6)  # [Fx, Fy, Fz, Mx, My, Mz]
        self.boundary_conditions = [False] * 6  # [Ux, Uy, Uz, Rx, Ry, Rz]

    def apply_force(self, forces):
        self.forces = np.array(forces)

    def set_boundary_condition(self, conditions):
        self.boundary_conditions = conditions

class BeamElement3D:
    def __init__(self, id, node1, node2, E, nu, A, Iy, Iz, J, nodes):
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
        return local_elastic_stiffness_matrix_3D_beam(
            self.E, self.nu, self.A, self.L, self.Iy, self.Iz, self.J
        )

    def transformation_matrix(self):
        gamma = rotation_matrix_3D(
            self.node1.x, self.node1.y, self.node1.z,
            self.node2.x, self.node2.y, self.node2.z
        )
        return transformation_matrix_3D(gamma)

class DirectStiffness:
    def __init__(self):
        self.nodes = {}
        self.elements = []

    def add_node(self, node):
        self.nodes[node.id] = node

    def add_element(self, element):
        self.elements.append(element)

    def assemble_global_stiffness_matrix(self):
        num_nodes = len(self.nodes)
        num_dofs = num_nodes * 6
        K_global = np.zeros((num_dofs, num_dofs))

        for element in self.elements:
            k_local = element.local_stiffness_matrix()
            T = element.transformation_matrix()
            k_global = T.T @ k_local @ T

            dofs_node1 = [element.node1.id * 6 + dof for dof in element.node1.dofs]
            dofs_node2 = [element.node2.id * 6 + dof for dof in element.node2.dofs]
            dofs = dofs_node1 + dofs_node2

            for i, dof_i in enumerate(dofs):
                for j, dof_j in enumerate(dofs):
                    K_global[dof_i, dof_j] += k_global[i, j]

        return K_global

    def assemble_force_vector(self):
        num_nodes = len(self.nodes)
        F = np.zeros(num_nodes * 6)

        for node in self.nodes.values():
            dofs = [node.id * 6 + dof for dof in node.dofs]
            F[dofs] = node.forces

        return F

    def apply_boundary_conditions(self, K_global, F):
        constrained_dofs = []
        for node in self.nodes.values():
            for dof, is_fixed in enumerate(node.boundary_conditions):
                if is_fixed:
                    constrained_dofs.append(node.id * 6 + dof)

        free_dofs = sorted(set(range(K_global.shape[0])) - set(constrained_dofs))
        supported_dofs = sorted(constrained_dofs)

        dof_order = free_dofs + supported_dofs
        K_copy = K_global[dof_order, :][:, dof_order]
        F_copy = F[dof_order]

        m = len(free_dofs)
        K_ff = K_copy[:m, :m]
        K_fs = K_copy[:m, m:]
        K_sf = K_copy[m:, :m]
        K_ss = K_copy[m:, m:]

        F_f = F_copy[:m]
        F_s = F_copy[m:]

        return K_ff, K_fs, K_sf, K_ss, F_f, F_s, free_dofs

    def solve(self):
        K_global = self.assemble_global_stiffness_matrix()
        F = self.assemble_force_vector()
        K_ff, K_fs, K_sf, K_ss, F_f, F_s, free_dofs = self.apply_boundary_conditions(K_global, F)

        U_f = np.linalg.solve(K_ff, F_f)
        F_s = K_sf @ U_f
        
        # Solve for displacement
        displacements = np.zeros(len(self.nodes) * 6)
        displacements[free_dofs] = U_f
        displacements = displacements.reshape(-1, 6)
        
        # Solve for reaction force
        num_nodes = len(self.nodes)
        reaction_full = np.zeros((num_nodes, 6))
        constrained_dofs = []
        for node in self.nodes.values():
            for dof, is_fixed in enumerate(node.boundary_conditions):
                if is_fixed:
                    constrained_dofs.append(node.id * 6 + dof)
        
        reaction_flat = np.zeros(num_nodes * 6)
        for i, dof in enumerate(constrained_dofs):
            if i < len(F_s):
                reaction_flat[dof] = F_s[i]
        reaction_full = reaction_flat.reshape(num_nodes, 6)

        return displacements, reaction_full