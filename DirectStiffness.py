import numpy as np
from math_utils import (
    local_elastic_stiffness_matrix_3D_beam,
    rotation_matrix_3D,
    transformation_matrix_3D
)

class Node:
    def __init__(self, node_i, x, y, z):
        """
        node_i: node name.
        x, y, z: Coordinates of the node.
        """
        self.node_i = node_i
        self.coordinates = (x, y, z)
        self.forces = [0] * 6  # [Fx, Fy, Fz, Mx, My, Mz]
        self.boundary_conditions = [False] * 6  # [Ux, Uy, Uz, Rx, Ry, Rz] (False = free, True = constrained)

    def apply_force(self, force_vector):
        """
        Apply a force/moment to the node.
        force_vector: List of 6 elements [Fx, Fy, Fz, Mx, My, Mz].
        """
        if len(force_vector) != 6:
            raise ValueError("Force vector must have exactly 6 elements.")
        self.forces = force_vector

    def set_boundary_condition(self, bc_vector):
        """
        Apply boundary conditions to the node.
        bc_vector: List of 6 boolean elements [Ux, Uy, Uz, Rx, Ry, Rz].
        """
        if len(bc_vector) != 6:
            raise ValueError("Boundary condition vector must have exactly 6 elements.")
        self.boundary_conditions = bc_vector

class BeamElement3D:
    def __init__(self, E: float, nu: float, A: float, L: float, Iy: float, Iz: float, J: float):
        """
        E: Young's modulus
        nu: Poisson's ratio
        A: Cross-sectional area
        L: Length of the beam
        Iy: Second moment of area about y-axis
        Iz: Second moment of area about z-axis
        J: Torsional moment of inertia
        """
        self.E = E
        self.nu = nu
        self.A = A
        self.L = L
        self.Iy = Iy
        self.Iz = Iz
        self.J = J

    def local_elastic_stiffness_matrix(self) -> np.ndarray:
        """Computes the local element elastic stiffness matrix."""
        return local_elastic_stiffness_matrix_3D_beam(self.E, self.nu, self.A, self.L, self.Iy, self.Iz, self.J)

    @staticmethod
    def rotation_matrix_3D(x1, y1, z1, x2, y2, z2, v_temp=None):
        """Computes the 3D rotation matrix."""
        return rotation_matrix_3D(x1, y1, z1, x2, y2, z2, v_temp)

    @staticmethod
    def transformation_matrix_3D(gamma):
        """Computes the 12x12 transformation matrix."""
        return transformation_matrix_3D(gamma)
    
class Function:
    def __init__(self):
        """Initialize an empty structure."""
        self.nodes = []
        self.elements = []

    def add_node(self, node):
        """Add a node to the structure."""
        self.nodes.append(node)

    def add_element(self, element):
        """Add an element to the structure."""
        self.elements.append(element)

    def assemble_global_stiffness(self):
        """Construct the global stiffness matrix."""
        num_dof = len(self.nodes) * 6  
        K_global = np.zeros((num_dof, num_dof))

        for element in self.elements:
            k_local = element.local_elastic_stiffness_matrix()        
        return K_global

    def apply_boundary_conditions(self, K, F):
        """Modifies the stiffness matrix and force vector for boundary conditions."""
        for node in self.nodes:
            for dof, is_fixed in enumerate(node.boundary_conditions):
                if is_fixed:
                    index = node.node_i * 6 + dof
                    if index >= K.shape[0]:  # Check if it is in the bounds.
                        continue  # Skip
                    K[index, :] = 0
                    K[:, index] = 0
                    K[index, index] = 1
                    F[index] = 0
        return K, F

    def solve(self):
        """Solves for displacements and reactions."""
        K_global = self.assemble_global_stiffness()
        F_global = np.concatenate([node.forces for node in self.nodes])
        K_mod, F_mod = self.apply_boundary_conditions(K_global, F_global)
        displacements = np.linalg.solve(K_mod, F_mod)

        # Reactions
        reactions = np.dot(K_global, displacements) - F_global

        return displacements, reactions
