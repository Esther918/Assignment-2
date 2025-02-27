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
    def __init__(self, node_i, node_j, E: float, nu: float, A: float, Iy: float, Iz: float, J: float, nodes):
        """
        E: Young's modulus
        nu: Poisson's ratio
        A: Cross-sectional area
        L: Length of the beam
        Iy: Second moment of area about y-axis
        Iz: Second moment of area about z-axis
        J: Torsional moment of inertia
        """
        self.node_i = node_i 
        self.node_j = node_j  
        self.E = E
        self.nu = nu
        self.A = A
        self.Iy = Iy
        self.Iz = Iz
        self.J = J
        # Compute length
        coord_i = np.array(nodes[node_i].coordinates)
        coord_j = np.array(nodes[node_j].coordinates)
        self.L = np.linalg.norm(coord_j - coord_i)
        
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

    def add_element(self, beam: BeamElement3D):
        """Add an element to the structure."""
        self.elements.append(beam)

    def assemble_global_stiffness(self):
        """Construct the global stiffness matrix."""
        num_dof = len(self.nodes) * 6
        K_global = np.zeros((num_dof, num_dof))

        for element in self.elements: 
            # Compute the local stiffness matrix
            k_local = element.local_elastic_stiffness_matrix()

            # Check if k_local is 12x12
            if k_local.shape != (12, 12):
                raise ValueError(f"Local stiffness matrix must be 12x12, but got {k_local.shape}")

            node_i = self.nodes[0]  # First node of the element
            node_j = self.nodes[1]  # Second node of the element
            x1, y1, z1 = node_i.coordinates
            x2, y2, z2 = node_j.coordinates
            gamma = element.rotation_matrix_3D(x1, y1, z1, x2, y2, z2)
            T = element.transformation_matrix_3D(gamma)

            # Check if T is 12x12
            if T.shape != (12, 12):
                raise ValueError(f"Transformation matrix must be 12x12, but got {T.shape}")

            # Transform the local stiffness matrix to the global coordinate system
            k_global = T.T @ k_local @ T

            # Assemble the global stiffness matrix
            dof_i = self.nodes.index(node_i) * 6  # Use node index in self.nodes
            dof_j = self.nodes.index(node_j) * 6  # Use node index in self.nodes

            # Check if the indices are within bounds
            if dof_i + 6 > num_dof or dof_j + 6 > num_dof:
                raise ValueError("Indices out of bounds for global stiffness matrix")

            # Assemble the 12x12 k_global into K_global
            K_global[dof_i:dof_i+6, dof_i:dof_i+6] += k_global[:6, :6]
            K_global[dof_i:dof_i+6, dof_j:dof_j+6] += k_global[:6, 6:]
            K_global[dof_j:dof_j+6, dof_i:dof_i+6] += k_global[6:, :6]
            K_global[dof_j:dof_j+6, dof_j:dof_j+6] += k_global[6:, 6:]

        return K_global
    
    def apply_boundary_conditions(self, K, F):
        """Modifies the stiffness matrix and force vector for boundary conditions by reducing to a 12x12 matrix."""
        selected_indices = list(range(0, 12)) 

        # Reduce K and F to the selected DOFs
        K_reduced = K[np.ix_(selected_indices, selected_indices)]  # Reduce K to 12x12
        F_reduced = F[list(range(6, 18))]  # Reduce F to 12x1

        return K_reduced, F_reduced

    def solve(self):
        """Solves for displacements and reactions."""
        K_global = self.assemble_global_stiffness()
        F_global = np.concatenate([node.forces for node in self.nodes])
        K_reduced, F_reduced = self.apply_boundary_conditions(K_global, F_global)
        # Debug: Check if the matrix is singular using condition number
        cond_number = np.linalg.cond(K_reduced)
        # Debug:
        print("K_reduced:")
        print(K_reduced)
        print("F_reduced:")
        print(F_reduced)
        
        print(f"Condition number of the stiffness matrix: {cond_number}")

        if cond_number > 1e18:  # Threshold for singularity
            raise ValueError("The stiffness matrix is singular. Check boundary conditions and structure stability.")

        # Solve for displacements
        displacements = np.zeros(K_global.shape[0])
        value_array = np.array(displacements)
        selected_indices = list(range(0, len(value_array)))
        displacements[selected_indices] = displacements
        
        # Reactions
        reactions = np.dot(K_global, displacements) - F_global

        return displacements, reactions