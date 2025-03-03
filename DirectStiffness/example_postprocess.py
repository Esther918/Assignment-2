import numpy as np
import matplotlib.pyplot as plt
from DirectStiffness import Node, BeamElement3D, Function
from PostProcess import compute_local_forces_moments, plot_local_forces_moments, plot_deformed_shape


# Define Nodes (node_index, x, y, z)
node0 = Node(0, 0, 0, 0)
node1 = Node(1, -5, 1, 10)
node2 = Node(2, -1, 5, 13)
node3 = Node(3, -3, 7, 11)
node4 = Node(4, 6, 9, 5)

# Apply force at Node 1 and momentum at Node 2
node1.apply_force([0.05, 0.05, -0.1, 0, 0, 0])  # [Fx, Fy, Fz, Mx, My, Mz]
node2.apply_force([0, 0, 0, -0.1, -0.1, 0.3])  # [Fx, Fy, Fz, Mx, My, Mz]

# Define properties
r = 1
E = 500  
nu = 0.3 
A = np.pi * r**2
Iy = np.pi * r**4 / 4
Iz = np.pi * r**4 / 4
J = np.pi * r**4 / 2

# Define the beam elements
nodes = {0: node0, 1: node1, 2: node2, 3: node3, 4: node4}  
E0 = BeamElement3D(0, 0, 1, E, nu, A, Iy, Iz, J, nodes) 
E1 = BeamElement3D(1, 1, 2, E, nu, A, Iy, Iz, J, nodes)
E2 = BeamElement3D(2, 2, 3, E, nu, A, Iy, Iz, J, nodes)
E3 = BeamElement3D(3, 2, 4, E, nu, A, Iy, Iz, J, nodes)

# Fix Node 0, Node 3, Node 4
node0.set_boundary_condition([False, False, True, False, False, False]) # Fixed in z
node3.set_boundary_condition([True, True, True, True, True, True]) # Fixed
node4.set_boundary_condition([True, True, True, False, False, False]) # Pinned

# Create the structure and add nodes and elements
structure = Function()
structure.add_node(node0)
structure.add_node(node1)
structure.add_node(node2)
structure.add_node(node3)
structure.add_node(node4)
structure.add_element(E0)
structure.add_element(E1)
structure.add_element(E2)
structure.add_element(E3)

# Solve displacements and reactions
num_nodes = len(structure.nodes)
displacements = np.zeros(num_nodes * 6) 
displacements, reactions = structure.solve()

# # Debug: Print displacements
# print(f"displacements shape: {displacements.shape}")
# print(f"displacements: {displacements}")

# Compute local forces and moments for each element (show results)
for element in structure.elements:
    local_forces_moments = compute_local_forces_moments(element, displacements)
    print(f"Element {element.id} local forces and moments:\n{local_forces_moments}")

# Plot local forces and moments for each element
for element in structure.elements:
    local_forces_moments = compute_local_forces_moments(element, displacements)
    plot_local_forces_moments(element, local_forces_moments)

# Plot the deformed shape of the structure
plot_deformed_shape(structure, displacements, scale=10)