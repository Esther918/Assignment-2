import numpy as np
import math
from DirectStiffness import Node, BeamElement3D, DirectStiffness

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
A = math.pi * r**2
Iy = math.pi * r**4 / 4
Iz = math.pi * r**4 / 4
J = math.pi * r**4 / 2

# Define the beam elements (element name, id, node1, node2, E, nu, A, Iy, Iz, J, nodes):
nodes = {0: node0, 1: node1, 2: node2, 3: node3, 4: node4}  
E0 = BeamElement3D(0, 0, 1, E, nu, A, Iy, Iz, J, nodes) 
E1 = BeamElement3D(1, 1, 2, E, nu, A, Iy, Iz, J, nodes)
E2 = BeamElement3D(2, 2, 3, E, nu, A, Iy, Iz, J, nodes)
E3 = BeamElement3D(3, 2, 4, E, nu, A, Iy, Iz, J, nodes)

# Fix Node 0, Node 3, Node 4 (True = fixed)
node0.set_boundary_condition([False, False, True, False, False, False]) # Fixed in z
node3.set_boundary_condition([True, True, True, True, True, True]) # Fully Fixed
node4.set_boundary_condition([True, True, True, False, False, False]) # Pinned

# Create the structure and add nodes and elements
structure = DirectStiffness()
structure.add_node(node0)
structure.add_node(node1)
structure.add_node(node2)
structure.add_node(node3)
structure.add_node(node4)
structure.add_element(E0)
structure.add_element(E1)
structure.add_element(E2)
structure.add_element(E3)

# Solve for displacements and reactions
displacements, reactions = structure.solve()

# Displacements and reaction forces output
print("Displacements at Nodes:")
for i, d in enumerate(displacements.reshape(-1, 6)): 
    print(f"Node {i}: {d}")

print("\nReaction Forces and Rotations at Nodes:")
for i, r in enumerate(reactions.reshape(-1, 6)):
    print(f"Node {i}: {r}")