import numpy as np
from DirectStiffness import Node, BeamElement3D, DirectStiffness

# Define Nodes (node_index, x, y, z)
node0 = Node(0, 0, 0, 10.0)
node1 = Node(1, 30, 40, 0)

# Apply a 10 kN force at Node 1
node1.apply_force([-0.6, -0.2, 0, 0, 0, 0])  # [Fx, Fy, Fz, Mx, My, Mz]

# Define properties
r = 1
E = 1000  
nu = 0.3 
A = np.pi * r**2
Iy = np.pi * r**4 / 4
Iz = np.pi * r**4 / 4
J = np.pi * r**4 / 2
L = 50
P_analytical = np.pi**2 * E * Iz / (2 * L)**2

# Define the beam element (element name, id, node1, node2, E, nu, A, Iy, Iz, J, nodes):
nodes = {0: node0, 1: node1}  
E0 = BeamElement3D(0, 0, 1, E, nu, A, Iy, Iz, J, nodes) 

# Fix Node 0 and Node 2 (True = fixed)
node0.set_boundary_condition([True, True, True, True, True, True]) # Fixed

structure = DirectStiffness()
structure.add_node(node0)
structure.add_node(node1)
structure.add_element(E0)

displacements, reactions = structure.solve()

# Displacements and reaction forces output
print("Displacements at Nodes:")
for i, d in enumerate(displacements): 
    print(f"Node {i}: {d}")

print("\nReaction Forces and Rotations at Nodes:")
for i, r in enumerate(reactions):
    print(f"Node {i}: {r}")