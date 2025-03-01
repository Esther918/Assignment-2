import numpy as np
from DirectStiffness import Node, BeamElement3D, Function

# Define Nodes (node_index, x, y, z)
node0 = Node(0, 0, 0, 10.0)
node1 = Node(1, 15.0, 0, 10.0)
node2 = Node(2, 15.0, 0, 0)

# Apply a 10 kN force at Node 1
node1.apply_force([-0.05, 0.075, 0.1, -0.05, 0.1, -0.25])  # [Fx, Fy, Fz, Mx, My, Mz]
node2.apply_force([0, 0, 0, 0, 0, 0])  # [Fx, Fy, Fz, Mx, My, Mz]

# Define properties
b = 0.5
h = 1.0
E = 1000  
nu = 0.3 
A = b * h 
Iy = h * b**3 / 12
Iz = b * h**3 / 12
J = 0.02861

# Define the beam element (element name, id, node1, node2, E, nu, A, Iy, Iz, J, nodes):
nodes = {0: node0, 1: node1, 2: node2}  
E0 = BeamElement3D(0, 0, 1, E, nu, A, Iy, Iz, J, nodes) 
E1 = BeamElement3D(1, 1, 2, E, nu, A, Iy, Iz, J, nodes) 

# Fix Node 0 and Node 2 (True = fixed)
node0.set_boundary_condition([True, True, True, True, True, True]) # Fixed
node2.set_boundary_condition([True, True, True, False, False, False]) # Pinned

structure = Function()
structure.add_node(node0)
structure.add_node(node1)
structure.add_node(node2)
structure.add_element(E0)
structure.add_element(E1)

displacements, reactions = structure.solve()

# Displacements and reaction forces output
print("Displacements at Nodes:")
for i, d in enumerate(displacements): 
    print(f"Node {i}: {d}")

print("\nReaction Forces and Rotations at Nodes:")
for i, r in enumerate(reactions):
    print(f"Node {i}: {r}")