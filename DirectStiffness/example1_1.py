from DirectStiffness import Node, BeamElement3D, DirectStiffness
import numpy as np

structure = DirectStiffness()
# Define nodes (node_index, x, y, z)
structure.add_node(Node(0, 0, 0, 10))  
structure.add_node(Node(1, 15, 0, 10))  
structure.add_node(Node(2, 15, 0, 0))  

# Set boundary conditions (True = fixed)
structure.nodes[0].set_boundary_condition([True, True, True, True, True, True])  
structure.nodes[2].set_boundary_condition([True, True, True, False, False, False])

# Define properties
E = 1000
nu = 0.3
b = 0.5
h = 1
A = b * h
Iy = h * b**3 / 12
Iz = b * h**3 / 12
Ip = b * h / 12 * (b**2 + h**2)
J = 0.02861
localz_0 = [0, 0, 1]
localz_1 = [1, 0, 0]

# Define the beam element (element name, node_i, node_j, E, nu, A, Iy, Iz, J, nodes, localz):
structure.add_element(BeamElement3D(0, 0, 1, E, nu, A, Iy, Iz, J, structure.nodes, localz_0))
structure.add_element(BeamElement3D(1, 1, 2, E, nu, A, Iy, Iz, J, structure.nodes, localz_1))

# Apply forces at nodes
structure.nodes[1].apply_force([-0.05, 0.075, 0.1, -0.05, 0.1, -0.25])  

# Displacements and reaction forces output
displacements, reactions = structure.solve()
print("Displacements:")
for i, d in enumerate(displacements): 
    print(f"Node {i}: {d}")
    
print("\nReaction Forces and Rotations:")
for node_id, node in structure.nodes.items():
    print(f"Node {node_id}: {reactions[node_id]}")
    

