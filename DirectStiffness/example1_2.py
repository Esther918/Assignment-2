from DirectStiffness import Node, BeamElement3D, DirectStiffness
import numpy as np

structure = DirectStiffness()
# Define nodes (node_index, x, y, z)
structure.add_node(Node(0, 0, 0, 0))  
structure.add_node(Node(1, -5, 1, 10))  
structure.add_node(Node(2, -1, 5, 13))  
structure.add_node(Node(3, -3, 7, 11))
structure.add_node(Node(4, 6, 9, 5))    

# Set boundary conditions (True = fixed)
structure.nodes[0].set_boundary_condition([False, False, True, False, False, False])  
structure.nodes[3].set_boundary_condition([True, True, True, True, True, True])  
structure.nodes[4].set_boundary_condition([True, True, True, False, False, False])

# Define properties
E = 500
nu = 0.3
r = 1
A = np.pi * r**2
Iy = np.pi * r**4 / 4
Iz = np.pi * r**4 / 4
Ip = np.pi * r**4 / 2
J = np.pi * r**4 / 2
localz = None

# Define the beam element (element name, node_i, node_j, E, nu, A, Iy, Iz, J, nodes, localz):
structure.add_element(BeamElement3D(0, 0, 1, E, nu, A, Iy, Iz, J, structure.nodes, localz))
structure.add_element(BeamElement3D(1, 1, 2, E, nu, A, Iy, Iz, J, structure.nodes, localz))
structure.add_element(BeamElement3D(2, 2, 3, E, nu, A, Iy, Iz, J, structure.nodes, localz))
structure.add_element(BeamElement3D(3, 2, 4, E, nu, A, Iy, Iz, J, structure.nodes, localz))

# Apply forces at nodes
structure.nodes[1].apply_force([0.05, 0.05, -0.1, 0, 0, 0])  
structure.nodes[2].apply_force([0, 0, 0, -0.1, -0.1, 0.3])  

# Displacements and reaction forces output
displacements, reactions = structure.solve()
print("Displacements:")
for i, d in enumerate(displacements): 
    print(f"Node {i}: {d}")
    
print("\nReaction Forces and Rotations:")
for node_id, node in structure.nodes.items():
    print(f"Node {node_id}: {reactions[node_id]}")
    