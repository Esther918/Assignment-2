from DirectStiffness import Node, BeamElement3D, DirectStiffness
from ElasticCriticalLoad import ElasticCriticalLoad
from PostProcess import plot_deformed_shape
import numpy as np

structure = DirectStiffness()
# Define Nodes (node_index, x, y, z)
structure.add_node(Node(0, 0, 0, 0))
structure.add_node(Node(1, 10, 0, 0))  
structure.add_node(Node(2, 10, 20, 0))  
structure.add_node(Node(3, 0, 20, 0))  
structure.add_node(Node(4, 0, 0, 25))  
structure.add_node(Node(5, 10, 0, 25))  
structure.add_node(Node(6, 10, 20, 25))  
structure.add_node(Node(7, 0, 20, 25)) 

# Set boundary conditions (True = fixed)
structure.nodes[0].set_boundary_condition([True, True, True, True, True, True]) 
structure.nodes[1].set_boundary_condition([True, True, True, True, True, True])  
structure.nodes[2].set_boundary_condition([True, True, True, True, True, True])  
structure.nodes[3].set_boundary_condition([True, True, True, True, True, True])  

# Define properties
r = 0.5
E = 500  
nu = 0.3 
A = np.pi * r**2
Iy = np.pi * r**4 / 4
Iz = np.pi * r**4 / 4
J = np.pi * r**4 / 2
localz = None

# Define the beam elements (element name, id, node1, node2, E, nu, A, Iy, Iz, J, nodes, localz):
E0 = BeamElement3D(0, 0, 4, E, nu, A, Iy, Iz, J, structure.nodes, localz) 
E1 = BeamElement3D(1, 1, 5, E, nu, A, Iy, Iz, J, structure.nodes, localz)
E2 = BeamElement3D(2, 2, 6, E, nu, A, Iy, Iz, J, structure.nodes, localz)
E3 = BeamElement3D(3, 3, 7, E, nu, A, Iy, Iz, J, structure.nodes, localz)
E4 = BeamElement3D(4, 4, 5, E, nu, A, Iy, Iz, J, structure.nodes, localz)
E5 = BeamElement3D(5, 5, 6, E, nu, A, Iy, Iz, J, structure.nodes, localz)
E6 = BeamElement3D(6, 6, 7, E, nu, A, Iy, Iz, J, structure.nodes, localz)
E7 = BeamElement3D(7, 4, 7, E, nu, A, Iy, Iz, J, structure.nodes, localz)

# Apply force at Node 1 and momentum at Node 2
structure.nodes[4].apply_force([0, 0, -1, 0, 0, 0])  
structure.nodes[5].apply_force([0, 0, -1, 0, 0, 0])  
structure.nodes[6].apply_force([0, 0, -1, 0, 0, 0])  
structure.nodes[7].apply_force([0, 0, -1, 0, 0, 0])  

# Solve for displacements and reactions
displacements, reactions = structure.solve()

# Displacements and reaction forces output
print("Displacements at Nodes:")
for i, d in enumerate(displacements.reshape(-1, 6)): 
    print(f"Node {i}: {d}")

print("\nReaction Forces and Rotations at Nodes:")
for i, r in enumerate(reactions.reshape(-1, 6)):
    print(f"Node {i}: {r}")