from DirectStiffness import Node, BeamElement3D, DirectStiffness
from ElasticCriticalLoad import ElasticCriticalLoad
from PostProcess import plot_deformed_shape
import numpy as np

structure = DirectStiffness()
x0, y0, z0 = 0, 0, 0
x1, y1, z1 = 18, 56, 44
# Define nodes (node_index, x, y, z)
structure.add_node(Node(0, x0, y0, z0))  
structure.add_node(Node(1, x1/6, y1/6, z1/6))  
structure.add_node(Node(2, 2 * (x1/6), 2 * (y1/6), 2 * (z1/6)))  
structure.add_node(Node(3, 3 * (x1/6), 3 * (y1/6), 3 * (z1/6)))  
structure.add_node(Node(4, 4 * (x1/6), 4 * (y1/6), 4 * (z1/6)))  
structure.add_node(Node(5, 5 * (x1/6), 5 * (y1/6), 5 * (z1/6)))  
structure.add_node(Node(6, x1, y1, z1))  

# Set boundary conditions (True = fixed)
structure.nodes[0].set_boundary_condition([True, True, True, True, True, True])  
structure.nodes[6].set_boundary_condition([False, False, False, False, False, False])

# Define properties
E = 10000
nu = 0.3
r = 1
A = np.pi * r ** 2
Iy = np.pi * r ** 4 / 4
Iz = np.pi * r ** 4 / 4
Ip = np.pi * r ** 4 / 2
J = np.pi * r ** 4 / 2 
localz = None

# Define the beam element (element name, node_i, node_j, E, nu, A, Iy, Iz, J, nodes, localz):
structure.add_element(BeamElement3D(0, 0, 1, E, nu, A, Iy, Iz, J, structure.nodes, localz))
structure.add_element(BeamElement3D(1, 1, 2, E, nu, A, Iy, Iz, J, structure.nodes, localz))
structure.add_element(BeamElement3D(2, 2, 3, E, nu, A, Iy, Iz, J, structure.nodes, localz))
structure.add_element(BeamElement3D(3, 3, 4, E, nu, A, Iy, Iz, J, structure.nodes, localz))
structure.add_element(BeamElement3D(4, 4, 5, E, nu, A, Iy, Iz, J, structure.nodes, localz))
structure.add_element(BeamElement3D(5, 5, 6, E, nu, A, Iy, Iz, J, structure.nodes, localz))

# Apply forces at nodes
Fx = 0.05
Fy = -0.1
Fz = 0.23
Mx = 0.1
My = -0.025
Mz = -0.08
structure.nodes[6].apply_force([Fx, Fy, Fz, Mx, My, Mz])  

# Displacements and reaction forces output
displacements, reactions = structure.solve()
print("Displacements:")
for i, d in enumerate(displacements): 
    print(f"Node {i}: {d}")
    
print("\nReaction Forces and Rotations:")
for node_id, node in structure.nodes.items():
    print(f"Node {node_id}: {reactions[node_id]}")
    

# # Elastic critical load output
# ecl = ElasticCriticalLoad(structure)
# critical_load_factor, buckling_mode = ecl.elastic_critical_load_analysis(displacements)
# print(f"Critical Load Factor: {critical_load_factor}")
# print("Buckling Mode:\n", buckling_mode)

# # Plot
# plot_deformed_shape(structure, displacements, scale=10)