from DirectStiffness import Node, BeamElement3D, DirectStiffness
from ElasticCriticalLoad import ElasticCriticalLoad
from PostProcess import plot_deformed_shape
import numpy as np

structure = DirectStiffness()
x0, y0, z0 = 0, 0, 0
x1, y1, z1 = 25, 50, 37
# Define nodes (node_index, x, y, z)
structure.add_node(Node(0, 0, 0, 0))  
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

# Euler buckling analytical
L = np.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)
P = 1

# Define the beam element (element name, node_i, node_j, E, nu, A, Iy, Iz, J, nodes):
structure.add_element(BeamElement3D(0, 0, 1, E, nu, A, Iy, Iz, J, structure.nodes))
structure.add_element(BeamElement3D(1, 1, 2, E, nu, A, Iy, Iz, J, structure.nodes))
structure.add_element(BeamElement3D(2, 2, 3, E, nu, A, Iy, Iz, J, structure.nodes))
structure.add_element(BeamElement3D(3, 3, 4, E, nu, A, Iy, Iz, J, structure.nodes))
structure.add_element(BeamElement3D(4, 4, 5, E, nu, A, Iy, Iz, J, structure.nodes))
structure.add_element(BeamElement3D(5, 5, 6, E, nu, A, Iy, Iz, J, structure.nodes))

# Apply forces at nodes
Fx = -1 * P * (x1 - x0)/L
Fy = -1 * P * (y1 - y0)/L
Fz = -1 * P * (z1 - z0)/L

structure.nodes[6].apply_force([Fx, Fy, Fz, 0, 0, 0])  

# Displacements and reaction forces output
displacements, reactions = structure.solve()
print("Displacements:")
for i, d in enumerate(displacements): 
    print(f"Node {i}: {d}")
    
print("\nReaction Forces and Rotations:")
for node_id, node in structure.nodes.items():
    print(f"Node {node_id}: {reactions[node_id]}")
    

# Elastic critical load output
ecl = ElasticCriticalLoad(structure)
critical_load_factor, buckling_mode = ecl.elastic_critical_load_analysis(displacements)
print(f"Critical Load Factor: {critical_load_factor}")
# print("Buckling Mode:\n", buckling_mode)

# # Plot
# plot_deformed_shape(structure, displacements, scale=10)