from DirectStiffness import Node, BeamElement3D, DirectStiffness
from ElasticCriticalLoad import ElasticCriticalLoad
from PostProcess import plot_deformed_shape
import numpy as np

structure = DirectStiffness()
x0, y0, z0 = 0, 0, 0
L1 = 11
L2 = 23
L3 = 15
L4 = 13

# Define nodes (node_index, x, y, z)
structure.add_node(Node(0, x0, y0, z0))  
structure.add_node(Node(1, x0 + L1, y0, z0))  
structure.add_node(Node(2, x0 + L1, y0 + L2, z0))  
structure.add_node(Node(3, x0, y0 + L2, z0))  
structure.add_node(Node(4, x0, y0, z0 + L3))  
structure.add_node(Node(5, x0 + L1, y0, z0 + L3))  
structure.add_node(Node(6, x0 + L1, y0 + L2, z0 + L3))
structure.add_node(Node(7, x0, y0 + L2, z0 + L3))  
structure.add_node(Node(8, x0, y0, z0 + L3 + L4))  
structure.add_node(Node(9, x0 + L1, y0, z0 + L3 + L4))  
structure.add_node(Node(10, x0 + L1, y0 + L2, z0 + L3 + L4))
structure.add_node(Node(11, x0, y0 + L2, z0 + L3 + L4))  

# Set boundary conditions (True = fixed)
structure.nodes[0].set_boundary_condition([True, True, True, True, True, True])  
structure.nodes[1].set_boundary_condition([True, True, True, True, True, True])  
structure.nodes[2].set_boundary_condition([True, True, True, True, True, True])  
structure.nodes[3].set_boundary_condition([True, True, True, True, True, True])  

# Define properties (A element)
E_a = 10000
nu_a = 0.3
r = 1
A_a = np.pi * r ** 2
Iy_a = np.pi * r ** 4 / 4
Iz_a = np.pi * r ** 4 / 4
Ip_a = np.pi * r ** 4 / 2
J_a = np.pi * r ** 4 / 2 
# Define properties (B element)
E_b = 50000
nu_b = 0.3
b = 0.5
h = 1
A_b = b * h
Iy_b = h * b**3 / 12
Iz_b = b * h**3 / 12
Ip_b = b * h / 12 * (b**2 + h**2)
J_b = 0.028610026041666667 

# Define the beam element (element name, node_i, node_j, E, nu, A, Iy, Iz, J, nodes):
# A element type
structure.add_element(BeamElement3D(0, 0, 4, E_a, nu_a, A_a, Iy_a, Iz_a, J_a, structure.nodes))
structure.add_element(BeamElement3D(1, 1, 5, E_a, nu_a, A_a, Iy_a, Iz_a, J_a, structure.nodes))
structure.add_element(BeamElement3D(2, 2, 6, E_a, nu_a, A_a, Iy_a, Iz_a, J_a, structure.nodes))
structure.add_element(BeamElement3D(3, 3, 7, E_a, nu_a, A_a, Iy_a, Iz_a, J_a, structure.nodes))
structure.add_element(BeamElement3D(4, 4, 8, E_a, nu_a, A_a, Iy_a, Iz_a, J_a, structure.nodes))
structure.add_element(BeamElement3D(5, 5, 9, E_a, nu_a, A_a, Iy_a, Iz_a, J_a, structure.nodes))
structure.add_element(BeamElement3D(6, 6, 10, E_a, nu_a, A_a, Iy_a, Iz_a, J_a, structure.nodes))
structure.add_element(BeamElement3D(7, 7, 11, E_a, nu_a, A_a, Iy_a, Iz_a, J_a, structure.nodes))
# B element type
structure.add_element(BeamElement3D(8, 4, 5, E_b, nu_b, A_b, Iy_b, Iz_b, J_b, structure.nodes))
structure.add_element(BeamElement3D(9, 5, 6, E_b, nu_b, A_b, Iy_b, Iz_b, J_b, structure.nodes))
structure.add_element(BeamElement3D(10, 6, 7, E_b, nu_b, A_b, Iy_b, Iz_b, J_b, structure.nodes))
structure.add_element(BeamElement3D(11, 4, 7, E_b, nu_b, A_b, Iy_b, Iz_b, J_b, structure.nodes))
structure.add_element(BeamElement3D(12, 8, 9, E_b, nu_b, A_b, Iy_b, Iz_b, J_b, structure.nodes))
structure.add_element(BeamElement3D(13, 9, 10, E_b, nu_b, A_b, Iy_b, Iz_b, J_b, structure.nodes))
structure.add_element(BeamElement3D(14, 10, 11, E_b, nu_b, A_b, Iy_b, Iz_b, J_b, structure.nodes))
structure.add_element(BeamElement3D(15, 8, 11, E_b, nu_b, A_b, Iy_b, Iz_b, J_b, structure.nodes))

# Apply forces at nodes
structure.nodes[8].apply_force([0, 0, -1, 0, 0, 0])  
structure.nodes[9].apply_force([0, 0, -1, 0, 0, 0])  
structure.nodes[10].apply_force([0, 0, -1, 0, 0, 0])  
structure.nodes[11].apply_force([0, 0, -1, 0, 0, 0])  

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
print("Buckling Mode:\n", buckling_mode)

# Plot
plot_deformed_shape(structure, displacements, scale=1000)
