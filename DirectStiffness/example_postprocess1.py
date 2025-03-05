from DirectStiffness import Node, BeamElement3D, DirectStiffness
from ElasticCriticalLoad import ElasticCriticalLoad
from PostProcess import plot_deformed_shape
import numpy as np

structure = DirectStiffness()
# Define nodes (node_index, x, y, z)
structure.add_node(Node(0, 0, 0, 0))  
structure.add_node(Node(1, 25/6, 50/6, 37/6))  
structure.add_node(Node(2, 50/6, 100/6, 74/6))  
structure.add_node(Node(3, 75/6, 150/6, 111/6))  
structure.add_node(Node(4, 100/6, 200/6, 148/6))  
structure.add_node(Node(5, 125/6, 250/6, 185/6))  
structure.add_node(Node(6, 25, 50, 37))  

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
L = 50
P_analytical = np.pi ** 2 * E * Iz / (2 * L) ** 2 

# Define the beam element (element name, node_i, node_j, E, nu, A, Iy, Iz, J, nodes):
structure.add_element(BeamElement3D(0, 0, 1, E, nu, A, Iy, Iz, J, structure.nodes))
structure.add_element(BeamElement3D(1, 1, 2, E, nu, A, Iy, Iz, J, structure.nodes))
structure.add_element(BeamElement3D(2, 2, 3, E, nu, A, Iy, Iz, J, structure.nodes))
structure.add_element(BeamElement3D(3, 3, 4, E, nu, A, Iy, Iz, J, structure.nodes))
structure.add_element(BeamElement3D(4, 4, 5, E, nu, A, Iy, Iz, J, structure.nodes))
structure.add_element(BeamElement3D(5, 5, 6, E, nu, A, Iy, Iz, J, structure.nodes))

# Apply forces at nodes
structure.nodes[6].apply_force([0.05, -0.1, 0.23, 0.1, -0.025, -0.08])  

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
print("P_analytical:\n", P_analytical)

# Plot
plot_deformed_shape(structure, displacements, scale=10)