import numpy as np
from DirectStiffness import Node, BeamElement3D, Function

'''
Frame Geometry
Nodes:
    Node 0: (0, 0, 0) (Fixed Support)
    Node 1: (0, 3, 0) (Beam-column connection)
    Node 2: (4, 3, 0) (Free end of beam)
Elements:
    Column: Node 0 → Node 1
    Beam: Node 1 → Node 2
Properties:
    E = 200 GPa
    A = 0.01 m^2
    I = 1 * 10^-4 m^4
Loading: 
    10 kN at Node 2

The goal is to determine the nodal displacements and reactions when a point load is applied at the free end.
'''
# Define Nodes (node_index, x, y, z)
node0 = Node(0, 0, 0, 0)
node1 = Node(1, 0, 3, 0)
node2 = Node(2, 4, 3, 0)

# Define properties
E = 200e9  # N/m^2
nu = 0.3 
A = 0.01  # m^2 
Iy = 1e-4 
Iz = 1e-4  
J = 2e-4  

# Define the beam element
column = BeamElement3D(E, nu, A, 3, Iy, Iz, J) # L = 3 m
beam = BeamElement3D(E, nu, A, 4, Iy, Iz, J) # L = 4 m

# Fix Node 0 and Node 1
node0.set_boundary_condition([True, True, True, True, True, True])
node1.set_boundary_condition([True, True, False, True, False, True])

# Apply a 10 kN force at Node 2
node2.apply_force([0, -10000, 0, 0, 0, 0])  # [Fx, Fy, Fz, Mx, My, Mz]


structure = Function()
structure.add_node(node0)
structure.add_node(node1)
structure.add_node(node2)
structure.add_element(beam)
structure.add_element(column)
displacements, reactions = structure.solve()

# Displacements and reaction forces output
print("Displacements at Nodes:")
for i, d in enumerate(displacements.reshape(-1, 6)): 
    print(f"Node {i}: {d}")

print("\nReaction Forces at Node 0:")
for i, r in enumerate(reactions.reshape(-1, 6)):
    print(f"Node {i}: {r}")
