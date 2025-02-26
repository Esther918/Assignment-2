import numpy as np
from DirectStiffness import Node, BeamElement3D, Function

'''
Frame Geometry
Nodes:
    Node 0: (0, 0, 0) (Fixed Support)
    Node 1: (4, 0, 0) (Free end)
Elements:
    E0: Node 0 to Node 1
Properties:
    E = 200 GPa
    A = 0.01 m^2
    I = 1 * 10^-4 m^4
Loading: 
    10 kN at Node 1

The goal is to determine the nodal displacements and reactions when a point load is applied at the free end.
'''

# Define Nodes (node_index, x, y, z)
node0 = Node(0, 0, 0, 0)  # Fixed support
node1 = Node(1, 4, 0, 0)  # Free end

# Define properties
E = 200e9  # N/m^2
nu = 0.3 
A = 0.01  # m^2 
Iy = 1e-4  # m^4
Iz = 1e-4  # m^4
J = 2e-4   # m^4

# Define the beam elements
nodes = {0: node0, 1: node1}  # Dictionary to store nodes
E0 = BeamElement3D(0, 1, E, nu, A, Iy, Iz, J, nodes)

# Apply boundary conditions
node0.set_boundary_condition([True, True, True, True, True, True]) 

# Apply a 10 kN force at Node 1
node1.apply_force([0, -10000, 0, 0, 0, 0])  # [Fx, Fy, Fz, Mx, My, Mz]

# Create the structure and add nodes and elements
structure = Function()
structure.add_node(node0)
structure.add_node(node1)
structure.add_element(E0)

# Solve for displacements and reactions
displacements, reactions = structure.solve()

# Displacements and reaction forces output
print("Displacements at Nodes:")
for i, d in enumerate(displacements.reshape(-1, 6)): 
    print(f"Node {i}: {d}")

print("\nReaction Forces at Nodes:")
for i, r in enumerate(reactions.reshape(-1, 6)):
    print(f"Node {i}: {r}")
