import pytest
import numpy as np
from DirectStiffness import Node, BeamElement3D, DirectStiffness

# Test Node
def test_node_initialization():
    node = Node(0, 0, 0, 0)
    assert node.id == 0
    assert node.x == 0
    assert node.y == 0
    assert node.z == 0
    assert np.array_equal(node.forces, np.zeros(6))
    assert node.boundary_conditions == [False] * 6

def test_apply_force():
    node = Node(0, 0, 0, 0)
    node.apply_force([1, 2, 3, 4, 5, 6])
    assert np.array_equal(node.forces, np.array([1, 2, 3, 4, 5, 6]))

def test_set_boundary_condition():
    node = Node(0, 0, 0, 0)
    node.set_boundary_condition([True, False, True, False, True, False])
    assert node.boundary_conditions == [True, False, True, False, True, False]

# Test BeamElement3D
def test_beam_element_initialization():
    """Test BeamElement3D initialization and automatic length calculation."""
    nodes = {
        0: Node(0, 0, 0, 0),
        1: Node(1, 1, 1, 1)
    }
    E = 210e9
    nu = 0.3
    A = 0.01
    Iy = 1e-5
    Iz = 1e-5
    J = 1e-5
    beam = BeamElement3D(0, 0, 1, E, nu, A, Iy, Iz, J, nodes)
    
    expected_L = np.sqrt(1**2 + 1**2 + 1**2) 
    assert np.isclose(beam.L, expected_L)
    assert beam.E == 210e9
    assert beam.node1.id == 0
    assert beam.node2.id == 1

def test_local_stiffness_matrix():
    """Test the local stiffness matrix shape and symmetry."""
    nodes = {
        0: Node(0, 0, 0, 0),
        1: Node(1, 1, 0, 0)
    }
    E = 210e9
    nu = 0.3
    A = 0.01
    Iy = 1e-5
    Iz = 1e-5
    J = 1e-5
    beam = BeamElement3D(0, 0, 1, E, nu, A, Iy, Iz, J, nodes)
    k_local = beam.local_stiffness_matrix()

    assert k_local.shape == (12, 12)  
    assert np.allclose(k_local, k_local.T)  

# Test DirectStiffness
def test_assemble_global_stiffness_matrix():
    """Test the global stiffness matrix assembly."""
    structure = DirectStiffness()
    
    node1 = Node(0, 0, 0, 0)
    node2 = Node(1, 1, 0, 0)
    structure.add_node(node1)
    structure.add_node(node2)

    E = 210e9
    nu = 0.3
    A = 0.01
    Iy = 1e-5
    Iz = 1e-5
    J = 1e-5
    nodes = {0: node1, 1: node2}
    beam = BeamElement3D(0, 0, 1, E, nu, A, Iy, Iz, J, nodes)
    structure.add_element(beam)
    
    K_global = structure.assemble_global_stiffness_matrix()

    expected_size = len(structure.nodes) * 6  
    assert K_global.shape == (expected_size, expected_size)
    assert np.allclose(K_global, K_global.T)  

def test_solve_cantilever():
    """Test solving a simple cantilever beam."""
    structure = DirectStiffness()
    
    # Define nodes
    node1 = Node(0, 0, 0, 0)
    node2 = Node(1, 10, 0, 0)
    structure.add_node(node1)
    structure.add_node(node2)

    # Set boundary conditions
    structure.nodes[0].set_boundary_condition([True, True, True, True, True, True])  # 固定端
    structure.nodes[1].set_boundary_condition([False, False, False, False, False, False])

    # Define properties
    E = 200e9
    nu = 0.3
    r = 0.1
    A = np.pi * r ** 2
    Iy = np.pi * r ** 4 / 4
    Iz = np.pi * r ** 4 / 4
    J = np.pi * r ** 4 / 2
    nodes = {0: node1, 1: node2}
    beam = BeamElement3D(0, 0, 1, E, nu, A, Iy, Iz, J, nodes)
    structure.add_element(beam)

    # Apply force
    structure.nodes[1].apply_force([0, 0, -1000, 0, 0, 0])

    displacements, reactions = structure.solve()

    # Check displacement
    assert np.allclose(displacements[0], np.zeros(6)) 
    assert displacements[1][2] < 0  

    # Check reaction force and rotation
    assert np.allclose(reactions[1], np.zeros(6))  
    assert np.isclose(reactions[0][2], 1000, atol=1e-5)  