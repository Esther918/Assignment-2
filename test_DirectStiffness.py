import pytest
import numpy as np
from DirectStiffness import Node, BeamElement3D, Function

# Test Node
def test_node_initialization():
    node = Node(1, 0, 0, 0)
    assert node.node_i == 1
    assert node.coordinates == (0, 0, 0)
    assert node.forces == [0] * 6
    assert node.boundary_conditions == [False] * 6

def test_apply_force():
    node = Node(1, 0, 0, 0)
    node.apply_force([1, 2, 3, 4, 5, 6])
    assert node.forces == [1, 2, 3, 4, 5, 6]

def test_set_boundary_condition():
    node = Node(1, 0, 0, 0)
    node.set_boundary_condition([True, False, True, False, True, False])
    assert node.boundary_conditions == [True, False, True, False, True, False]

def test_apply_force_invalid_length():
    node = Node(1, 0, 0, 0)
    with pytest.raises(ValueError, match="Force vector must have exactly 6 elements."):
        node.apply_force([1, 2, 3])  

def test_set_boundary_condition_invalid_length():
    node = Node(1, 0, 0, 0)
    with pytest.raises(ValueError, match="Boundary condition vector must have exactly 6 elements."):
        node.set_boundary_condition([True, False])
        
# Test BeamElement3D
def test_local_elastic_stiffness_matrix():
    E = 210e9
    nu = 0.3
    A = 0.01
    L = 2.0
    Iy = 1e-5
    Iz = 1e-5
    J = 1e-5
    beam = BeamElement3D(E, nu, A, L, Iy, Iz, J)
    k_local = beam.local_elastic_stiffness_matrix()
    assert k_local.shape == (12, 12)  
    assert np.allclose(k_local, k_local.T)  

def test_rotation_matrix_3D():
    x1, y1, z1 = 0, 0, 0
    x2, y2, z2 = 1, 0, 0
    gamma = BeamElement3D.rotation_matrix_3D(x1, y1, z1, x2, y2, z2)
    assert gamma.shape == (3, 3)  
    assert np.allclose(np.dot(gamma, gamma.T), np.eye(3))  

def test_transformation_matrix_3D():
    gamma = np.eye(3)  
    Gamma = BeamElement3D.transformation_matrix_3D(gamma)
    assert Gamma.shape == (12, 12)  
    assert np.allclose(Gamma[:3, :3], gamma) 

# Test Function
def test_assemble_global_stiffness():
    func = Function()
    node1 = Node(1, 0, 0, 0)
    node2 = Node(2, 1, 0, 0)
    func.add_node(node1)
    func.add_node(node2)
    
    E = 210e9
    nu = 0.3
    A = 0.01
    L = 1.0
    Iy = 1e-5
    Iz = 1e-5
    J = 1e-5
    beam = BeamElement3D(E, nu, A, L, Iy, Iz, J)
    func.add_element(beam)
    
    K_global = func.assemble_global_stiffness()
    assert K_global.shape == (12, 12)

def test_apply_boundary_conditions():
    """Test the apply_boundary_conditions method."""
    func = Function()
    node1 = Node(0, 0, 0, 0) 
    node1.set_boundary_condition([True, True, True, False, False, False])
    func.add_node(node1)
    
    K = np.eye(6)  # 6x6 Matrix
    F = np.array([1, 2, 3, 4, 5, 6])
    
    K_mod, F_mod = func.apply_boundary_conditions(K, F)
    
    assert np.allclose(K_mod[:3, :3], np.eye(3)) 
    assert np.allclose(F_mod[:3], [0, 0, 0])  