import pytest
import numpy as np
from DirectStiffness import Node, BeamElement3D, DirectStiffness
from ElasticCriticalLoad import ElasticCriticalLoad

# Test ElasticCriticalLoad
def test_elastic_critical_load_initialization():
    """Test ElasticCriticalLoad initialization."""
    structure = DirectStiffness()
    node1 = Node(0, 0, 0, 0)
    node2 = Node(1, 0, 0, 10)
    structure.add_node(node1)
    structure.add_node(node2)
    
    ecl = ElasticCriticalLoad(structure)
    assert ecl.structure == structure

def test_assemble_global_geometric_stiffness_matrix():
    """Test the global geometric stiffness matrix assembly."""
    structure = DirectStiffness()
    
    node1 = Node(0, 0, 0, 0)
    node2 = Node(1, 0, 0, 10)
    structure.add_node(node1)
    structure.add_node(node2)

    structure.nodes[0].set_boundary_condition([True, True, True, True, True, True])
    structure.nodes[1].set_boundary_condition([False, False, False, False, False, False])

    E = 200e9
    nu = 0.3
    r = 0.1
    A = np.pi * r ** 2
    Iy = np.pi * r ** 4 / 4
    Iz = np.pi * r ** 4 / 4
    J = np.pi * r ** 4 / 2
    beam = BeamElement3D(0, 0, 1, E, nu, A, Iy, Iz, J, structure.nodes)
    structure.add_element(beam)

    displacements = np.zeros((2, 6))
    displacements[1, 2] = 0.001  
    structure.nodes[1].apply_force([0, 0, -1000, 0, 0, 0])  

    ecl = ElasticCriticalLoad(structure)
    K_geometric = ecl.assemble_global_geometric_stiffness_matrix(displacements)

    expected_size = len(structure.nodes) * 6  
    assert K_geometric.shape == (expected_size, expected_size)
    assert not np.allclose(K_geometric, np.zeros((expected_size, expected_size)))