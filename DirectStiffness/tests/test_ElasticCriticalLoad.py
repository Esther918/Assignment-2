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
    
    # Define node
    node1 = Node(0, 0, 0, 0)
    node2 = Node(1, 0, 0, 10)
    structure.add_node(node1)
    structure.add_node(node2)

    # Set boundary condition
    structure.nodes[0].set_boundary_condition([True, True, True, True, True, True])
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
    displacements, _ = structure.solve()

    ecl = ElasticCriticalLoad(structure)
    K_geometric = ecl.assemble_global_geometric_stiffness_matrix(displacements)

    expected_size = len(structure.nodes) * 6  
    assert K_geometric.shape == (expected_size, expected_size)
    assert not np.allclose(K_geometric, np.zeros((expected_size, expected_size)))

def test_elastic_critical_load_analysis():
    """Test elastic critical load analysis on a simple column."""
    structure = DirectStiffness()
    
    # Define nodes
    node1 = Node(0, 0, 0, 0)
    node2 = Node(1, 0, 0, 10)
    structure.add_node(node1)
    structure.add_node(node2)

    # Set boundary condition
    structure.nodes[0].set_boundary_condition([True, True, True, True, True, True])
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
    displacements, _ = structure.solve()

    ecl = ElasticCriticalLoad(structure)
    critical_load_factor, buckling_mode = ecl.elastic_critical_load_analysis(displacements)

    assert critical_load_factor > 0  
    assert buckling_mode.shape == (12,)  
    assert not np.allclose(buckling_mode, np.zeros(12)) 