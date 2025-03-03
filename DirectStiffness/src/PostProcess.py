import numpy as np
import matplotlib.pyplot as plt
from DirectStiffness import Node, BeamElement3D, Function

def compute_local_forces_moments(element, displacements):
    """
    Compute internal forces and moments.
    """
    displacements_flat = displacements.flatten()
    dofs_node1 = [element.node1.id * 6 + dof for dof in element.node1.dofs]
    dofs_node2 = [element.node2.id * 6 + dof for dof in element.node2.dofs]
    dofs = dofs_node1 + dofs_node2
    # Debug: Print displacements and dofs
    # print(f"displacements shape: {displacements.shape}")
    # print(f"dofs: {dofs}")

    global_displacements = displacements_flat[dofs]

    T = element.transformation_matrix()
    local_displacements = T @ global_displacements

    k_local = element.local_stiffness_matrix()
    local_forces_moments = k_local @ local_displacements

    return local_forces_moments

def plot_local_forces_moments(element, local_forces_moments):
    """
    Plot internal forces and moments.
    """
    Fx1, Fy1, Fz1, Mx1, My1, Mz1, Fx2, Fy2, Fz2, Mx2, My2, Mz2 = local_forces_moments

    # Plot forces
    plt.figure()
    plt.title(f"Element {element.id} Internal Forces")
    plt.plot([0, element.L], [Fx1, Fx2], label="Fx")
    plt.plot([0, element.L], [Fy1, Fy2], label="Fy")
    plt.plot([0, element.L], [Fz1, Fz2], label="Fz")
    plt.xlabel("Length along element")
    plt.ylabel("Force")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot moments
    plt.figure()
    plt.title(f"Element {element.id} Internal Moments")
    plt.plot([0, element.L], [Mx1, Mx2], label="Mx")
    plt.plot([0, element.L], [My1, My2], label="My")
    plt.plot([0, element.L], [Mz1, Mz2], label="Mz")
    plt.xlabel("Length along element")
    plt.ylabel("Moment")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_deformed_shape(structure, displacements, scale=10):
    """
    Plot the deformed shape of the whole structure.
    """     
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot original shape
    for element in structure.elements:
        x = [element.node1.x, element.node2.x]
        y = [element.node1.y, element.node2.y]
        z = [element.node1.z, element.node2.z]
        ax.plot(x, y, z, 'k--', label="Original" if element.id == 0 else "")

    # Plot deformed shape
    for element in structure.elements:
        disp_node1 = displacements[element.node1.id, :3]
        disp_node2 = displacements[element.node2.id, :3]
        # # Debug: Print disp_node1 and disp_node2
        # print(f"Element {element.id}: disp_node1 = {disp_node1}, disp_node2 = {disp_node2}")

        x_def = [element.node1.x + scale * disp_node1[0], element.node2.x + scale * disp_node2[0]]
        y_def = [element.node1.y + scale * disp_node1[1], element.node2.y + scale * disp_node2[1]]
        z_def = [element.node1.z + scale * disp_node1[2], element.node2.z + scale * disp_node2[2]]

        ax.plot(x_def, y_def, z_def, 'r-', label="Deformed" if element.id == 0 else "")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()