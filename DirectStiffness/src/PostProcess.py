import numpy as np
import matplotlib.pyplot as plt

def plot_deformed_shape(structure, displacements, scale=10):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot original structure
    for element in structure.elements:
        x = [element.node1.x, element.node2.x]
        y = [element.node1.y, element.node2.y]
        z = [element.node1.z, element.node2.z]
        ax.plot(x, y, z, 'k--', label="Original" if element.id == 0 else "")

    # Plot deformed structure
    for element in structure.elements:
        disp_node1 = displacements[element.node1.id, :3]
        disp_node2 = displacements[element.node2.id, :3]

        x_def = [element.node1.x + scale * disp_node1[0], element.node2.x + scale * disp_node2[0]]
        y_def = [element.node1.y + scale * disp_node1[1], element.node2.y + scale * disp_node2[1]]
        z_def = [element.node1.z + scale * disp_node1[2], element.node2.z + scale * disp_node2[2]]

        ax.plot(x_def, y_def, z_def, 'r-', label="Deformed" if element.id == 0 else "")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.title("Original vs Deformed Structure")
    plt.show()