import numpy as np
import matplotlib.pyplot as plt

def compute_interpolated_deformation(element, displacements, scale=10, num_points=20):
    """
    Compute interpolated deformed coordinates for a beam element using linear and Hermite interpolation.
    Returns both the original displacement and the Hermite interpolated displacement.
    """
    # Original coordinates
    x = np.array([element.node1.x, element.node2.x])
    y = np.array([element.node1.y, element.node2.y])
    z = np.array([element.node1.z, element.node2.z])

    # Displacement and rotation (avoid scaling rotation directly)
    disp_node1 = displacements[element.node1.id, :3] * scale  
    disp_node2 = displacements[element.node2.id, :3] * scale
    theta_node1 = displacements[element.node1.id, 3:]  
    theta_node2 = displacements[element.node2.id, 3:]

    # Deformed coordinates (initial)
    x_def = x + [disp_node1[0], disp_node2[0]]
    y_def = y + [disp_node1[1], disp_node2[1]]
    z_def = z + [disp_node1[2], disp_node2[2]]

    # Interpolation points
    L = element.L
    t = np.linspace(0, 1, num_points) # t = x/L
    
    # Linear interpolation along the axial direction (u)
    u1, u2 = disp_node1[0], disp_node2[0]
    u = np.interp(t, [0, 1], [u1, u2])

    # Cubic Hermite interpolation along the transverse directions (v, w)
    v1, v2 = disp_node1[1], disp_node2[1]
    w1, w2 = disp_node1[2], disp_node2[2]
    theta_z1, theta_z2 = theta_node1[2], theta_node2[2]
    theta_y1, theta_y2 = theta_node1[1], theta_node2[1]

    # Hermite basis functions
    h1 = 1 - 3 * t**2 + 2 * t**3
    h2 = 3 * t**2 - 2 * t**3
    h3 = t * (1 - t)**2
    h4 = t * (t**2 - t)

    # Hermite interpolation for v and w
    v = h1 * v1 + h2 * v2 + h3 * theta_z1 * L + h4 * theta_z2 * L
    w = h1 * w1 + h2 * w2 + h3 * theta_y1 * L + h4 * theta_y2 * L

    # Original displacement (linear interpolation)
    x_original = np.interp(t, [0, 1], x_def)
    y_original = np.interp(t, [0, 1], y_def)
    z_original = np.interp(t, [0, 1], z_def)

    # Hermite interpolated displacement
    x_hermite = x + u
    y_hermite = y + v
    z_hermite = z + w

    return (x_original, y_original, z_original), (x_hermite, y_hermite, z_hermite)

def plot_deformed_shape(structure, displacements, scale=10):
    """Plot the original displacement and Hermite interpolated deformed shape of the whole structure."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot original structure
    for element in structure.elements:
        x = [element.node1.x, element.node2.x]
        y = [element.node1.y, element.node2.y]
        z = [element.node1.z, element.node2.z]
        ax.plot(x, y, z, 'k--', label="Original Structure" if element.id == 0 else "")

    # Plot original displacement and Hermite interpolated deformation
    for element in structure.elements:
        (x_orig, y_orig, z_orig), (x_herm, y_herm, z_herm) = compute_interpolated_deformation(element, displacements, scale, num_points=20)
        
        # Plot original displacement
        ax.plot(x_orig, y_orig, z_orig, 'b-', label="Original Displacement" if element.id == 0 else "")
        
        # Plot Hermite interpolated deformation
        ax.plot(x_herm, y_herm, z_herm, 'r-', label="Hermite Deformed" if element.id == 0 else "")

    # Compute coordinate range
    all_x = [node.x for node in structure.nodes.values()]
    all_y = [node.y for node in structure.nodes.values()]
    all_z = [node.z for node in structure.nodes.values()]
    x_min = min(all_x) - max(abs(displacements[:, 0])) * scale * 1.2
    x_max = max(all_x) + max(abs(displacements[:, 0])) * scale * 1.2
    y_min = min(all_y) - max(abs(displacements[:, 1])) * scale * 1.2
    y_max = max(all_y) + max(abs(displacements[:, 1])) * scale * 1.2
    z_min = min(all_z) - max(abs(displacements[:, 2])) * scale * 1.2
    z_max = max(all_z) + max(abs(displacements[:, 2])) * scale * 1.2
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.grid(True)
    ax.set_title("Original Displacement vs Hermite Deformed Structure")
    plt.savefig("structure_deformed_shape.png")