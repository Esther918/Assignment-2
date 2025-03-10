import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def hermite_interpolation(t, p0, p1, m0, m1):
    """Perform cubic Hermite interpolation."""
    t2 = t * t
    t3 = t2 * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1

def compute_interpolated_deformation(element, displacements, scale=10, num_points=20):
    """Compute interpolated deformed coordinates for a beam element using linear and Hermite interpolation."""
    # Original coordinates
    x = np.array([element.node1.x, element.node2.x])
    y = np.array([element.node1.y, element.node2.y])
    z = np.array([element.node1.z, element.node2.z])

    # Displacement and rotation (avoid scaling rotation directly)
    disp_node1 = displacements[element.node1.id]  
    disp_node2 = displacements[element.node2.id] 
    disp_global = np.concatenate((disp_node1, disp_node2))

    # Transform to local coordinates
    T = element.transformation_matrix()  
    geo_disp_local = T.T @ disp_global * scale
    
    # Extract local displacements and rotations
    u_x1, u_y1, u_z1, theta_x1, theta_y1, theta_z1 = geo_disp_local[:6]
    u_x2, u_y2, u_z2, theta_x2, theta_y2, theta_z2 = geo_disp_local[6:]
    
    # Deformed coordinates (initial)
    x_def = x + [disp_node1[0], disp_node2[0]]
    y_def = y + [disp_node1[1], disp_node2[1]]
    z_def = z + [disp_node1[2], disp_node2[2]]

    # Interpolation points
    t = np.linspace(0, 1, num_points)
    L = element.L  

    # Linear interpolation along the axial direction
    x_def_local = [u_x1, u_x2]
    x_new_local = np.interp(t, [0, 1], x_def_local)

    # Hermite interpolation along the transverse directions
    m0_y = L * theta_z1 
    m1_y = L * theta_z2
    m0_z = L * theta_y1  
    m1_z = L * theta_y2

    y_new_local = hermite_interpolation(t, u_y1, u_y2, m0_y, m1_y)
    z_new_local = hermite_interpolation(t, u_z1, u_z2, m0_z, m1_z)
    local_dis = np.vstack((x_new_local, y_new_local, z_new_local))
    
    R = T[:3, :3]  
    global_dis = R @ local_dis
    
    global_coord = np.array([np.linspace(x[0], x[1], num_points),
                             np.linspace(y[0], y[1], num_points),
                             np.linspace(z[0], z[1], num_points)])
    deform_global_coord = global_coord + global_dis

    return deform_global_coord[0], deform_global_coord[1], deform_global_coord[2]

def plot_deformed_shape(structure, displacements, scale=10):
    """Plot the deformed shape of the whole structure with Hermite interpolation."""
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
        x_def, y_def, z_def = compute_interpolated_deformation(element, displacements, scale, num_points=20)
        ax.plot(x_def, y_def, z_def, 'r-', label="Deformed" if element.id == 0 else "")

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
    ax.set_title("Original vs Deformed Structure")
    plt.savefig("structure_deformed_shape.png")
   