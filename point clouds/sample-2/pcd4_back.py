import os
import numpy as np
import open3d as o3d

# TRANSLATION MULTIPLIER
X = 2000.0

# READ POINT CLOUD FILE (.ply)

file_path = r"C:\Users\Kaye Louise\Desktop\research-3D\GEN_3D_Plants\point clouds\sample-2\.plyS.plyD.ply_.plyL.plyC.ply..plyj.plyp.plyg.ply"           
pcd4 = o3d.io.read_point_cloud(file_path)
#pcd4 = o3d.io.read_point_cloud("PB_D26_PM_SD_LC.j.ply")

# ==================== PCD4 BACK POINT CLOUD ======================= #

"""
# Remove Outlier Points
cl, ind = pcd4.remove_statistical_outlier(nb_neighbors=300, std_ratio=0.5)
print(f"\nBack:\nAfter statistical outlier removal: {len(ind)} points left.")
pcd4 = pcd4.select_by_index(ind)

# DISPLAY SOR POINT CLOUD
#o3d.visualization.draw_geometries([pcd4])

cl, ind = pcd4.remove_radius_outlier(nb_points=3, radius=10000)
print(f"After radius outlier removal: {len(ind)} points left.")
pcd4 = pcd4.select_by_index(ind)
"""

#o3d.visualization.draw_geometries([pcd4])

# ==================== TRANSFORMATION ======================= #

# Translate
#pcd4.translate((-0.4 * X, -0.5 * X, -5.2 * X))
points = np.asarray(pcd4.points)
points += np.array([500 * X, 0 * X, 0 * X])  # Apply manual translation x, y, z
pcd4.points = o3d.utility.Vector3dVector(points)

#o3d.visualization.draw_geometries([pcd4])

# Rotate
# Define theta for each axis (adjust as needed)
theta_x = np.radians(187)  # X-axis rotation (in radians)
theta_y = np.radians(327)  # Y-axis rotation (in radians)
theta_z = np.radians(0)  # Z-axis rotation (in radians)

# Rotation matrices for each axis
rotation_matrix_x = np.array([[1, 0, 0],
                              [0, np.cos(theta_x), -np.sin(theta_x)],
                              [0, np.sin(theta_x), np.cos(theta_x)]])

rotation_matrix_y = np.array([[ np.cos(theta_y), 0, np.sin(theta_y)],
                              [ 0, 1, 0],
                              [-np.sin(theta_y), 0, np.cos(theta_y)]])

rotation_matrix_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                              [np.sin(theta_z), np.cos(theta_z), 0],
                              [0, 0, 1]])

combined_rotation_matrix = rotation_matrix_x @ rotation_matrix_y @ rotation_matrix_z    # Combine the rotation matrices
points = np.asarray(pcd4.points)                             # Access the point cloud points
points_rotated = points @ combined_rotation_matrix.T    # Matrix multiplication, Apply the combined rotation to the points
pcd4.points = o3d.utility.Vector3dVector(points_rotated)    # Update the point cloud with the rotated points

#o3d.visualization.draw_geometries([pcd4])

# Scale
#pcd3.scale(1.0, center=pcd3.get_center())
scale_factor = 1.0  # Define your scaling factor
points_scaled = points * scale_factor  # Scale directly without translating to origin
pcd4.points = o3d.utility.Vector3dVector(points_scaled)

o3d.visualization.draw_geometries([pcd4])
#"""

# ==================== CONVERT POINT CLOUDS TO MESH ======================= #

# Down Sample Point Cloud
down_pcd = pcd4.voxel_down_sample(voxel_size=1)
#o3d.visualization.draw_geometries([down_pcd])

# Estimate Point Cloud Normals
down_pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
down_pcd.estimate_normals()
#o3d.visualization.draw_geometries([down_pcd], point_show_normal=True)

# Create Mesh from Ball Pivoting Algorithm
# Result: Creates a 3D Mesh with the colors retained, however there are holes in the geometry that must be filled
avg_distance = np.mean(down_pcd.compute_nearest_neighbor_distance())
radii = o3d.utility.DoubleVector([avg_distance, avg_distance * 6, avg_distance * 8])
bp_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(down_pcd, radii)
#o3d.visualization.draw_geometries([bp_mesh], mesh_show_back_face=True)

# SAVE MESH FILES
o3d.io.write_triangle_mesh("mesh_bp.ply", bp_mesh)

# Correctly loading the saved meshes
bp_mesh = o3d.io.read_triangle_mesh("mesh_bp.ply")

# Check if the meshes were loaded properly
print(bp_mesh)    # Outputs mesh info for bp_mesh

# Visualize the meshes
o3d.visualization.draw_geometries([bp_mesh], mesh_show_back_face=True)

print(" >>> End Program")