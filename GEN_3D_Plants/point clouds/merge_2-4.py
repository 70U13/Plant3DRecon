import os
import numpy as np
import open3d as o3d

# TRANSLATION MULTIPLIER
X = 2000.0

pcd2 = o3d.io.read_point_cloud("PA_D31_AM_SB_LC.j.ply")
pcd4 = o3d.io.read_point_cloud("PA_D31_AM_SD_LC.j.ply")

# ==================== PCD3 RIGHT POINT CLOUD ======================= #

# ==================== REMOVING OUTLIER POINTS ======================= #

"""
cl, ind = pcd2.remove_statistical_outlier(nb_neighbors=300, std_ratio=0.5)
pcd2 = pcd2.select_by_index(ind)
#print(f"Front:\nAfter statistical outlier removal: {len(ind)} points left.")

#o3d.visualization.draw_geometries([pcd2])

cl, ind = pcd2.remove_radius_outlier(nb_points=4, radius=10000) # 2 10000
pcd2 = pcd2.select_by_index(ind)
#print(f"After radius outlier removal: {len(ind)} points left.")

#o3d.visualization.draw_geometries([pcd2])
#"""

# ==================== TRASNFORMING ======================= #

#pcd2.translate((0, 0, 0))
points = np.asarray(pcd2.points)
points += np.array([-280 * X, 30 * X, -90 * X])  # Apply manual translation x, y, z
pcd2.points = o3d.utility.Vector3dVector(points)

#o3d.visualization.draw_geometries([pcd2])

# Define theta for each axis (adjust as needed)
theta_x = np.radians(8)  # 187 X-axis rotation (in radians)
theta_y = np.radians(10)  # 325 Y-axis rotation (in radians)
theta_z = np.radians(-8)  # Z-axis rotation (in radians)

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
points = np.asarray(pcd2.points)                             # Access the point cloud points
points_rotated = points @ combined_rotation_matrix.T    # Matrix multiplication, Apply the combined rotation to the points
pcd2.points = o3d.utility.Vector3dVector(points_rotated)    # Update the point cloud with the rotated points

#o3d.visualization.draw_geometries([pcd2])

#pcd2.scale(0.9, center=pcd2.get_center())
scale_factor = 1.0  # Define your scaling factor
points_scaled = points * scale_factor  # Scale directly without translating to origin
pcd2.points = o3d.utility.Vector3dVector(points_scaled)

#o3d.visualization.draw_geometries([pcd2])

# ==================== PCD4 BACK POINT CLOUD ======================= #

# ==================== REMOVING OUTLIER POINTS ======================= #

"""
cl, ind = pcd4.remove_statistical_outlier(nb_neighbors=300, std_ratio=0.5)
pcd4 = pcd4.select_by_index(ind)
#print(f"\nBack:\nAfter statistical outlier removal: {len(ind)} points left.")

#o3d.visualization.draw_geometries([pcd4])

cl, ind = pcd4.remove_radius_outlier(nb_points=2, radius=8000)
pcd4 = pcd4.select_by_index(ind)
#print(f"After radius outlier removal: {len(ind)} points left.")

#o3d.visualization.draw_geometries([pcd4])
#"""

# ==================== TRANSFORMING ======================= #

#pcd4.translate((-0.4 * X, -0.5 * X, -5.2 * X))
points = np.asarray(pcd4.points)
points += np.array([-300 * X, -130 * X, -870 * X])  # -350, -150, -950
pcd4.points = o3d.utility.Vector3dVector(points) # -900 -100 -900

#o3d.visualization.draw_geometries([pcd4])

# Define theta for each axis (adjust as needed)
theta_x = np.radians(-15)  # 360 X-axis rotation (in radians)
theta_y = np.radians(200)  # 330 Y-axis rotation (in radians)
theta_z = np.radians(10)  # Z-axis rotation (in radians)

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

#pcd3.scale(1.0, center=pcd3.get_center())
scale_factor = 1.0  # Define your scaling factor
points_scaled = points * scale_factor  # Scale directly without translating to origin
pcd4.points = o3d.utility.Vector3dVector(points_scaled)

#o3d.visualization.draw_geometries([pcd4])

# ==================== MERGE 2 POINT CLOUDS ==================== #

merged_pcd = pcd2 + pcd4 

# Down Sample Point Cloud
down_pcd = merged_pcd.voxel_down_sample(voxel_size=1)
#o3d.visualization.draw_geometries([down_pcd])

# Estimate Point Cloud Normals
down_pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
down_pcd.estimate_normals()

"""
# REMOVE REMAINING OUTLIERS
cl, ind = merged_pcd.remove_statistical_outlier(nb_neighbors=2000, std_ratio=2)
merged_pcd = merged_pcd.select_by_index(ind)
cl, ind = merged_pcd.remove_radius_outlier(nb_points=2, radius=8000)
merged_pcd = merged_pcd.select_by_index(ind)
"""

o3d.visualization.draw_geometries([merged_pcd])

# BPA Mesh
avg_distance = np.mean(down_pcd.compute_nearest_neighbor_distance())
radii = o3d.utility.DoubleVector([avg_distance * 2, avg_distance * 5, avg_distance * 10])
bp_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(down_pcd, radii)

# Save and load the BPA mesh
o3d.io.write_triangle_mesh("merged_bp.ply", bp_mesh)
bp_mesh = o3d.io.read_triangle_mesh("merged_bp.ply")
o3d.visualization.draw_geometries([bp_mesh], mesh_show_back_face=True)