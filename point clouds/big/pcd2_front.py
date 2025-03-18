import os
import numpy as np
import open3d as o3d
import pyvista as pv

# TRANSLATION MULTIPLIER
X = 2000.0

file_path = r"C:\Users\Kaye Louise\Desktop\research-3D\GEN_3D_Plants\point clouds\big\.ply2.ply_.plyL.plyC.ply..plyj.plyp.plyg.ply"         
pcd2 = o3d.io.read_point_cloud(file_path)
#pcd2 = o3d.io.read_point_cloud(".ply2.ply_.plyL.plyC.ply..plyj.plyp.plyg.ply")

# ==================== PCD2 FRONT POINT CLOUD ======================= #

"""
# REMOVE OUTLIER POINTS
cl, ind = pcd2.remove_statistical_outlier(nb_neighbors=300, std_ratio=0.5)
pcd2 = pcd2.select_by_index(ind)
#print(f"Front:\nAfter statistical outlier removal: {len(ind)} points left.")

# DISPLAY SOR POINT CLOUD
#o3d.visualization.draw_geometries([pcd2])

cl, ind = pcd2.remove_radius_outlier(nb_points=2, radius=10000)
pcd2 = pcd2.select_by_index(ind)
#print(f"After radius outlier removal: {len(ind)} points left.")

# DISPLAY ROR POINT CLOUD
#o3d.visualization.draw_geometries([pcd2])
"""

# ==================== TRANSFORMATION ======================= #

# Translate
#pcd2.translate((0, 0, 0))
points = np.asarray(pcd2.points)
points += np.array([0, 0, 0])  # Apply manual translation x, y, z
pcd2.points = o3d.utility.Vector3dVector(points)

#o3d.visualization.draw_geometries([pcd2])

# Rotate
# Define theta for each axis (adjust as needed)
theta_x = np.radians(186)  # X-axis rotation (in radians)
theta_y = np.radians(328)  # Y-axis rotation (in radians)
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
points = np.asarray(pcd2.points)                             # Access the point cloud points
points_rotated = points @ combined_rotation_matrix.T    # Matrix multiplication, Apply the combined rotation to the points
pcd2.points = o3d.utility.Vector3dVector(points_rotated)    # Update the point cloud with the rotated points

#o3d.visualization.draw_geometries([pcd2])

# Scale
#pcd2.scale(0.9, center=pcd2.get_center())
scale_factor = 1.0  # Define your scaling factor
points_scaled = points * scale_factor  # Scale directly without translating to origin
pcd2.points = o3d.utility.Vector3dVector(points_scaled)

o3d.visualization.draw_geometries([pcd2])
#"""

# ==================== CONVERT POINT CLOUDS TO MESH ======================= #

# Down Sample Point Cloud
down_pcd = pcd2.voxel_down_sample(voxel_size=1)
#o3d.visualization.draw_geometries([down_pcd])

# Estimate Point Cloud Normals
down_pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
down_pcd.estimate_normals()
#o3d.visualization.draw_geometries([down_pcd], point_show_normal=True)

# Create Mesh from Ball Pivoting Algorithm
# Result: Creates a 3D Mesh with the colors retained, however there are holes in the geometry that must be filled
avg_distance = np.mean(down_pcd.compute_nearest_neighbor_distance())
radii = o3d.utility.DoubleVector([avg_distance, avg_distance * 4, avg_distance * 6])
bp_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(down_pcd, radii)
#o3d.visualization.draw_geometries([bp_mesh], mesh_show_back_face=True)

# Save Mesh Files
o3d.io.write_triangle_mesh("mesh_bp.ply", bp_mesh) # where point locations (x,y,z) are stored

# Loading the saved mesh
bp_mesh = o3d.io.read_triangle_mesh("mesh_bp.ply")

# Check if the meshes were loaded properly
print(bp_mesh)    # Outputs mesh info for bp_mesh

# Visualize the mesh
#o3d.visualization.draw_geometries([bp_mesh], mesh_show_back_face=True)

print(" >>> End Program")