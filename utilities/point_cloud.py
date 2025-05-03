import numpy as np
import torch
import open3d as o3d
import kaolin.ops.conversions as conversions
from utils import device
import trimesh
from scipy.spatial import cKDTree

def visualize_affordance_pointcloud(points, labels, color_positive=[1.0, 0.0, 0.0], color_negative=[0.6, 0.6, 0.6]):
    """
    Visualize a point cloud and highlight affordance-labeled points.

    Args:
        points (np.ndarray): shape (N, 3)
        labels (np.ndarray): binary or float mask, shape (N,)
        color_positive (list): RGB for highlighted points (default: red)
        color_negative (list): RGB for background points (default: gray)
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    colors = np.tile(color_negative, (points.shape[0], 1))
    colors[np.where(labels > 0.5)] = color_positive
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])

def pointcloud_to_voxel_mesh(points, resolution=64, threshold=0.5, export_path=None):
  if not isinstance(points, torch.Tensor):
    points = torch.tensor(points, dtype=torch.float32)

  min_coords, _ = points.min(dim=0)
  max_coords, _ = points.max(dim=0)
  scale = max_coords - min_coords
  points_norm = (points - min_coords) / scale

  voxel_grid = conversions.pointclouds_to_voxelgrids(points_norm.unsqueeze(0), resolution=resolution).to(device)
  verts_faces = conversions.voxelgrids_to_trianglemeshes(voxel_grid, iso_value=threshold)

  verts = verts_faces[0][0].cpu() / resolution
  faces = verts_faces[1][0].cpu()

  # Denormalize
  scale = scale.cpu()
  min_coords = min_coords.cpu()
  verts = verts * scale + min_coords

  if verts.numel() == 0 or faces.numel() == 0:
      raise ValueError("Empty mesh generated from voxel grid.")

  # Create mesh
  mesh = trimesh.Trimesh(vertices=verts.numpy(), faces=faces.numpy())

  # Smoothing and export
  mesh = trimesh.smoothing.filter_laplacian(
      mesh, lamb=0.2, iterations=8,
      implicit_time_integration=False,
      volume_constraint=True
  )

  if export_path:
    mesh.export(export_path)

  return mesh

def project_mesh_labels_to_pointcloud_torch(mesh: trimesh.Trimesh, face_labels: torch.Tensor, pointcloud: torch.Tensor):
    """
    Projects mesh face labels onto a point cloud using nearest surface sampling.

    Args:
        mesh (trimesh.Trimesh): mesh with predicted per-face labels.
        face_labels (torch.Tensor): shape (F,), tensor of predicted labels per face.
        pointcloud (torch.Tensor): shape (N, 3), input point cloud.

    Returns:
        torch.Tensor: shape (N,), predicted label per point.
    """
    assert face_labels.shape[0] == len(mesh.faces), "Mismatch between number of mesh faces and face_labels"

    # Convert pointcloud to numpy for KD-tree query
    pointcloud_np = pointcloud.detach().cpu().numpy()

    # Sample dense points on mesh surface
    samples_np, face_indices_np = trimesh.sample.sample_surface(mesh, 10 * pointcloud_np.shape[0])

    # KD-tree on sampled mesh surface points
    tree = cKDTree(samples_np)
    _, nearest_sample_idxs = tree.query(pointcloud_np, k=1)

    # Map nearest point back to face index
    nearest_face_ids = face_indices_np[nearest_sample_idxs]

    # Get the label for each face and convert back to torch tensor
    projected_labels = face_labels[nearest_face_ids]

    return projected_labels.to(pointcloud.device)