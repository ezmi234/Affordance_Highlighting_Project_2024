import torch
import kaolin.ops.conversions as conversions
from utils import device
import trimesh

def pointcloud_to_voxel_mesh(points, resolution=16, threshold=0.5, export_path=None):
  """
  Converts a point cloud into a voxel-based triangular mesh.

  Parameters:
      points (torch.Tensor or array-like): Input point cloud data, typically of shape (N, 3),
          where N is the number of points.
      resolution (int, optional): Resolution of the voxel grid. Default is 16.
      threshold (float, optional): Iso-surface value for mesh extraction. Default is 0.5.
      export_path (str, optional): Path to save the generated mesh. If None, the mesh is not saved.

  Returns:
      trimesh.Trimesh: A triangular mesh object created from the voxel grid.

  Raises:
      ValueError: If the generated mesh is empty (no vertices or faces).
  """
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

  # Create a triangular mesh object
  mesh = trimesh.Trimesh(vertices=verts.numpy(), faces=faces.numpy())

  # Apply Laplacian smoothing to improve mesh quality
  mesh = trimesh.smoothing.filter_laplacian(
      mesh, lamb=0.2, iterations=8,
      implicit_time_integration=False,
      volume_constraint=True
  )

  if export_path:
    mesh.export(export_path)

  return mesh