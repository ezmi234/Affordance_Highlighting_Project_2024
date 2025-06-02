import numpy as np
import torch
import open3d as o3d
import kaolin.ops.conversions as conversions
from utils import device
import trimesh
from scipy.spatial import cKDTree

def visualize_affordance_pointcloud(points, labels, color_positive=[204/255, 1., 0.], color_negative=[180/255, 180/255, 180/255], point_size=5.0):
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

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    render_opt = vis.get_render_option()
    render_opt.point_size = point_size  # Set point size here
    vis.run()
    vis.destroy_window()

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

  # Compute the minimum and maximum coordinates of the point cloud
  min_coords, _ = points.min(dim=0)
  max_coords, _ = points.max(dim=0)
  scale = max_coords - min_coords

  # Normalize the point cloud to fit within a unit cube
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

def project_vertex_scores_to_pointcloud(mesh: trimesh.Trimesh, vertex_scores: torch.Tensor, pointcloud: torch.Tensor, device: str = 'cpu'):
    """
    Projects per-vertex scores (float) from mesh to point cloud via face interpolation.

    Args:
        mesh (trimesh.Trimesh): mesh with faces and vertices
        vertex_scores (torch.Tensor): [V], score per vertex
        pointcloud (torch.Tensor): [N, 3], target point cloud
        device (str): device to use for computation

    Returns:
        torch.Tensor: [N], predicted scores for each point
    """

    face_tensor = torch.tensor(mesh.faces, device=device)
    face_scores = vertex_scores[face_tensor].mean(dim=1)

    samples_np, face_indices_np = trimesh.sample.sample_surface(mesh, 10 * pointcloud.shape[0])
    tree = cKDTree(samples_np)

    # Nearest mesh sample to each point
    pointcloud_np = pointcloud.detach().cpu().numpy()
    _, nearest_idx = tree.query(pointcloud_np, k=1)
    nearest_faces = face_indices_np[nearest_idx]

    # Assign scores from face to point
    return face_scores[nearest_faces].to(pointcloud.device)
