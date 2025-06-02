from mesh import Mesh
import kaolin as kal
from utils import get_camera_from_view2
import matplotlib.pyplot as plt
from utils import device
import torch
import numpy as np


class Renderer:
    def __init__(self, mesh='sample.obj',
                 lights=torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                 camera=kal.render.camera.generate_perspective_projection(np.pi / 3).to(device),
                 dim=(224, 224)):
        if camera is None:
            camera = kal.render.camera.generate_perspective_projection(np.pi / 3).to(device)

        self.lights = lights.unsqueeze(0).to(device)
        self.camera_projection = camera
        self.dim = dim

    def render_views(self, mesh, num_views=5, std=8, center_elev=0, center_azim=0, show=False, lighting=True,
                     background=None, mask=False, return_views=False, return_mask=False, background_img=None):
        verts = mesh.vertices.to(device)
        faces = mesh.faces.to(device)
        n_faces = faces.shape[0]

        elev = torch.randn(num_views) * np.pi / std + center_elev
        azim = torch.randn(num_views) * 2 * np.pi / std + center_azim

        images = []
        masks = []

        if background is not None:
            face_attributes = [
                mesh.face_attributes,
                torch.ones((1, n_faces, 3, 1), device=device)
            ]
        else:
            face_attributes = mesh.face_attributes

        for i in range(num_views):
            camera_transform = get_camera_from_view2(elev[i], azim[i], r=2).to(device)
            face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
                verts, faces, self.camera_projection, camera_transform=camera_transform)

            # Add a small offset to depth to prevent z-fighting
            face_vertices_camera[:, :, :, -1] += 1e-4

            image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
                self.dim[1], self.dim[0], face_vertices_camera[:, :, :, -1],
                face_vertices_image, face_attributes, face_normals[:, :, -1])
            masks.append(soft_mask)

            if background is not None:
                image_features, mask = image_features

            image = torch.clamp(image_features, 0.0, 1.0)

            if lighting:
                # Normalize normals for lighting calculations
                normalized_normals = face_normals / (face_normals.norm(dim=-1, keepdim=True) + 1e-8)
                image_normals = normalized_normals[:, face_idx].squeeze(0)
                image_lighting = kal.render.mesh.spherical_harmonic_lighting(image_normals, self.lights).unsqueeze(0)
                image = image * image_lighting.repeat(1, 3, 1, 1).permute(0, 2, 3, 1).to(device)
                image = torch.clamp(image, 0.0, 1.0)

            if background is not None:
                background_mask = torch.zeros_like(image).to(device)
                mask = mask.squeeze(-1)
                background_idx = torch.where(mask == 0)
                background_mask[background_idx] = background
                image = torch.clamp(image + background_mask, 0.0, 1.0)

            images.append(image)

        images = torch.cat(images, dim=0).permute(0, 3, 1, 2)
        masks = torch.cat(masks, dim=0)

        if background_img is not None:
            bg_tensor = background_img.expand(num_views, -1, -1, -1)
            masks_binary = (masks == 0).unsqueeze(1).expand_as(images)
            images = torch.where(masks_binary, bg_tensor, images)

        if show:
            with torch.no_grad():
                fig, axs = plt.subplots(1 + (num_views - 1) // 4, min(4, num_views), figsize=(15, 15))
                for i in range(num_views):
                    ax = axs.flat[i] if num_views > 1 else axs
                    ax.imshow(images[i].permute(1, 2, 0).cpu().numpy())
                    ax.axis("off")
                plt.show()

        if return_views:
            if return_mask:
                return images, elev, azim, masks
            else:
                return images, elev, azim
        else:
            return images