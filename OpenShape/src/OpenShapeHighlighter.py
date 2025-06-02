import csv
import pickle
import random

import MinkowskiEngine as ME
import numpy as np
import open3d as o3d
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

import models
from param import parse_args
from utils.data import normalize_pc
from utils.dataset import get_affordance_label
from utils.misc import load_config


def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(config, model_name):
    """
    Load the OpenShape model with weights from Huggingface Hub.
    """
    # print(f"Loading OpenShape MinkowskiFCNN pre-trained model")

    # --- Initialize Model ---
    model = models.make(config).cuda()

    if config.model.name.startswith('Mink'):
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
    else:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # --- Load Checkpoint from HuggingFace ---
    checkpoint = torch.load(hf_hub_download(repo_id=model_name, filename="model.pt"))
    checkpoint_dict = checkpoint['state_dict']

    model_dict = model.state_dict()
    filtered_checkpoint = {k: v for k, v in checkpoint_dict.items() if
                           k in model_dict and v.size() == model_dict[k].size()}

    model_dict.update(filtered_checkpoint)
    model.load_state_dict(model_dict)

    return model


class ProjectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim=768):
        super(ProjectionLayer, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        return self.projection(x)


# ===== OpenShapeHighlighter =======
class OpenShapeHighlighter:

    def __init__(self, clip_model_name, config, model, projection_layer):
        # print("Initializing OpenShapeHighlighter")
        self.config = config
        self.model = model
        self.projection_layer = projection_layer

        # Initialize OpenCLIP
        # print("Loading OpenCLIP model")
        self.clip_model, _, _ = open_clip.create_model_and_transforms(clip_model_name, pretrained='openai', force_quick_gelu=True)
        self.clip_model = self.clip_model.cuda().eval()
        # print("Models Loaded Successfully.")

    def extract_features(self, sample):
        # Prepare the point cloud
        original_xyz = sample["data_info"]["coordinate"]
        normalized_xyz = normalize_pc(original_xyz)
        xyz_tensor = torch.from_numpy(normalized_xyz).float().to('cuda')
        batched_xyz = ME.utils.batched_coordinates([xyz_tensor], dtype=torch.float32)
        feats = torch.from_numpy(normalized_xyz).float().to('cuda')

        # Extract features
        shape_feat = self.model(batched_xyz, feats, device='cuda',
                                quantization_size=float(self.config.model.voxel_size))
        projected_features = self.projection_layer(shape_feat)
        return projected_features, original_xyz

    @torch.no_grad()
    def extract_text_features(self, text_prompt):
        text_tokens = open_clip.tokenizer.tokenize([text_prompt]).cuda()
        text_features = self.clip_model.encode_text(text_tokens)
        return F.normalize(text_features, dim=1)

    def optimize_affordance(self, projected_features, text_features, steps=400, lr=1e-4, contrastive_weight=0.0):
        # Clone and make trainable
        optimized = projected_features.clone().detach()
        optimized.requires_grad_(True)

        # Detach text to prevent autograd tracking
        text_features = F.normalize(text_features.detach(), dim=1)

        optimizer = torch.optim.Adam([optimized], lr=lr)
        losses = []

        for step in range(steps):
            optimizer.zero_grad()
            similarity = F.cosine_similarity(F.normalize(optimized, dim=1),
                                             F.normalize(text_features, dim=1), dim=-1)

            # Loss is the negative similarity
            loss = -similarity.mean()
            losses.append(loss.item())

            loss.backward()
            optimizer.step()

            # Optional: Normalize the optimized features with Contrastive loss
            # norm_proj = F.normalize(optimized, dim=1)
            #
            # # Main similarity loss
            # similarity = torch.matmul(norm_proj, text_features.T).squeeze(-1)
            # original_loss = -similarity.mean()
            #
            # # Contrastive loss
            # if contrastive_weight > 0:
            #     contrastive = self.compute_contrastive_loss(norm_proj, text_features)
            #     loss = (1 - contrastive_weight) * original_loss + contrastive_weight * contrastive
            # else:
            #     loss = original_loss

            # Backprop & update
            # loss.backward()
            # optimizer.step()

            # Optional log every 10%
            if step % max(1, steps // 10) == 0:
                print(f"Step {step}/{steps} | Loss: {loss.item():.6f}")

        return optimized.detach(), loss.item()

    def visualize_point_cloud(self, coordinates, similarity, threshold=0.2):
        colors = np.zeros((similarity.shape[0], 3))
        colors[similarity > threshold] = [204 / 255, 1., 0.]
        colors[similarity <= threshold] = [180 / 255, 180 / 255, 180 / 255]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coordinates)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])

    def compute_contrastive_loss(self, projected_features, text_features, temperature=0.07):
        """
        Contrastive InfoNCE loss for zero-shot affordance:
        - positive: dot(point, text)
        - negatives: dot(other_points, text)
        """
        text_features = F.normalize(text_features, dim=1)
        projected_features = F.normalize(projected_features, dim=1)

        logits = torch.matmul(projected_features, text_features.T) / temperature
        labels = torch.zeros(len(projected_features), dtype=torch.long, device=logits.device)

        contrastive_loss = F.cross_entropy(logits, labels)
        return contrastive_loss

    def detect_affordance(self, sample, text_prompt, threshold=0.2, optimize=True, steps=100, lr=1e-4, contrastive_weight=0.0):
        projected_features, original_xyz = self.extract_features(sample)
        text_features = self.extract_text_features(text_prompt)
        final_loss = None
        if optimize:
            projected_features, final_loss = self.optimize_affordance(projected_features, text_features, steps=steps,
                                                                      lr=lr, contrastive_weight=contrastive_weight)
        # similarity = torch.matmul(F.normalize(projected_features, dim=1), text_features.T).squeeze(-1)
        # print(f"Similarity Shape: {similarity.shape}")
        # similarity = similarity.cpu().detach().numpy()

        similarity = F.cosine_similarity(projected_features, text_features.expand_as(projected_features), dim=-1)
        similarity = (similarity - similarity.min()) / (similarity.max() - similarity.min() + 1e-8)

        # Convert to numpy for visualization
        pred_mask = similarity.detach().cpu().numpy()

        # gt = get_affordance_label(sample, affordance_class=affordance) > 0.1
        # prediction_mask = similarity > threshold
        # iou = self.compute_iou(prediction_mask, gt)
        # print(f"IoU Score: {iou:.4f}")

        # pred_mask = region_growing(original_xyz, similarity, threshold=threshold, radius=0.04)

        # self.visualize_point_cloud(original_xyz, pred_mask, threshold)

        return pred_mask, final_loss, original_xyz

def to_numpy_bool(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(bool)
    return x.astype(bool)

def compute_metrics(pred_mask, gt_mask):
    pred_mask = to_numpy_bool(pred_mask)
    gt_mask = to_numpy_bool(gt_mask)

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / union if union > 0 else 0

    tp = intersection
    fp = np.logical_and(pred_mask, ~gt_mask).sum()
    fn = np.logical_and(~pred_mask, gt_mask).sum()

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    return iou, precision, recall

def compute_iou(prediction_mask, ground_truth):
    intersection = np.logical_and(prediction_mask, ground_truth).sum()
    union = np.logical_or(prediction_mask, ground_truth).sum()
    iou = intersection / union if union != 0 else 0
    return iou

# Region Growing Algorithm for Affordance Segmentation
def region_growing(coords, similarity, threshold=0.5, radius=0.05, min_size=20):
    coords = coords.cpu().numpy() if isinstance(coords, torch.Tensor) else coords
    similarity = similarity.cpu().numpy() if isinstance(similarity, torch.Tensor) else similarity

    seeds = similarity > threshold
    visited = np.zeros(len(coords), dtype=bool)
    grown_mask = np.zeros(len(coords), dtype=bool)

    nbrs = NearestNeighbors(radius=radius).fit(coords)

    for i in np.where(seeds)[0]:
        if visited[i]:
            continue

        cluster = [i]
        queue = [i]
        visited[i] = True

        while queue:
            idx = queue.pop()
            neighbors = nbrs.radius_neighbors([coords[idx]], return_distance=False)[0]
            for n in neighbors:
                if not visited[n] and similarity[n] > 0.3:  # can lower than threshold
                    visited[n] = True
                    queue.append(n)
                    cluster.append(n)

        if len(cluster) > min_size:
            grown_mask[cluster] = True

    return grown_mask


if __name__ == "__main__":
    # ---- Load the dataset only once ----
    print("Loading dataset")
    input_path = "demo/"
    object_class = "Table"
    affordance = "support"
    text_prompt = "A gray table with highlighted support surface"
    data_path = f"val_set_{object_class}_{affordance}.pkl"
    with open(input_path + data_path, 'rb') as f:
        dataset = pickle.load(f)

    print(f"Loaded {len(dataset)} samples from {data_path}.")

    # === Hyperparameter Grid ===
    learning_rate = 0.005
    steps = 100
    threshold = 0.5
    clip_model_name = "ViT-L-14"
    output_size = 768
    voxel_size = 0.4  # Controls spatial granularity
    channels = [3, 32, 64, 128, 256, 512]
    i = 0
    with open("output.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["sample#", "clip-model", "proj_dim", "voxel_size", "channels", "lr", "steps", "threshold", "steps", "iou", "final_loss"])
        iou_list = []
        score_list = []
        for sample in tqdm(dataset):
            shape_id = sample["shape_id"]
            gt_mask = get_affordance_label(sample, affordance_class=affordance)
            i += 1
            if clip_model_name == "ViT-L-14":
                output_size = 768
            elif clip_model_name == "ViT-B-32":
                output_size = 512
            elif clip_model_name == "ViT-B-16":
                output_size = 512

            set_deterministic()
            cli_args, extras = parse_args([])
            config = load_config("src/configs/train.yaml", cli_args=vars(cli_args), extra_args=extras)
            project_dim = proj_dim = sum(channels[1:-1])
            config.model.out_channel = proj_dim
            config.model.voxel_size = voxel_size
            config.model.channels = channels
            model_name = "OpenShape/openshape-spconv-all"
            model = load_model(config, model_name).eval()
            projection_layer = ProjectionLayer(input_dim=proj_dim, output_dim=output_size).to('cuda')
            highlighter = OpenShapeHighlighter(clip_model_name, config, model, projection_layer)

            pred_class, final_loss, original_xyz = highlighter.detect_affordance(
                sample,
                text_prompt=text_prompt,
                threshold=threshold,
                optimize=True,
                steps=steps,
                lr=learning_rate,
                contrastive_weight=0.0
            )

            iou, precision, recall = compute_metrics(pred_class > threshold, gt_mask > 0.05)

            score = iou * (
                        precision + recall) / 2


            iou_list.append(iou)
            score_list.append(score)
            writer.writerow([i, clip_model_name, proj_dim, voxel_size, "Medium Channel", learning_rate, steps, threshold, steps, iou, final_loss])
            print(f"[✓] {i} | clip-model: {clip_model_name} | proj_dim: {proj_dim} | "
                  f"voxel_size: {voxel_size} | channels: Medium Channel | lr: {learning_rate} | "
                  f"threshold: {threshold} | steps: {steps} | IoU: {iou:.4f} | "
                    f"final_loss: {final_loss:.4f} | ")

    print(f"Average IoU: {np.mean(iou_list):.4f} | Average Score: {np.mean(score_list):.4f}")