import MinkowskiEngine as ME
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.preprocessing import normalize
import numpy as np
from tqdm import tqdm
import open_clip
import matplotlib.pyplot as plt
import open3d as o3d
import pickle
from huggingface_hub import hf_hub_download
import models
from utils.misc import load_config
from param import parse_args
from utils.data import normalize_pc
from utils.dataset import get_affordance_label
import random
import csv
import optuna


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
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
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

        for step in range(steps):
            optimizer.zero_grad()

            norm_proj = F.normalize(optimized, dim=1)

            # Main similarity loss
            similarity = torch.matmul(norm_proj, text_features.T).squeeze(-1)
            original_loss = -similarity.mean()

            # Contrastive loss
            if contrastive_weight > 0:
                contrastive = self.compute_contrastive_loss(norm_proj, text_features)
                loss = (1 - contrastive_weight) * original_loss + contrastive_weight * contrastive
            else:
                loss = original_loss

            # Backprop & update
            loss.backward()
            optimizer.step()

            # Optional log every 10%
            # if step % max(1, steps // 10) == 0:
            #     print(f"Step {step}/{steps} | Loss: {loss.item():.6f}")

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
        similarity = torch.matmul(F.normalize(projected_features, dim=1), text_features.T).squeeze(-1)
        # print(f"Similarity Shape: {similarity.shape}")
        similarity = similarity.cpu().detach().numpy()

        # gt = get_affordance_label(sample, affordance_class=affordance) > 0.1
        # prediction_mask = similarity > threshold
        # iou = self.compute_iou(prediction_mask, gt)
        # print(f"IoU Score: {iou:.4f}")

        # self.visualize_point_cloud(original_xyz, similarity, threshold)

        return similarity, final_loss


def compute_iou(prediction_mask, ground_truth):
    intersection = np.logical_and(prediction_mask, ground_truth).sum()
    union = np.logical_or(prediction_mask, ground_truth).sum()
    iou = intersection / union if union != 0 else 0
    return iou

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

print("Loading dataset")
input_path = "demo/"
object_class = "Table"
affordance = "support"
# text_prompt = "A gray table with highlighted support surface"
text_prompt = "A gray table with highlighted support surface"
data_path = f"val_set_{object_class}_{affordance}.pkl"
with open(input_path + data_path, 'rb') as f:
    dataset = pickle.load(f)

print(f"Loaded {len(dataset)} samples from {data_path}.")

def objective(trial):
    # Sample hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    steps = trial.suggest_int("steps", 10, 500, step=10)
    threshold = trial.suggest_float("threshold", 0.2, 0.7, step=0.05)
    contrastive_weight = trial.suggest_float("contrastive_weight", 0.0, 0.8, step=0.1)
    voxel_size = trial.suggest_float("voxel_size", 0.005, 0.15, step=0.005)
    clip_model_name = trial.suggest_categorical("clip_model", [
        # "ViT-B-16", "ViT-B-32", "ViT-L-14"
        "ViT-B-32-quickgelu"
    ])
    channels_id = trial.suggest_categorical("channels_id", [
        "medium-channels", "lighter-channels", "heavier-channels", "minimal-channels"
    ])

    channel_configs = {
        "minimal-channels": [3, 8, 16, 32, 64, 128],  # minimal model
        "medium-channels": [3, 32, 64, 128, 256, 512],
        "lighter-channels": [3, 16, 32, 64, 128, 256],  # lighter model
        "heavier-channels": [3, 64, 128, 256, 512, 1024],  # heavier model
    }

    channels = channel_configs[channels_id]

    # Rebuild model
    cli_args, extras = parse_args([])
    config = load_config("src/configs/train.yaml", cli_args=vars(cli_args), extra_args=extras)
    config.model.out_channel = sum(channels[1:-1])
    config.model.channels = channels
    config.model.voxel_size = voxel_size

    output_size = 768
    if str(clip_model_name).startswith("ViT-B-32") or clip_model_name == "ViT-B-16":
        output_size = 512

    # Run on multiple samples
    ious = []
    scores = []
    for sample in dataset[:]:
        gt_mask = get_affordance_label(sample, affordance_class="support")

        model = load_model(config, "OpenShape/openshape-spconv-all").eval()
        projection_layer = ProjectionLayer(input_dim=config.model.out_channel, output_dim=output_size).to('cuda')
        highlighter = OpenShapeHighlighter(clip_model_name, config, model, projection_layer)

        pred_mask, _ = highlighter.detect_affordance(
            sample,
            text_prompt=text_prompt,
            threshold=threshold,
            optimize=True,
            steps=steps,
            lr=lr,
            contrastive_weight=contrastive_weight,
        )

        iou, precision, recall = compute_metrics(pred_mask > threshold, gt_mask > 0.05)
        ious.append(iou)

        # score = (iou + precision + recall) / 3
        score = iou * (precision + recall) / 2  # favors good precision/recall if IoU is high
        scores.append(score)

        del model, projection_layer, highlighter
        # Report intermediate to allow pruning
        trial.report(score, step=len(ious))
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()


    print(f"Trial {trial.number} | IoUs {ious} | Mean IoU: {np.mean(ious):.4f} | Score: {np.mean(scores):.4f}")

    return np.mean(scores)


if __name__ == "__main__":
    # study.optimize(objective, n_trials=100, timeout=3600)
    study = optuna.create_study(
        direction="maximize",  # maximize IoU
        study_name="affordance_optimization",
        storage="sqlite:///affordance_study.db",  # local DB
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=50)
    )

    study.optimize(objective, n_trials=20, timeout=3600)

    print("Best trial:")
    trial = study.best_trial

    print("  Value (mIoU):", trial.value)
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

