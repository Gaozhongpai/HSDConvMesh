#!/usr/bin/env python3
"""Run latent-space interpolation experiments and quantify smoothness/reconstruction."""
import argparse
import json
import os
import random
import time
from typing import List, Tuple

import numpy as np
import torch

from autoencoder_dataset import autoencoder_dataset
from config import Config
from device import device
from main import create_model
from setup_utils import (load_or_generate_matrices, load_shape_data,
                         setup_directories)


def _prepare_config(root_dir_override: str | None) -> Config:
    config = Config()
    if root_dir_override:
        config.root_dir = root_dir_override
        config.__post_init__()
    config.mode = 'test'
    return config


def _load_model(config: Config,
                vertices,
                Adj,
                tD,
                tU,
                sizes) -> torch.nn.Module:
    model = create_model(config, vertices, Adj, tD, tU, sizes)
    checkpoint_path = os.path.join(
        config.results_folder,
        'checkpoints',
        config.name,
        f"{config.checkpoint_file}.pth.tar"
    )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Missing checkpoint: {checkpoint_path}')

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = model.state_dict()
    pretrained = checkpoint['autoencoder_state_dict']
    pretrained = {
        k: v for k, v in pretrained.items()
        if k in state_dict and 'U.' not in k and 'D.' not in k
    }
    state_dict.update(pretrained)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _select_pairs(dataset_len: int, num_pairs: int, rng: random.Random) -> List[Tuple[int, int]]:
    if num_pairs * 2 > dataset_len:
        raise ValueError('Requested more pairs than available unique samples')
    indices = list(range(dataset_len))
    rng.shuffle(indices)
    pairs = []
    for i in range(num_pairs):
        pairs.append((indices[2 * i], indices[2 * i + 1]))
    return pairs


def _to_tensor(data, *, target_device) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.to(target_device)
    return torch.tensor(data, device=target_device, dtype=torch.float32)


def _denormalize(mesh: torch.Tensor,
                 mean: torch.Tensor,
                 std: torch.Tensor) -> torch.Tensor:
    return mesh * std + mean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-dir', type=str, default=None,
                        help='Override dataset root directory from config.py')
    parser.add_argument('--num-pairs', type=int, default=10,
                        help='Number of random interpolation pairs to evaluate')
    parser.add_argument('--num-steps', type=int, default=10,
                        help='Number of interpolation samples (including endpoints)')
    parser.add_argument('--seed', type=int, default=2,
                        help='Random seed for pair sampling')
    parser.add_argument('--mm-constant', type=float, default=1000.0,
                        help='Scale factor to convert units to millimeters')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to store interpolated meshes and metrics')
    parser.add_argument('--save-normalized', action='store_true',
                        help='Also save normalized meshes with dummy node appended')
    args = parser.parse_args()

    if args.num_steps < 2:
        raise ValueError('num_steps must be at least 2 to form an interpolation path')

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = _prepare_config(args.root_dir)
    paths = setup_directories(config)

    shapedata = load_shape_data(config)
    vertices, Adj, tD, tU, sizes = load_or_generate_matrices(config, shapedata)
    model = _load_model(config, vertices, Adj, tD, tU, sizes)

    dataset_test = autoencoder_dataset(
        root_dir=config.data_dir,
        points_dataset='test',
        shapedata=shapedata,
        normalization=config.normalization
    )

    if args.num_pairs * 2 > len(dataset_test):
        raise ValueError('Not enough test samples for the requested number of pairs')

    mean = _to_tensor(shapedata.mean, target_device=device)
    std = _to_tensor(shapedata.std, target_device=device)

    output_root = args.output_dir or os.path.join(
        config.results_folder,
        'interpolation_experiments'
    )
    _ensure_dir(output_root)

    pairs = _select_pairs(len(dataset_test), args.num_pairs, rng)
    alphas = torch.linspace(0.0, 1.0, steps=args.num_steps, device=device)

    displacement_records: List[torch.Tensor] = []
    self_recon_records: List[torch.Tensor] = []
    pair_summaries = []

    with torch.no_grad():
        for pair_idx, (idx_a, idx_b) in enumerate(pairs):
            sample_a = dataset_test[idx_a]['points'].unsqueeze(0).to(device)
            sample_b = dataset_test[idx_b]['points'].unsqueeze(0).to(device)

            batch = torch.cat([sample_a, sample_b], dim=0)
            if getattr(model, 'is_hierarchical', False):
                model.update()
            latents = model.encode(batch)
            z_a, z_b = latents[0:1], latents[1:2]

            z_interp = (1.0 - alphas.view(-1, 1)) * z_a + alphas.view(-1, 1) * z_b

            decoded_full_norm = model.decode(z_interp)
            decoded_norm = decoded_full_norm[:, :-1, :]
            decoded_denorm = _denormalize(decoded_norm, mean, std)

            pair_dir = os.path.join(output_root, f'pair_{pair_idx:03d}_{idx_a}-{idx_b}')
            _ensure_dir(pair_dir)

            for step_idx in range(args.num_steps):
                step_path = os.path.join(pair_dir, f'interp_step_{step_idx:02d}.tch')
                torch.save(decoded_denorm[step_idx].cpu(), step_path)
                if args.save_normalized:
                    norm_path = os.path.join(pair_dir, f'interp_step_{step_idx:02d}_norm.tch')
                    torch.save(decoded_full_norm[step_idx].cpu(), norm_path)

            diffs = decoded_denorm[1:] - decoded_denorm[:-1]
            disp = torch.sqrt((diffs ** 2).sum(dim=2)) * args.mm_constant
            mean_disp = disp.mean(dim=1)
            displacement_records.append(mean_disp.cpu())

            recon_norm_full = model(decoded_full_norm)
            recon_norm = recon_norm_full[:, :-1, :]

            interp_denorm = _denormalize(decoded_norm, mean, std)
            recon_denorm = _denormalize(recon_norm, mean, std)
            recon_diff = torch.sqrt(((recon_denorm - interp_denorm) ** 2).sum(dim=2)) * args.mm_constant
            mean_self = recon_diff.mean(dim=1)
            self_recon_records.append(mean_self.cpu())

            pair_summary = {
                'pair_index': pair_idx,
                'sample_indices': [int(idx_a), int(idx_b)],
                'sample_paths': [str(dataset_test.paths[idx_a]), str(dataset_test.paths[idx_b])],
                'per_step_mean_displacement_mm': mean_disp.cpu().tolist(),
                'per_mesh_self_recon_mm': mean_self.cpu().tolist(),
            }
            pair_summaries.append(pair_summary)

    disp_tensor = torch.stack(displacement_records)
    self_tensor = torch.stack(self_recon_records)

    metrics = {
        'num_pairs': args.num_pairs,
        'num_steps': args.num_steps,
        'seed': args.seed,
        'mm_constant': args.mm_constant,
        'root_dir': config.root_dir,
        'avg_step_displacement_mm': disp_tensor.mean().item(),
        'std_step_displacement_mm': disp_tensor.flatten().std(unbiased=False).item(),
        'avg_self_recon_error_mm': self_tensor.mean().item(),
        'std_self_recon_error_mm': self_tensor.flatten().std(unbiased=False).item(),
        'per_step_mean_displacement_mm': disp_tensor.mean(dim=0).tolist(),
        'per_step_displacement_std_mm': disp_tensor.std(dim=0, unbiased=False).tolist(),
        'per_mesh_self_recon_mm': self_tensor.mean(dim=0).tolist(),
        'per_mesh_self_recon_std_mm': self_tensor.std(dim=0, unbiased=False).tolist(),
    }

    timestamp = time.strftime('%Y%m%d-%H%M%S')
    summary_path = os.path.join(output_root, f'experiment_metrics_{timestamp}.json')
    with open(summary_path, 'w', encoding='utf-8') as fp:
        json.dump({'metrics': metrics, 'pairs': pair_summaries}, fp, indent=2)

    print(f'Saved interpolation meshes under: {output_root}')
    print(f'Metrics written to: {summary_path}')
    print('Average per-step displacement (mm):', metrics['avg_step_displacement_mm'])
    print('Average self-reconstruction error (mm):', metrics['avg_self_recon_error_mm'])


if __name__ == '__main__':
    main()
