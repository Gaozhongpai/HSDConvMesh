#!/usr/bin/env python3
"""Extract and save downsampled mesh templates from downsampling_matrices.pkl as PLY files."""

import argparse
import os
import pickle
import sys

import numpy as np
import trimesh


def save_mesh_as_ply(vertices: np.ndarray, faces: np.ndarray, output_path: str) -> None:
    """Save mesh as PLY file using trimesh."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(output_path)
    print(f"Saved: {output_path} ({len(vertices)} vertices, {len(faces)} faces)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='dataset/DFAUST-dataset',
        help='Dataset directory containing COMA_downsample/ folder'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for PLY files (default: same as COMA_downsample/)'
    )
    args = parser.parse_args()

    # Paths
    downsample_dir = os.path.join(args.dataset_dir, 'COMA_downsample')
    matrices_path = os.path.join(downsample_dir, 'downsampling_matrices.pkl')
    template_path = os.path.join(args.dataset_dir, 'template.obj')

    output_dir = args.output_dir or downsample_dir
    os.makedirs(output_dir, exist_ok=True)

    # Check files exist
    if not os.path.exists(matrices_path):
        print(f"Error: {matrices_path} not found!")
        sys.exit(1)
    if not os.path.exists(template_path):
        print(f"Error: {template_path} not found!")
        sys.exit(1)

    print(f"Loading downsampling matrices from: {matrices_path}")
    with open(matrices_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    # Extract data
    # data structure: {'M_verts_faces': (verts, faces) tuples, 'A': adjacency, 'D': downsample, 'U': upsample, 'F': faces}
    M_verts_faces = data.get('M_verts_faces') or data.get('M')
    F = data.get('F')  # List of face arrays
    D = data.get('D')  # Downsampling matrices

    if M_verts_faces is None:
        print("Error: 'M_verts_faces' or 'M' not found in pickle file!")
        print(f"Available keys: {list(data.keys())}")
        sys.exit(1)

    print(f"\nFound {len(M_verts_faces)} mesh levels (including original template)")

    # Check if M_verts_faces contains tuples (verts, faces) or mesh objects
    first_item = M_verts_faces[0]
    if isinstance(first_item, tuple):
        print(f"Mesh 0: {first_item[0].shape[0]} vertices (original template)")
    else:
        print(f"Mesh 0: {first_item.v.shape[0]} vertices (original template)")

    # Save original template as PLY
    template_mesh = trimesh.load(template_path, process=False)
    original_ply_path = os.path.join(output_dir, 'template_d0.ply')
    save_mesh_as_ply(template_mesh.vertices, template_mesh.faces, original_ply_path)

    # Save each downsampled level
    for i in range(1, len(M_verts_faces)):
        # Handle both tuple format (verts, faces) and object format
        if isinstance(M_verts_faces[i], tuple):
            vertices, faces = M_verts_faces[i]
        else:
            vertices = M_verts_faces[i].v
            faces = M_verts_faces[i].f if hasattr(M_verts_faces[i], 'f') else F[i-1] if F else None

        if faces is None:
            print(f"Warning: No faces found for level {i}, skipping...")
            continue

        output_path = os.path.join(output_dir, f'template_d{i}.ply')
        save_mesh_as_ply(vertices, faces, output_path)

        # Print downsampling ratio
        original_verts = M_verts_faces[0][0] if isinstance(M_verts_faces[0], tuple) else M_verts_faces[0].v
        ratio = vertices.shape[0] / original_verts.shape[0]
        print(f"  → Level {i}: {vertices.shape[0]} vertices ({ratio*100:.1f}% of original)")

    print(f"\n✓ All downsampled templates saved to: {output_dir}")
    print(f"  Files: template_d0.ply (original) through template_d{len(M_verts_faces)-1}.ply")


if __name__ == '__main__':
    main()
