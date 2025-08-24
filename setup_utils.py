"""
Utility functions for setting up directories and preparing data.
"""
import os
import pickle
import torch
import numpy as np
import trimesh
import scipy.sparse as sp
from typing import Tuple, List

from shape_data import ShapeData
from utils import get_adj, sparse_mx_to_torch_sparse_tensor
from device import device
import mesh_sampling


def setup_directories(config) -> dict:
    """Create necessary directories and return paths."""
    paths = {
        'results': config.results_folder,
        'summaries': os.path.join(config.results_folder, 'summaries', config.name),
        'checkpoints': os.path.join(config.results_folder, 'checkpoints', config.name),
        'samples': os.path.join(config.results_folder, 'samples', config.name),
        'predictions': os.path.join(config.results_folder, 'predictions', config.name),
        'downsample': config.downsample_directory
    }
    
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    return paths


def load_shape_data(config) -> ShapeData:
    """Load and prepare shape data."""
    np.random.seed(config.seed)
    print("Loading data...")
    
    mean_file = os.path.join(config.data_dir, 'mean.tch')
    std_file = os.path.join(config.data_dir, 'std.tch')
    
    if not os.path.exists(mean_file) or not os.path.exists(std_file):
        shapedata = ShapeData(
            nVal=config.nVal,
            train_file=os.path.join(config.data_dir, 'train.npy'),
            test_file=os.path.join(config.data_dir, 'test.npy'),
            reference_mesh_file=config.reference_mesh_file,
            normalization=config.normalization,
            meshpackage=config.meshpackage,
            load_flag=True
        )
        torch.save(mean_file, shapedata.mean)
        torch.save(std_file, shapedata.std)
    else:
        shapedata = ShapeData(
            nVal=config.nVal,
            train_file=os.path.join(config.data_dir, 'train.npy'),
            test_file=os.path.join(config.data_dir, 'test.npy'),
            reference_mesh_file=config.reference_mesh_file,
            normalization=config.normalization,
            meshpackage=config.meshpackage,
            load_flag=False
        )
        shapedata.mean = torch.load(mean_file)
        shapedata.std = torch.load(std_file)
        shapedata.n_vertex = shapedata.mean.shape[0]
        shapedata.n_features = shapedata.mean.shape[1]
    
    return shapedata


def load_or_generate_matrices(config, shapedata) -> Tuple[List, List, List]:
    """Load or generate downsampling matrices."""
    downsample_file = os.path.join(config.downsample_directory, 'downsampling_matrices.pkl')
    pai_matrices_file = os.path.join(config.downsample_directory, 'pai_matrices.pkl')
    
    # Load/generate downsampling matrices
    if not os.path.exists(downsample_file):
        if shapedata.meshpackage == 'trimesh':
            raise NotImplementedError('Rerun with mpi-mesh as meshpackage')
        
        print("Generating Transform Matrices...")
        if config.downsample_method == 'COMA_downsample':
            M, A, D, U, F = mesh_sampling.generate_transform_matrices(shapedata.reference_mesh, config.ds_factors)
        
        M_verts_faces = [(M[i].v, M[i].f) for i in range(len(M))]
        with open(downsample_file, 'wb') as fp:
            pickle.dump({'M_verts_faces': M_verts_faces, 'A': A, 'D': D, 'U': U, 'F': F}, fp)
    else:
        print("Loading Transform Matrices...")
        with open(downsample_file, 'rb') as fp:
            downsampling_matrices = pickle.load(fp)
        
        M_verts_faces = downsampling_matrices['M_verts_faces']
        if shapedata.meshpackage == 'trimesh':
            M = [trimesh.base.Trimesh(vertices=M_verts_faces[i][0], faces=M_verts_faces[i][1], process=False) 
                 for i in range(len(M_verts_faces))]
        A = downsampling_matrices['A']
        D = downsampling_matrices['D']
        U = downsampling_matrices['U']
        F = downsampling_matrices['F']
    
    # Create vertex tensors
    vertices = [torch.cat([torch.tensor(M_verts_faces[i][0], dtype=torch.float32), 
                          torch.zeros((1, 3), dtype=torch.float32)], 0).to(device) 
               for i in range(len(M_verts_faces))]
    
    # Get sizes
    if shapedata.meshpackage == 'trimesh':
        sizes = [x.vertices.shape[0] for x in M]
    
    # Load/generate adjacency matrices
    if not os.path.exists(pai_matrices_file):
        print("Generating adjacency matrices...")
        Adj = get_adj(A)
        bU, bD = [], []
        
        for i in range(len(D)):
            d = np.zeros((1, D[i].shape[0] + 1, D[i].shape[1] + 1))
            u = np.zeros((1, U[i].shape[0] + 1, U[i].shape[1] + 1))
            d[0, :-1, :-1] = D[i].todense()
            u[0, :-1, :-1] = U[i].todense()
            d[0, -1, -1] = 1
            u[0, -1, -1] = 1
            bD.append(d)
            bU.append(u)
        
        bD = [sp.csr_matrix(s[0, ...]) for s in bD]
        bU = [sp.csr_matrix(s[0, ...]) for s in bU]
        
        with open(pai_matrices_file, 'wb') as fp:
            pickle.dump([Adj, sizes, bD, bU], fp)
    else:
        print("Loading adjacency matrices...")
        with open(pai_matrices_file, 'rb') as fp:
            Adj, sizes, bD, bU = pickle.load(fp)
    
    tD = [sparse_mx_to_torch_sparse_tensor(s) for s in bD]
    tU = [sparse_mx_to_torch_sparse_tensor(s) for s in bU]
    
    return vertices, Adj, tD, tU, sizes


def create_optimizer(model, config):
    """Create optimizer with proper parameter grouping."""
    trainables_wo_index = [param for name, param in model.named_parameters()
                          if param.requires_grad and 'adjweight' not in name]
    trainables_wt_index = [param for name, param in model.named_parameters()
                          if param.requires_grad and 'adjweight' in name]
    
    optimizer = torch.optim.Adam([
        {'params': trainables_wo_index, 'weight_decay': config.regularization},
        {'params': trainables_wt_index}
    ], lr=config.lr)
    
    scheduler = None
    if config.scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.decay_steps, gamma=config.decay_rate)
    
    return optimizer, scheduler


def create_loss_function(loss_type: str):
    """Create loss function."""
    if loss_type == 'l1':
        def loss_l1(outputs, targets):
            return torch.abs(outputs - targets).mean()
        return loss_l1
    else:
        raise NotImplementedError(f"Loss type {loss_type} not implemented")