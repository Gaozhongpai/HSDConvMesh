#!/usr/bin/env python3
"""
Spectral Dictionary Mesh Autoencoder - Main Script
Supports SDConv (IJCAI 2021) and HSDConv (TPAMI extension)

This script maintains backwards compatibility while providing cleaner structure.
For new projects, consider using main.py with config.py for better organization.
"""
import numpy as np
import json
import os
import copy
import pickle
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import mesh_sampling
import trimesh
from shape_data import ShapeData
from autoencoder_dataset import autoencoder_dataset
from utils import get_adj, sparse_mx_to_torch_sparse_tensor, IOStream
from models import PaiAutoencoder
from train_funcs import train_autoencoder_dataloader
from test_funcs import test_autoencoder_dataloader
import scipy.sparse as sp
from device import device

# ============================================================================
# CONFIGURATION PARAMETERS - Modify these as needed
# ============================================================================

# Dataset and model configuration
root_dir = 'dataset/COMA-dataset'   # 'COMA-dataset', 'DFAUST-dataset', or 'MANO-dataset'
is_hierarchical = True              # True: HSDConv, False: SDConv
is_same_param = False               # Parameter configuration variant
is_old_filter = False               # False: updated filter (-x), True: original filter
mode = 'train'                       # 'train' or 'test'
meshpackage = 'trimesh'             # 'trimesh' or 'mpi-mesh'

# Generate model name automatically
generative_model = 'SDConvFinal'
if not is_old_filter:
    generative_model += '-x'
if is_hierarchical:
    generative_model = 'H' + generative_model
if is_same_param:
    generative_model += '-param'

name = 'sliced'
device_idx = 0


# ============================================================================
# NETWORK AND TRAINING PARAMETERS
# ============================================================================

# Network architecture parameters
downsample_method = 'COMA_downsample'  # 'COMA_downsample' or 'meshlab_downsample'
ds_factors = [4, 4, 4, 4]
kernal_size = [9, 9, 9, 9, 9]
step_sizes = [2, 2, 1, 1, 1]
base_size = 32

# Default filter sizes
filter_sizes_enc = [3, 16, 32, 64, 128]
filter_sizes_dec = [128, 64, 32, 32, 16, 3]

# Adjust filter sizes for specific datasets if using parameter variant
if is_same_param: 
    if "COMA" in root_dir:
        filter_sizes_enc = [3, 32, 45, 64, 128]
        filter_sizes_dec = [128, 80, 48, 32, 32, 3]
    elif "DFAUST" in root_dir:
        filter_sizes_enc = [3, 32, 42, 80, 128]
        filter_sizes_dec = [128, 80, 64, 40, 32, 3]

# File paths
reference_mesh_file = os.path.join(root_dir, 'template.obj')
downsample_directory = os.path.join(root_dir, downsample_method)
data_dir = os.path.join(root_dir, 'Processed', name)
results_folder = os.path.join(root_dir, 'results', generative_model, 'latent_32')

# Training parameters
args = {
    'generative_model': generative_model,
    'name': name, 
    'data': data_dir,
    'results_folder': results_folder,
    'reference_mesh_file': reference_mesh_file, 
    'downsample_directory': downsample_directory,
    'checkpoint_file': 'checkpoint',
    'seed': 2, 
    'loss': 'l1',
    'batch_size': 32, 
    'num_epochs': 300, 
    'eval_frequency': 200, 
    'num_workers': 8,
    'filter_sizes_enc': filter_sizes_enc, 
    'filter_sizes_dec': filter_sizes_dec,
    'nz': 32,  # latent dimension
    'ds_factors': ds_factors, 
    'step_sizes': step_sizes,
    'lr': 1e-3, 
    'regularization': 5e-5,
    'scheduler': True, 
    'decay_rate': 0.99,
    'decay_steps': 1,
    'resume': False,
    'mode': mode, 
    'shuffle': True, 
    'nVal': 100, 
    'normalization': True
}

# ============================================================================
# SETUP DIRECTORIES
# ============================================================================

def setup_directories():
    """Create necessary directories."""
    directories = [
        args['results_folder'],
        os.path.join(args['results_folder'], 'summaries', name),
        os.path.join(args['results_folder'], 'checkpoints', name),
        os.path.join(args['results_folder'], 'samples', name),
        os.path.join(args['results_folder'], 'predictions', name),
        downsample_directory
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    return {
        'summary': os.path.join(args['results_folder'], 'summaries', name),
        'checkpoint': os.path.join(args['results_folder'], 'checkpoints', name),
        'samples': os.path.join(args['results_folder'], 'samples', name),
        'predictions': os.path.join(args['results_folder'], 'predictions', name)
    }

paths = setup_directories()


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_shape_data():
    """Load and prepare shape data."""
    np.random.seed(args['seed'])
    print("Loading data...")
    
    mean_file = os.path.join(args['data'], 'mean.tch')
    std_file = os.path.join(args['data'], 'std.tch')
    
    if not os.path.exists(mean_file) or not os.path.exists(std_file):
        shapedata = ShapeData(
            nVal=args['nVal'],
            train_file=os.path.join(args['data'], 'train.npy'),
            test_file=os.path.join(args['data'], 'test.npy'),
            reference_mesh_file=args['reference_mesh_file'],
            normalization=args['normalization'],
            meshpackage=meshpackage, 
            load_flag=True
        )
        torch.save(mean_file, shapedata.mean)
        torch.save(std_file, shapedata.std)
    else:
        shapedata = ShapeData(
            nVal=args['nVal'],
            train_file=os.path.join(args['data'], 'train.npy'),
            test_file=os.path.join(args['data'], 'test.npy'),
            reference_mesh_file=args['reference_mesh_file'],
            normalization=args['normalization'],
            meshpackage=meshpackage, 
            load_flag=False
        )
        shapedata.mean = torch.load(mean_file)
        shapedata.std = torch.load(std_file)
        shapedata.n_vertex = shapedata.mean.shape[0]
        shapedata.n_features = shapedata.mean.shape[1]
    
    return shapedata

def load_matrices(shapedata):
    """Load or generate downsampling and adjacency matrices."""
    downsample_file = os.path.join(args['downsample_directory'], 'downsampling_matrices.pkl')
    pai_matrices_file = os.path.join(args['downsample_directory'], 'pai_matrices.pkl')
    
    # Load/generate downsampling matrices
    if not os.path.exists(downsample_file):
        if shapedata.meshpackage == 'trimesh':
            raise NotImplementedError('Rerun with mpi-mesh as meshpackage')
        
        print("Generating Transform Matrices...")
        if downsample_method == 'COMA_downsample':
            M, A, D, U, F = mesh_sampling.generate_transform_matrices(shapedata.reference_mesh, args['ds_factors'])
        
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

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def create_dataloaders(shapedata):
    """Create training/testing dataloaders."""
    if args['mode'] == 'train':
        dataset_train = autoencoder_dataset(
            root_dir=args['data'],
            points_dataset='train',
            shapedata=shapedata,
            normalization=args['normalization']
        )
        dataloader_train = DataLoader(
            dataset_train, 
            batch_size=args['batch_size'],
            shuffle=args['shuffle'],
            num_workers=args['num_workers']
        )
        
        dataset_val = autoencoder_dataset(
            root_dir=args['data'],
            points_dataset='val',
            shapedata=shapedata,
            normalization=args['normalization']
        )
        dataloader_val = DataLoader(
            dataset_val, 
            batch_size=args['batch_size'],
            shuffle=False,
            num_workers=args['num_workers']
        )
        return dataloader_train, dataloader_val
    else:
        dataset_test = autoencoder_dataset(
            root_dir=args['data'],
            points_dataset='test',
            shapedata=shapedata,
            normalization=args['normalization']
        )
        dataloader_test = DataLoader(
            dataset_test, 
            batch_size=args['batch_size'],
            shuffle=False
        )
        return dataloader_test, None


def main():
    """Main execution function."""
    # Set random seeds
    torch.manual_seed(args['seed'])
    
    # Initialize logging
    io = IOStream(os.path.join(args['results_folder'], 'run.log'))
    io.cprint(f"Configuration: {args}")
    io.cprint(f"Model: {generative_model}")
    io.cprint(f"Device: {device}")
    
    # Load data and matrices
    shapedata = load_shape_data()
    vertices, Adj, tD, tU, sizes = load_matrices(shapedata)
    
    # Create dataloaders
    if args['mode'] == 'train':
        dataloader_train, dataloader_val = create_dataloaders(shapedata)
    else:
        dataloader_test, _ = create_dataloaders(shapedata)

    # Create model
    model = PaiAutoencoder(
        filters_enc=args['filter_sizes_enc'],
        filters_dec=args['filter_sizes_dec'],
        latent_size=args['nz'],
        sizes=sizes,
        t_vertices=vertices,
        num_neighbors=kernal_size,
        x_neighbors=Adj,
        D=tD, U=tU, 
        is_hierarchical=is_hierarchical,
        is_old_filter=is_old_filter,
        base_size=base_size
    ).to(device)
    
    # Create optimizer
    trainables_wo_index = [param for name, param in model.named_parameters()
                          if param.requires_grad and 'adjweight' not in name]
    trainables_wt_index = [param for name, param in model.named_parameters()
                          if param.requires_grad and 'adjweight' in name]
    
    optim = torch.optim.Adam([
        {'params': trainables_wo_index, 'weight_decay': args['regularization']},
        {'params': trainables_wt_index}
    ], lr=args['lr'])
    
    scheduler = None
    if args['scheduler']:
        scheduler = torch.optim.lr_scheduler.StepLR(optim, args['decay_steps'], gamma=args['decay_rate'])
    
    # Create loss function
    if args['loss'] == 'l1':
        def loss_l1(outputs, targets):
            return torch.abs(outputs - targets).mean()
        loss_fn = loss_l1
    
    # Print model information
    params = sum(param.numel() for name, param in model.named_parameters() 
                if param.requires_grad and 't_vertices' not in name and 'attpool' not in name)
    print(f"Model: {generative_model}")
    print(f"Total trainable parameters: {params:,}")
    io.cprint(f"Total trainable parameters: {params:,}")
    
    # Training mode
    if args['mode'] == 'train':
        writer = SummaryWriter(paths['summary'])
        
        # Save configuration
        with open(os.path.join(paths['checkpoint'], f"{args['name']}_params.json"), 'w') as fp:
            saveparams = copy.deepcopy(args)
            json.dump(saveparams, fp, indent=2)
        
        start_epoch = 0
        if args['resume']:
            checkpoint_file = os.path.join(paths['checkpoint'], f"{args['checkpoint_file']}.pth.tar")
            print(f'Loading checkpoint from {checkpoint_file}')
            checkpoint_dict = torch.load(checkpoint_file, map_location=device)
            start_epoch = checkpoint_dict['epoch'] + 1
            
            model_dict = model.state_dict()
            pretrained_dict = checkpoint_dict['autoencoder_state_dict']
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(pretrained_dict, strict=False)
            optim.load_state_dict(checkpoint_dict['optimizer_state_dict'])
            if scheduler:
                scheduler.load_state_dict(checkpoint_dict['scheduler_state_dict'])
            print(f'Resuming from epoch {start_epoch}')
        
        train_autoencoder_dataloader(
            dataloader_train, dataloader_val, device, model, optim, loss_fn, io,
            bsize=args['batch_size'],
            start_epoch=start_epoch,
            n_epochs=args['num_epochs'],
            eval_freq=args['eval_frequency'],
            scheduler=scheduler,
            writer=writer,
            save_recons=True,
            shapedata=shapedata,
            metadata_dir=paths['checkpoint'],
            samples_dir=paths['samples'],
            checkpoint_path=args['checkpoint_file']
        )
    
    # Testing mode
    elif args['mode'] == 'test':
        checkpoint_file = os.path.join(paths['checkpoint'], f"{args['checkpoint_file']}.pth.tar")
        print(f'Loading checkpoint from {checkpoint_file}')
        checkpoint_dict = torch.load(checkpoint_file, map_location=device)
        
        print(f'Current Epoch is {checkpoint_dict["epoch"]}')
        model_dict = model.state_dict()
        pretrained_dict = checkpoint_dict['autoencoder_state_dict']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and "U." not in k and "D." not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        
        predictions, norm_l1_loss, l2_loss = test_autoencoder_dataloader(
            device, model, dataloader_test, shapedata, mm_constant=1000
        )
        
        # Save results
        torch.save(predictions, os.path.join(paths['predictions'], 'predictions.tch'))
        torch.save({'norm_l1_loss': norm_l1_loss, 'l2_loss': l2_loss}, 
                  os.path.join(paths['predictions'], 'loss.tch'))
        
        io.cprint(f'autoencoder: normalized loss={norm_l1_loss.item()}')
        io.cprint(f'autoencoder: euclidean distance in mm={l2_loss.item()}')


if __name__ == "__main__":
    main()
