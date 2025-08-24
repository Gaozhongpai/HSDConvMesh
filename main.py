#!/usr/bin/env python3
"""
Main script for training and testing spectral dictionary mesh autoencoders.
Supports both SDConv (IJCAI 2021) and HSDConv (TPAMI extension) variants.
"""
import os
import json
import copy
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from config import Config
from setup_utils import (setup_directories, load_shape_data, load_or_generate_matrices, 
                        create_optimizer, create_loss_function)
from autoencoder_dataset import autoencoder_dataset
from models import PaiAutoencoder
from train_funcs import train_autoencoder_dataloader
from test_funcs import test_autoencoder_dataloader
from utils import IOStream
from device import device


def create_dataloaders(config, shapedata):
    """Create training and testing dataloaders."""
    if config.mode == 'train':
        dataset_train = autoencoder_dataset(
            root_dir=config.data_dir,
            points_dataset='train',
            shapedata=shapedata,
            normalization=config.normalization
        )
        dataloader_train = DataLoader(
            dataset_train, 
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers
        )
        
        dataset_val = autoencoder_dataset(
            root_dir=config.data_dir,
            points_dataset='val',
            shapedata=shapedata,
            normalization=config.normalization
        )
        dataloader_val = DataLoader(
            dataset_val,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )
        return dataloader_train, dataloader_val
    
    else:  # test mode
        dataset_test = autoencoder_dataset(
            root_dir=config.data_dir,
            points_dataset='test',
            shapedata=shapedata,
            normalization=config.normalization
        )
        dataloader_test = DataLoader(
            dataset_test,
            batch_size=config.batch_size,
            shuffle=False
        )
        return dataloader_test, None


def create_model(config, vertices, Adj, tD, tU, sizes):
    """Create the autoencoder model."""
    model = PaiAutoencoder(
        filters_enc=config.filter_sizes_enc,
        filters_dec=config.filter_sizes_dec,
        latent_size=config.nz,
        sizes=sizes,
        t_vertices=vertices,
        num_neighbors=config.kernal_size,
        x_neighbors=Adj,
        D=tD, U=tU,
        is_hierarchical=config.is_hierarchical,
        is_old_filter=config.is_old_filter,
        base_size=config.base_size
    ).to(device)
    
    return model


def train_model(config, paths, model, dataloader_train, dataloader_val, optimizer, scheduler, loss_fn, io, shapedata):
    """Train the model."""
    writer = SummaryWriter(paths['summaries'])
    
    # Save configuration
    with open(os.path.join(paths['checkpoints'], f"{config.name}_params.json"), 'w') as fp:
        saveparams = copy.deepcopy(config.__dict__)
        json.dump(saveparams, fp, indent=2)
    
    start_epoch = 0
    if config.resume:
        checkpoint_path = os.path.join(paths['checkpoints'], f"{config.checkpoint_file}.pth.tar")
        print(f'Loading checkpoint from {checkpoint_path}')
        checkpoint_dict = torch.load(checkpoint_path, map_location=device)
        start_epoch = checkpoint_dict['epoch'] + 1
        
        model_dict = model.state_dict()
        pretrained_dict = checkpoint_dict['autoencoder_state_dict']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        if scheduler:
            scheduler.load_state_dict(checkpoint_dict['scheduler_state_dict'])
        print(f'Resuming from epoch {start_epoch}')
    
    train_autoencoder_dataloader(
        dataloader_train, dataloader_val, device, model, optimizer, loss_fn, io,
        bsize=config.batch_size,
        start_epoch=start_epoch,
        n_epochs=config.num_epochs,
        eval_freq=config.eval_frequency,
        scheduler=scheduler,
        writer=writer,
        save_recons=True,
        shapedata=shapedata,
        metadata_dir=paths['checkpoints'],
        samples_dir=paths['samples'],
        checkpoint_path=config.checkpoint_file
    )


def test_model(config, paths, model, dataloader_test, shapedata, io):
    """Test the model."""
    checkpoint_path = os.path.join(paths['checkpoints'], f"{config.checkpoint_file}.pth.tar")
    print(f'Loading checkpoint from {checkpoint_path}')
    checkpoint_dict = torch.load(checkpoint_path, map_location=device)
    
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


def main():
    """Main execution function."""
    # Load configuration
    config = Config()
    
    # Set random seeds
    torch.manual_seed(config.seed)
    
    # Setup directories
    paths = setup_directories(config)
    
    # Initialize logging
    io = IOStream(os.path.join(config.results_folder, 'run.log'))
    io.cprint(f"Configuration: {config}")
    io.cprint(f"Model: {config.generative_model}")
    io.cprint(f"Device: {device}")
    
    # Load data and matrices
    shapedata = load_shape_data(config)
    vertices, Adj, tD, tU, sizes = load_or_generate_matrices(config, shapedata)
    
    # Create model
    model = create_model(config, vertices, Adj, tD, tU, sizes)
    
    # Print model info
    params = sum(param.numel() for name, param in model.named_parameters() 
                if param.requires_grad and 't_vertices' not in name and 'attpool' not in name)
    print(f"Model: {config.generative_model}")
    print(f"Total trainable parameters: {params:,}")
    io.cprint(f"Total trainable parameters: {params:,}")
    
    # Create optimizer and loss function
    optimizer, scheduler = create_optimizer(model, config)
    loss_fn = create_loss_function(config.loss)
    
    # Create dataloaders and run training/testing
    if config.mode == 'train':
        dataloader_train, dataloader_val = create_dataloaders(config, shapedata)
        train_model(config, paths, model, dataloader_train, dataloader_val, 
                   optimizer, scheduler, loss_fn, io, shapedata)
    else:
        dataloader_test, _ = create_dataloaders(config, shapedata)
        test_model(config, paths, model, dataloader_test, shapedata, io)


if __name__ == "__main__":
    main()