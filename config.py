"""
Configuration settings for spectral dictionary mesh learning.
"""
import os
from dataclasses import dataclass, field
from typing import List, Literal


@dataclass
class Config:
    # Dataset and model configuration
    root_dir: str = 'dataset/COMA-dataset'  # 'COMA-dataset', 'DFAUST-dataset', or 'MANO-dataset'
    is_hierarchical: bool = True  # True: HSDConv, False: SDConv
    is_same_param: bool = False  # Parameter configuration variant
    is_old_filter: bool = False  # False: updated filter (-x), True: original filter
    mode: Literal['train', 'test'] = 'test'  # Training or testing mode
    
    # Mesh processing
    meshpackage: Literal['trimesh', 'mpi-mesh'] = 'trimesh'
    downsample_method: str = 'COMA_downsample'  # or 'meshlab_downsample'
    
    # Network architecture
    ds_factors: List[int] = field(default_factory=lambda: [4, 4, 4, 4])
    kernal_size: List[int] = field(default_factory=lambda: [9, 9, 9, 9, 9])
    step_sizes: List[int] = field(default_factory=lambda: [2, 2, 1, 1, 1])
    filter_sizes_enc: List[int] = field(default_factory=lambda: [3, 16, 32, 64, 128])
    filter_sizes_dec: List[int] = field(default_factory=lambda: [128, 64, 32, 32, 16, 3])
    base_size: int = 32
    
    # Training parameters
    batch_size: int = 32
    num_epochs: int = 300
    eval_frequency: int = 200
    num_workers: int = 8
    nz: int = 32  # Latent dimension
    lr: float = 1e-3
    regularization: float = 5e-5
    scheduler: bool = True
    decay_rate: float = 0.99
    decay_steps: int = 1
    loss: str = 'l1'
    
    # Data processing
    seed: int = 2
    shuffle: bool = True
    nVal: int = 100
    normalization: bool = True
    resume: bool = False
    
    # GPU settings
    device_idx: int = 0
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Generate model name based on configuration
        self.generative_model = 'SDConvFinal'
        if not self.is_old_filter:
            self.generative_model += '-x'
        if self.is_hierarchical:
            self.generative_model = 'H' + self.generative_model
        if self.is_same_param:
            self.generative_model += '-param'
        
        # Adjust filter sizes for specific datasets if using same_param
        if self.is_same_param:
            if "COMA" in self.root_dir:
                self.filter_sizes_enc = [3, 32, 45, 64, 128]
                self.filter_sizes_dec = [128, 80, 48, 32, 32, 3]
            elif "DFAUST" in self.root_dir:
                self.filter_sizes_enc = [3, 32, 42, 80, 128]
                self.filter_sizes_dec = [128, 80, 64, 40, 32, 3]
        
        # Set up paths
        self.name = 'sliced'
        self.reference_mesh_file = os.path.join(self.root_dir, 'template.obj')
        self.downsample_directory = os.path.join(self.root_dir, self.downsample_method)
        self.data_dir = os.path.join(self.root_dir, 'Processed', self.name)
        self.results_folder = os.path.join(self.root_dir, 'results', self.generative_model, f'latent_{self.nz}')
        
        # Create checkpoint file name
        self.checkpoint_file = 'checkpoint'