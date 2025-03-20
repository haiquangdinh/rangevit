import os
import yaml
import argparse
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast

# Import necessary modules from the existing codebase
from models.rangevit import RangeViT
from dataset.semantic_kitti import SemanticKitti
from dataset.range_view_loader import RangeViewLoader

# Set TensorFlow environment variables
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs

# Configure PyTorch
if torch.cuda.is_available():
    # Print GPU information
    print("🚀 GPU Information:")
    gpu_count = torch.cuda.device_count()
    print(f"   Found {gpu_count} GPU(s)")
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"   GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")
    
    # Optimize CUDA
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    print("No GPU available, using CPU")

class RangeViTTrainer:
    """
    Modern PyTorch trainer for RangeViT on SemanticKITTI dataset
    """
    def __init__(self, args):
        self.args = args
        
        # Set up GPU and optimization
        if torch.cuda.is_available():
            cuda_device = 0
            print(f"GPU detected: {torch.cuda.get_device_name(cuda_device)}")
            print(f"CUDA capability: {torch.cuda.get_device_capability(cuda_device)}")
            print(f"Memory available: {torch.cuda.get_device_properties(cuda_device).total_memory / 1e9:.2f} GB")
            
            # Optimize CUDA for better performance
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            self.device = torch.device('cuda')
        else:
            print("No GPU detected, using CPU")
            self.device = torch.device('cpu')
        
        # Set up directories
        self.log_dir = os.path.join(args.save_path, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, 'checkpoints'), exist_ok=True)
        
        # Set up tensorboard
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Load configuration
        with open(args.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Save configuration
        with open(os.path.join(self.log_dir, 'config.yaml'), 'w') as f:
            yaml.dump(self.config, f)
            
        # Initialize model, datasets, and optimizers
        self._init_model()
        self._init_dataloaders()
        self._init_optimizer()
        
        # Set up mixed precision training
        self.scaler = GradScaler(enabled=args.use_amp)
        
        # Training state
        self.current_epoch = 0
        self.best_miou = 0.0
        
        # Load checkpoint if specified
        if args.checkpoint:
            self._load_checkpoint(args.checkpoint)
            
        print(f"Training on {self.device}")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
    
    def _init_model(self):
        """Initialize the RangeViT model"""
        print("Initializing model...")
        
        # Create model with all parameters
        self.model = RangeViT(
            in_channels=self.config.get('in_channels', 5),
            n_cls=self.config.get('n_classes', 20),
            backbone=self.config.get('vit_backbone', 'vit_small_patch16_384'),
            image_size=self.config.get('image_size', (64, 384)),
            pretrained_path=self.args.pretrained_model,
            new_patch_size=self.config.get('patch_size', [2, 8]),
            new_patch_stride=self.config.get('patch_stride', [2, 8]),
            reuse_pos_emb=self.config.get('reuse_pos_emb', True),
            reuse_patch_emb=self.config.get('reuse_patch_emb', False),
            conv_stem=self.config.get('conv_stem', 'ConvStem'),
            stem_base_channels=self.config.get('stem_base_channels', 32),
            stem_hidden_dim=self.config.get('D_h', 256),
            skip_filters=self.config.get('skip_filters', 256),
            decoder=self.config.get('decoder', 'up_conv'),
            up_conv_d_decoder=self.config.get('D_h', 256),
            up_conv_scale_factor=self.config.get('patch_stride', [2, 8]),
            use_kpconv=self.config.get('use_kpconv', True)
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Use multiple GPUs if available
        if torch.cuda.device_count() > 1 and not self.args.distributed:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        
        # If using distributed training
        if self.args.distributed:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank
            )
    
    def _init_dataloaders(self):
        """Initialize the dataloaders for training and validation"""
        print("Initializing dataloaders...")
        
        # Load SemanticKITTI dataset configuration
        data_config_path = 'dataset/semantic_kitti/semantic-kitti.yaml'
        data_config = yaml.safe_load(open(data_config_path, 'r'))
        
        # Use the train+val split if specified
        if self.config.get('use_trainval', False):
            print('Training with the train+val set.')
            train_sequences = data_config['split']['train'] + data_config['split']['valid']
        else:
            train_sequences = data_config['split']['train']
        
        # Initialize datasets
        trainset = SemanticKitti(
            root=self.args.data_root,
            sequences=train_sequences,
            config_path=data_config_path
        )
        
        valset = SemanticKitti(
            root=self.args.data_root,
            sequences=data_config['split']['valid'],
            config_path=data_config_path
        )
        
        # Calculate class weights for weighted loss
        self.cls_weight = 1 / (trainset.cls_freq + 1e-3)
        self.ignore_class = []
        for cl, _ in enumerate(self.cls_weight):
            if trainset.data_config['learning_ignore'][cl]:
                self.cls_weight[cl] = 0
            if self.cls_weight[cl] < 1e-10:
                self.ignore_class.append(cl)
        
        self.cls_weight = torch.tensor(self.cls_weight).to(self.device)
        print(f'Class weights: {self.cls_weight}')
        
        # Initialize range view loaders
        self.train_range_loader = RangeViewLoader(
            dataset=trainset,
            config=self.config,
            is_train=True,
            use_kpconv=self.config.get('use_kpconv', True)
        )
        
        self.val_range_loader = RangeViewLoader(
            dataset=valset,
            config=self.config,
            is_train=False,
            use_kpconv=self.config.get('use_kpconv', True)
        )
        
        # Get custom collate function if using KPConv
        collate_fn = None
        if self.config.get('use_kpconv', True):
            from dataset import custom_collate_kpconv_fn
            collate_fn = custom_collate_kpconv_fn
        
        # Initialize dataloaders
        self.train_loader = DataLoader(
            self.train_range_loader,
            batch_size=self.config.get('batch_size', 4),
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn
        )
        
        self.val_loader = DataLoader(
            self.val_range_loader,
            batch_size=self.config.get('batch_size_val', 1),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        self.mapped_cls_name = trainset.mapped_cls_name
    
    def _init_optimizer(self):
        """Initialize optimizer and learning rate scheduler"""
        print("Initializing optimizer...")
        
        # Set up optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('lr', 0.0004),
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        # Set up learning rate scheduler with warmup
        total_steps = len(self.train_loader) * self.config.get('n_epochs', 60)
        warmup_steps = len(self.train_loader) * self.config.get('warmup_epochs', 10)
        
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
            )
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _load_checkpoint(self, checkpoint_path):
        """Load model, optimizer, and training state from checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model weights
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint['model'])
        
        # Load optimizer and scheduler state if training
        if not self.args.val_only:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.current_epoch = checkpoint['epoch'] + 1
            self.best_miou = checkpoint['best_miou']
            
            if 'scaler' in checkpoint and checkpoint['scaler'] is not None:
                self.scaler.load_state_dict(checkpoint['scaler'])
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    def _save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint of the model"""
        checkpoint = {
            'model': self.model.state_dict() if not hasattr(self.model, 'module') else self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch,
            'best_miou': self.best_miou,
            'scaler': self.scaler.state_dict() if self.args.use_amp else None,
            'config': self.config
        }
        
        # Save the latest checkpoint
        torch.save(checkpoint, os.path.join(self.log_dir, 'checkpoints', 'latest.pth'))
        
        # Save checkpoint for the current epoch
        torch.save(checkpoint, os.path.join(self.log_dir, 'checkpoints', f'epoch_{epoch}.pth'))
        
        # Save the best checkpoint
        if is_best:
            torch.save(checkpoint, os.path.join(self.log_dir, 'checkpoints', 'best.pth'))
    
    def train_one_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        torch.cuda.empty_cache()  # Clear GPU cache at start of epoch
        
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.get('n_epochs', 60)}")
        
        for batch_idx, data in enumerate(progress_bar):
            if self.config.get('use_kpconv', True):
                feats, labels, _ = data
                feats = feats.to(self.device)
                labels = labels.to(self.device)
            else:
                feats, labels, _ = data
                feats = feats.to(self.device)
                labels = labels.to(self.device)
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.args.use_amp):
                outputs = self.model(feats)
                
                # Calculate loss
                criterion = nn.CrossEntropyLoss(weight=self.cls_weight, ignore_index=0)
                loss = criterion(outputs, labels)
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update learning rate
            self.scheduler.step()
            
            # Update metrics
            epoch_loss += loss.item()
            
            # Calculate accuracy (ignoring ignored classes)
            mask = labels != 0
            pred = outputs.argmax(dim=1)
            correct += (pred[mask] == labels[mask]).sum().item()
            total += mask.sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / max(1, total),
                'lr': self.scheduler.get_last_lr()[0]
            })
            
            # Log to tensorboard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('train/loss', loss.item(), global_step)
            self.writer.add_scalar('train/accuracy', 100. * correct / max(1, total), global_step)
            self.writer.add_scalar('train/learning_rate', self.scheduler.get_last_lr()[0], global_step)
        
        # Log epoch metrics
        epoch_loss /= len(self.train_loader)
        epoch_acc = 100. * correct / max(1, total)
        
        print(f"Epoch {epoch+1}/{self.config.get('n_epochs', 60)} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
        
        return epoch_loss, epoch_acc
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        torch.cuda.empty_cache()  # Clear GPU cache before validation
        
        # Initialize metrics
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(self.val_loader, desc="Validation")
        
        for batch_idx, batch_data in enumerate(progress_bar):
            # Print first batch data structure for debugging
            if batch_idx == 0:
                print(f"Batch data type: {type(batch_data)}")
                if isinstance(batch_data, dict):
                    print(f"Batch data keys: {batch_data.keys()}")
                elif isinstance(batch_data, (list, tuple)):
                    print(f"Batch data length: {len(batch_data)}")
                    
            try:
                if self.config.get('use_kpconv', True):
                    # Handle KPConv input format
                    if isinstance(batch_data, dict):
                        batch_dict = batch_data
                        # Change 'input_feature' to 'input2d'
                        feats = batch_dict['input2d'].to(self.device) if 'input2d' in batch_dict else None
                        labels = batch_dict['labels'].to(self.device) if 'labels' in batch_dict else None
                        
                        # Get all the required KPConv parameters
                        px = batch_dict['px'].to(self.device) if 'px' in batch_dict else None
                        py = batch_dict['py'].to(self.device) if 'py' in batch_dict else None
                        pxyz = batch_dict['points_xyz'].to(self.device) if 'points_xyz' in batch_dict else None
                        knns = batch_dict['knns'].to(self.device) if 'knns' in batch_dict else None
                        num_points = batch_dict['num_points'] if 'num_points' in batch_dict else None
                        
                        # Forward pass with all required parameters
                        outputs = self.model(feats, px, py, pxyz, knns, num_points)
                        
                        # Fix: Ensure labels has proper dimensions for CrossEntropyLoss (BxHxW)
                        if labels.dim() == 1:
                            # For KPConv output, we need to reshape labels to match the output format
                            # This assumes labels are point-wise and need to be reshaped to match outputs
                            labels = labels.unsqueeze(1).unsqueeze(2)  # Make 3D: B x 1 x 1
                    else:
                        # If it's a tuple/list, try to extract all needed elements
                        # This depends on exactly what your data loader returns
                        print(f"WARNING: Expected dict for KPConv data but got {type(batch_data)}")
                        continue  # Skip this batch
                else:
                    # Non-KPConv path
                    if isinstance(batch_data, dict):
                        feats = batch_data['input_feature'].to(self.device)
                        labels = batch_data['labels'].to(self.device)
                    else:
                        feats, labels, _ = batch_data
                        feats = feats.to(self.device)
                        labels = labels.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(feats)
                
                # Calculate loss
                criterion = nn.CrossEntropyLoss(weight=self.cls_weight, ignore_index=0)
                loss = criterion(outputs, labels)
                
                # Update metrics
                val_loss += loss.item()
                
                # Get predictions
                pred = outputs.argmax(dim=1)
                
                # Store predictions and labels for IoU calculation
                mask = labels != 0
                all_preds.append(pred.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                if batch_idx == 0:  # Only print detailed error for first batch
                    print(f"Batch data keys: {batch_data.keys() if isinstance(batch_data, dict) else 'Not a dict'}")
                    break
        
        # Calculate mean IoU
        val_loss /= len(self.val_loader)
        miou = self._calculate_iou(all_preds, all_labels)
        
        # Log validation metrics
        self.writer.add_scalar('val/loss', val_loss, epoch)
        self.writer.add_scalar('val/miou', miou, epoch)
        
        print(f"Validation - Loss: {val_loss:.4f}, mIoU: {miou:.4f}")
        
        return val_loss, miou
    
    def _calculate_iou(self, preds, labels):
        """Calculate mean IoU"""
        n_classes = self.config.get('n_classes', 20)
        
        # Initialize confusion matrix
        conf_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
        
        # Fill confusion matrix
        for pred, label in zip(preds, labels):
            pred = pred.flatten()
            label = label.flatten()
            
            mask = (label != 0)  # Ignore index 0
            pred = pred[mask]
            label = label[mask]
            
            # Add to confusion matrix
            for i in range(len(pred)):
                conf_matrix[label[i], pred[i]] += 1
        
        # Calculate IoU for each class
        iou = np.zeros(n_classes)
        for i in range(1, n_classes):  # Skip class 0 (ignored)
            tp = conf_matrix[i, i]
            fp = conf_matrix[:, i].sum() - tp
            fn = conf_matrix[i, :].sum() - tp
            
            iou[i] = tp / max(tp + fp + fn, 1)
        
        # Calculate mean IoU (excluding class 0)
        miou = np.mean(iou[1:])
        
        return miou
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        
        # Skip to validation only if specified
        if self.args.val_only:
            self.validate(0)
            return
        
        for epoch in range(self.current_epoch, self.config.get('n_epochs', 60)):
            # Train one epoch
            train_loss, train_acc = self.train_one_epoch(epoch)
            
            # Validate
            if (epoch + 1) % self.config.get('val_frequency', 1) == 0:
                val_loss, miou = self.validate(epoch)
                
                # Save checkpoint if this is the best model
                if miou > self.best_miou:
                    self.best_miou = miou
                    self._save_checkpoint(epoch, is_best=True)
                    print(f"New best mIoU: {miou:.4f}")
            
            # Always save the latest checkpoint
            self._save_checkpoint(epoch)
            
            # Log epoch results
            self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
            self.writer.add_scalar('epoch/train_acc', train_acc, epoch)
            
            print(f"Epoch {epoch+1}/{self.config.get('n_epochs', 60)} completed. Best mIoU: {self.best_miou:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RangeViT Trainer')
    
    # Required arguments
    parser.add_argument('config_path', type=str, help='Path to the configuration YAML file')
    parser.add_argument('--data_root', type=str, required=True, help='Path to the SemanticKITTI dataset')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save logs and checkpoints')
    
    # Optional arguments
    parser.add_argument('--pretrained_model', type=str, default=None, help='Path to pretrained model')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint for resuming training')
    parser.add_argument('--val_only', action='store_true', help='Run validation only')
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    
    # Distributed training arguments
    parser.add_argument('--distributed', action='store_true', help='Use distributed training')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    
    # Add GPU-related arguments
    parser.add_argument('--gpu', type=int, default=0, help='Specific GPU to use')
    
    args = parser.parse_args()
    
    # Set specific GPU if requested
    if torch.cuda.is_available() and args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
    
    # Enable AMP by default on GPU
    if torch.cuda.is_available() and not args.use_amp:
        print("Enabling automatic mixed precision for better GPU performance")
        args.use_amp = True
    
    # Start training
    trainer = RangeViTTrainer(args)
    trainer.train()
