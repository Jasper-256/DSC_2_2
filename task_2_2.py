import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import ToPILImage
import os
from pathlib import Path
from tqdm import tqdm
from IPython.display import display
import torchmetrics
from torchmetrics.image import StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity, PeakSignalNoiseRatio

# Hyperparameters for Task 2.2: Modal RGB Content -> Amodal RGB Content
N_FRAMES = 3
EPOCHS = 20
IMAGES_TO_SAVE = 100
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
SLIDING_WINDOW_STEP = 3
TEST_EPOCH_FREQ = 4  # Test on test set every 4 epochs

def get_img_dict(img_dir):
    """Get dictionary of image files organized by type"""
    img_files = [x for x in img_dir.iterdir() if x.name.endswith('.png') or x.name.endswith('.tiff')]
    img_files.sort()

    img_dict = {}
    for img_file in img_files:
        img_type = img_file.name.split('_')[0]
        if img_type not in img_dict:
            img_dict[img_type] = []
        img_dict[img_type].append(img_file)
    return img_dict


def get_sample_dict(sample_dir):
    """Get complete dictionary structure for a sample directory"""
    camera_dirs = [x for x in sample_dir.iterdir() if 'camera' in x.name]
    camera_dirs.sort()
    
    sample_dict = {}

    for cam_dir in camera_dirs:
        cam_dict = {}
        cam_dict['scene'] = get_img_dict(cam_dir)

        obj_dirs = [x for x in cam_dir.iterdir() if 'obj_' in x.name]
        obj_dirs.sort()
        
        for obj_dir in obj_dirs:
            cam_dict[obj_dir.name] = get_img_dict(obj_dir)

        sample_dict[cam_dir.name] = cam_dict

    return sample_dict


class MultiSceneDataset(Dataset):
    def __init__(self, data_root, split='train', transform=None, verbose=False, save_plots=False, plot_save_dir='./plots', n_frames=N_FRAMES):
        """
        Dataset for loading RGB images with amodal masks and amodal RGB content from multiple scene folders.
        Modified for Task 2.2 to handle video sequences: Modal RGB Content -> Amodal RGB Content.

        Args:
            data_root (str): Root directory containing train and test folders
            split (str): Either 'train' or 'test'
            transform (callable, optional): Transform to apply to RGB images
            verbose (bool): If True, plot images during iteration
            save_plots (bool): If True, save plots instead of displaying them
            plot_save_dir (str): Directory to save plots when save_plots=True
            n_frames (int): Number of consecutive frames to load as a sequence
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.verbose = verbose
        self.save_plots = save_plots
        self.plot_save_dir = plot_save_dir
        self.n_frames = n_frames
        self.data_samples = []

        # Create plot directory if saving plots
        if self.save_plots:
            os.makedirs(self.plot_save_dir, exist_ok=True)

        # Get the split directory (train or test)
        split_dir = self.data_root / split
        if not split_dir.exists():
            raise ValueError(f"Split directory {split_dir} does not exist")

        print(f"Loading {split} data from {split_dir}")
        
        # Get all scene directories in the split
        scene_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        scene_dirs.sort()
        
        print(f"Found {len(scene_dirs)} scene directories in {split} split")

        # Track filtering statistics
        total_sequences_checked = 0
        filtered_sequences = 0

        # Process each scene
        for scene_dir in scene_dirs:
            print(f"Processing scene: {scene_dir.name}")
            sample_dict = get_sample_dict(scene_dir)
            
            # For each camera in the scene
            for cam_name, cam_data in sample_dict.items():
                # For each object in the camera
                obj_names = [name for name in cam_data.keys() if name.startswith('obj_')]
                
                for obj_name in obj_names:
                    obj_id = int(obj_name.split('_')[1])
                    
                    # For video sequences: we need at least 2*n_frames+1 consecutive frames
                    # where n_frames before + center frame + n_frames after
                    num_frames = len(cam_data['scene']['rgba'])
                    total_frames_needed = 2 * self.n_frames + 1
                    
                    # Create sequences centered around target frames
                    # Center frame can range from n_frames to num_frames-n_frames-1
                    min_center_frame = self.n_frames
                    max_center_frame = num_frames - self.n_frames - 1
                    
                    for center_frame in range(min_center_frame, max_center_frame + 1, SLIDING_WINDOW_STEP):
                        if center_frame - self.n_frames < 0 or center_frame + self.n_frames >= num_frames:
                            break
                            
                        try:
                            total_sequences_checked += 1
                            
                            # Collect paths for this sequence
                            sequence_data = {
                                'rgb_paths': [],
                                'amodal_mask_paths': [],
                                'amodal_rgb_paths': [],
                                'obj_id': obj_id,
                                'scene_name': scene_dir.name,
                                'camera_name': cam_name,
                                'object_name': obj_name,
                                'center_frame': center_frame
                            }
                            
                            # Collect paths for all frames in sequence (n_frames before + center + n_frames after)
                            start_frame = center_frame - self.n_frames
                            end_frame = center_frame + self.n_frames + 1
                            for frame_idx in range(start_frame, end_frame):
                                rgb_path = cam_data['scene']['rgba'][frame_idx]
                                amodal_mask_path = cam_data[obj_name]['segmentation'][frame_idx]
                                amodal_rgb_path = cam_data[obj_name]['rgba'][frame_idx]
                                
                                sequence_data['rgb_paths'].append(str(rgb_path))
                                sequence_data['amodal_mask_paths'].append(str(amodal_mask_path))
                                sequence_data['amodal_rgb_paths'].append(str(amodal_rgb_path))
                            
                            # Check if center frame has valid amodal content (not all transparent) and valid amodal mask
                            center_amodal_rgb_path = sequence_data['amodal_rgb_paths'][self.n_frames]  # Center frame is at index n_frames
                            center_amodal_mask_path = sequence_data['amodal_mask_paths'][self.n_frames]  # Center frame mask at index n_frames
                            if self._is_valid_center_frame(center_amodal_rgb_path, center_amodal_mask_path, obj_id):
                                self.data_samples.append(sequence_data)
                            else:
                                filtered_sequences += 1
                                
                        except (IndexError, KeyError) as e:
                            # Skip if files are missing
                            continue

        print(f"Total video sequences checked: {total_sequences_checked}")
        print(f"Filtered out sequences (empty amodal content): {filtered_sequences}")
        print(f"Valid video sequences in {split} split: {len(self.data_samples)}")

    def _is_valid_center_frame(self, amodal_rgb_path, amodal_mask_path, obj_id):
        """
        Check if the center frame has valid amodal RGB content and amodal mask (not all black).
        
        Args:
            amodal_rgb_path (str): Path to amodal RGB file for center frame
            amodal_mask_path (str): Path to amodal mask file for center frame
            obj_id (int): Object ID (for consistency, not used in RGB validation)
            
        Returns:
            bool: True if center frame has valid amodal RGB content and mask, False otherwise
        """
        try:
            # Load amodal RGB content
            rgb_img = Image.open(amodal_rgb_path).convert('RGB')
            rgb_np = np.array(rgb_img)
            
            # Check if there are any non-black pixels (sum of RGB values > threshold)
            # A completely black image would indicate no content
            rgb_sum = np.sum(rgb_np)
            
            # Load amodal mask
            mask_img = Image.open(amodal_mask_path)
            mask_np = np.array(mask_img)
            
            # Check if mask has any non-zero pixels
            # A completely black mask would indicate no object presence
            mask_sum = np.sum(mask_np)
            
            # Both RGB content and mask should have meaningful content
            # Threshold: if sum is very low, consider it empty content
            rgb_valid = rgb_sum > 1000  # Arbitrary threshold - adjust as needed
            mask_valid = mask_sum > 100  # Threshold for mask (lower since it's binary-ish)
            
            return rgb_valid and mask_valid
                
        except Exception as e:
            # If we can't load the RGB content or mask, consider it invalid
            return False

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        sample_info = self.data_samples[idx]

        # Load sequence of RGB images (2*n_frames+1 total)
        rgb_sequences = []
        amodal_mask_sequences = []
        # Load amodal RGB content for all frames (for visualization), but only use center for training
        amodal_rgb_sequences = []
        total_frames = 2 * self.n_frames + 1

        # Get target size from transform or use original size
        target_size = None
        if self.transform:
            # Extract target size from transforms
            for t in self.transform.transforms:
                if isinstance(t, transforms.Resize):
                    if isinstance(t.size, int):
                        target_size = (t.size, t.size)
                    elif isinstance(t.size, (list, tuple)) and len(t.size) == 2:
                        target_size = (int(t.size[0]), int(t.size[1]))
                    break

        # Load all frames in the sequence (2*n_frames+1 total frames)
        for seq_idx in range(total_frames):
            # Load RGB image
            rgb_image = Image.open(sample_info['rgb_paths'][seq_idx]).convert('RGB')
            
            if target_size is None:
                target_size = rgb_image.size[::-1]  # (H, W)
            
            # Ensure target_size is a valid tuple of ints
            if not isinstance(target_size, tuple) or len(target_size) != 2:
                target_size = rgb_image.size[::-1]  # Fall back to original size
            target_size = (int(target_size[0]), int(target_size[1]))

            # Load and process amodal mask for all frames
            amodal_mask_img = Image.open(sample_info['amodal_mask_paths'][seq_idx])
            amodal_mask_np = np.array(amodal_mask_img) / 255.0  # Convert 0,255 to 0,1

            # Resize amodal mask to match image size
            amodal_mask_pil = Image.fromarray((amodal_mask_np * 255).astype(np.uint8))
            amodal_mask_resized = amodal_mask_pil.resize((target_size[1], target_size[0]), Image.Resampling.NEAREST)
            amodal_mask = torch.from_numpy(np.array(amodal_mask_resized) / 255.0).float()

            # Load and process amodal RGB content for all frames
            amodal_rgb_img = Image.open(sample_info['amodal_rgb_paths'][seq_idx]).convert('RGB')
            amodal_rgb_np = np.array(amodal_rgb_img) / 255.0  # Convert 0,255 to 0,1

            # Resize amodal RGB content to match image size
            amodal_rgb_pil = Image.fromarray((amodal_rgb_np * 255).astype(np.uint8))
            amodal_rgb_resized = amodal_rgb_pil.resize((target_size[1], target_size[0]), Image.Resampling.NEAREST)
            amodal_rgb_np_resized = np.array(amodal_rgb_resized) / 255.0
            amodal_rgb = torch.from_numpy(amodal_rgb_np_resized).float()  # RGB content

            # Apply transforms to RGB image
            if self.transform:
                rgb_image = self.transform(rgb_image)
            else:
                rgb_image = transforms.ToTensor()(rgb_image)

            rgb_sequences.append(rgb_image)
            amodal_mask_sequences.append(amodal_mask)
            amodal_rgb_sequences.append(amodal_rgb)

        # Stack sequences: (T, C, H, W) where T is temporal dimension
        rgb_sequence = torch.stack(rgb_sequences, dim=0)  # (2*N_FRAMES+1, 3, H, W)
        amodal_mask_sequence = torch.stack(amodal_mask_sequences, dim=0)  # (2*N_FRAMES+1, H, W)
        amodal_rgb_sequence_full = torch.stack(amodal_rgb_sequences, dim=0)  # (2*N_FRAMES+1, 3, H, W)
        
        # Extract only center frame amodal RGB for training (but return full sequence for visualization)
        center_amodal_rgb = amodal_rgb_sequence_full[self.n_frames:self.n_frames+1]  # (1, 3, H, W)
        
        # Ensure target_size is valid
        if target_size is None:
            target_size = (256, 256)  # Default fallback size

        # Verbose plotting (only plot the center frame of sequence for now)
        if self.verbose:
            if self.save_plots:
                self.save_sample_plot(rgb_sequence[self.n_frames], amodal_mask_sequence[self.n_frames], amodal_rgb_sequence_full[self.n_frames], idx)
            else:
                self.plot_sample(rgb_sequence[self.n_frames], amodal_mask_sequence[self.n_frames], amodal_rgb_sequence_full[self.n_frames], idx)

        # Return: full RGB and amodal mask sequences, center frame amodal RGB for training, full amodal RGB sequence for visualization
        return rgb_sequence, amodal_mask_sequence, center_amodal_rgb, amodal_rgb_sequence_full

    def save_sample_plot(self, rgb_image, amodal_mask, amodal_rgb_content, idx):
        """Save RGB image, amodal mask, and amodal RGB content in subplots to file"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot RGB image
        if rgb_image.shape[0] == 3:  # If tensor is CxHxW
            rgb_np = rgb_image.permute(1, 2, 0).numpy()
        else:
            rgb_np = rgb_image.numpy()

        axes[0].imshow(rgb_np)
        axes[0].set_title(f'RGB Image (Sample {idx})')
        axes[0].axis('off')

        # Plot amodal mask
        axes[1].imshow(amodal_mask.numpy(), cmap='gray')
        axes[1].set_title(f'Amodal Mask (Obj ID: {self.data_samples[idx]["obj_id"]})')
        axes[1].axis('off')

        # Plot amodal RGB content
        if amodal_rgb_content.shape[0] == 3:  # If tensor is CxHxW
            amodal_rgb_np = amodal_rgb_content.permute(1, 2, 0).numpy()
        else:
            amodal_rgb_np = amodal_rgb_content.numpy()
        axes[2].imshow(amodal_rgb_np)
        axes[2].set_title('Amodal RGB Content')
        axes[2].axis('off')

        plt.tight_layout()

        # Save the plot
        save_path = os.path.join(self.plot_save_dir, f'sample_{idx:04d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  # Close to prevent memory leaks
        print(f"Plot saved: {save_path}")

    def plot_sample(self, rgb_image, amodal_mask, amodal_rgb_content, idx):
        """Plot RGB image, amodal mask, and amodal RGB content in subplots"""
        matplotlib.use('Agg')  # Use non-interactive backend temporarily

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot RGB image
        if rgb_image.shape[0] == 3:  # If tensor is CxHxW
            rgb_np = rgb_image.permute(1, 2, 0).numpy()
        else:
            rgb_np = rgb_image.numpy()

        axes[0].imshow(rgb_np)
        axes[0].set_title(f'RGB Image (Sample {idx})')
        axes[0].axis('off')

        # Plot amodal mask
        axes[1].imshow(amodal_mask.numpy(), cmap='gray')
        axes[1].set_title(f'Amodal Mask (Obj ID: {self.data_samples[idx]["obj_id"]})')
        axes[1].axis('off')

        # Plot amodal RGB content
        if amodal_rgb_content.shape[0] == 3:  # If tensor is CxHxW
            amodal_rgb_np = amodal_rgb_content.permute(1, 2, 0).numpy()
        else:
            amodal_rgb_np = amodal_rgb_content.numpy()
        axes[2].imshow(amodal_rgb_np)
        axes[2].set_title('Amodal RGB Content')
        axes[2].axis('off')

        plt.tight_layout()

        # Force display in Jupyter
        display(fig)
        plt.close(fig)  # Close to prevent memory leaks


def get_transforms(image_size):
    """
    Get basic transformations without normalization

    Args:
        image_size (int): Target image size for resizing

    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])


# Set random seed for reproducible results
torch.manual_seed(42)
np.random.seed(42)

# Create datasets for train and test
print("Creating video datasets for Task 2.2 (Modal RGB -> Amodal RGB)...")
data_root = './'  # Current directory containing train and test folders
transform = get_transforms(image_size=256)

# Create datasets first - now using video sequences for RGB content prediction
train_dataset = MultiSceneDataset(
    data_root=data_root,
    split='train',
    transform=transform,
    verbose=False,
    save_plots=False,
    plot_save_dir='./plots',
    n_frames=N_FRAMES
)

test_dataset = MultiSceneDataset(
    data_root=data_root,
    split='test',
    transform=transform,
    verbose=False,
    save_plots=False,
    plot_save_dir='./plots',
    n_frames=N_FRAMES
)

# Create dataloaders
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,  # Set to 0 for debugging
    pin_memory=torch.cuda.is_available()
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print(f"Train batches: {len(train_dataloader)}")
print(f"Test batches: {len(test_dataloader)}")
print("-" * 50)

# Test each dataloader
print("\nTesting Train dataloader:")
for batch_idx, (images, amodal_masks, amodal_rgba_center, amodal_rgba_full) in enumerate(train_dataloader):
    print(f"  Batch {batch_idx}:")
    print(f"    Images shape: {images.shape}")
    print(f"    Amodal masks shape: {amodal_masks.shape}")
    print(f"    Amodal RGBA (center) shape: {amodal_rgba_center.shape}")
    print(f"    Amodal RGBA (full) shape: {amodal_rgba_full.shape}")
    print(f"    Images dtype: {images.dtype}")
    print(f"    Amodal masks dtype: {amodal_masks.dtype}")
    print(f"    Amodal RGBA dtype: {amodal_rgba_center.dtype}")
    # Only test first batch
    break

print("\nTesting Test dataloader:")
for batch_idx, (images, amodal_masks, amodal_rgba_center, amodal_rgba_full) in enumerate(test_dataloader):
    print(f"  Batch {batch_idx}:")
    print(f"    Images shape: {images.shape}")
    print(f"    Amodal masks shape: {amodal_masks.shape}")
    print(f"    Amodal RGBA (center) shape: {amodal_rgba_center.shape}")
    print(f"    Amodal RGBA (full) shape: {amodal_rgba_full.shape}")
    print(f"    Images dtype: {images.dtype}")
    print(f"    Amodal masks dtype: {amodal_masks.dtype}")
    print(f"    Amodal RGBA dtype: {amodal_rgba_center.dtype}")
    # Only test first batch
    break

print("-" * 50)
print("Dataloader test completed!")


# Lightweight 3D U-Net for Task 2.2: Video-based Modal RGB Content -> Amodal RGB Content (Center Frame Prediction)
class Lightweight3DUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, n_frames=N_FRAMES):
        super(Lightweight3DUNet, self).__init__()
        self.n_frames = n_frames

        # Much smaller channel sizes for lightweight model
        # Encoder (downsampling path) - using 3D convolutions
        self.enc1 = self.conv3d_block(in_channels, 16)
        self.enc2 = self.conv3d_block(16, 32)
        self.enc3 = self.conv3d_block(32, 64)

        # Bottleneck (smaller)
        self.bottleneck = self.conv3d_block(64, 128)

        # Decoder (upsampling path) - using 3D transpose convolutions
        self.upconv3 = nn.ConvTranspose3d(128, 64, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec3 = self.conv3d_block(128, 64)  # 64 + 64 from skip connection

        self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec2 = self.conv3d_block(64, 32)   # 32 + 32 from skip connection

        self.upconv1 = nn.ConvTranspose3d(32, 16, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = self.conv3d_block(32, 16)   # 16 + 16 from skip connection

        # Final layer
        self.final = nn.Conv3d(16, out_channels, kernel_size=1)

        # Max pooling (only spatial dimensions, preserve temporal)
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def conv3d_block(self, in_channels, out_channels):
        """Single 3D convolution with ReLU and batch norm"""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Input shape: (B, C, T, H, W) where T is temporal dimension (2*n_frames+1)
        
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc3))

        # Decoder with skip connections
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        # Final output for all frames
        output = self.final(dec1)  # (B, 3, T, H, W)
        
        # Extract only the center frame (index n_frames in temporal dimension)
        center_frame_output = output[:, :, self.n_frames:self.n_frames+1, :, :]  # (B, 3, 1, H, W)
        
        return center_frame_output

# Test the 3D model
print("Testing Lightweight 3D U-Net model for Task 2.2...")
test_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_3d = Lightweight3DUNet(in_channels=4, out_channels=3, n_frames=N_FRAMES).to(test_device)
total_frames = 2 * N_FRAMES + 1
test_input_3d = torch.randn(1, 4, total_frames, 256, 256).to(test_device)  # Batch size 1, 4 channels (RGB + amodal mask), total_frames temporal, 256x256
test_output_3d = model_3d(test_input_3d)
print(f"Input shape: {test_input_3d.shape}")
print(f"Output shape: {test_output_3d.shape}")
print(f"Total frames processed: {total_frames} (2*{N_FRAMES}+1)")
print(f"Predicting only center frame (index {N_FRAMES})")

# Count parameters
total_params_3d = sum(p.numel() for p in model_3d.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params_3d:,}")
print("3D model created successfully for video processing (RGB content prediction)!")


# Training utilities - Using TorchMetrics for RGB content evaluation
def create_metrics(device):
    """Create TorchMetrics objects for RGB content evaluation"""
    return {
        'mse': torchmetrics.MeanSquaredError().to(device),
        'mae': torchmetrics.MeanAbsoluteError().to(device),
        'psnr': PeakSignalNoiseRatio().to(device),
        'ssim': StructuralSimilarityIndexMeasure(data_range=1.0).to(device),
        'lpips': LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)
    }

def calculate_metrics(predictions, targets, metrics_dict):
    """Calculate all RGB content metrics using TorchMetrics"""
    # No sigmoid needed for RGB regression - predictions should be in [0,1] range
    predictions = torch.clamp(predictions, 0, 1)  # Ensure valid range
    targets = torch.clamp(targets, 0, 1)  # Ensure targets are also in valid range
    
    # Reshape for metric computation if needed
    # Most metrics expect (B, C, H, W) format
    if predictions.dim() == 5:  # (B, C, 1, H, W) -> (B, C, H, W)
        predictions = predictions.squeeze(2)
    if targets.dim() == 5:  # (B, C, 1, H, W) -> (B, C, H, W)
        targets = targets.squeeze(2)
    
    # Calculate basic metrics
    mse = metrics_dict['mse'](predictions, targets)
    mae = metrics_dict['mae'](predictions, targets)
    psnr = metrics_dict['psnr'](predictions, targets)
    
    # Calculate SSIM
    ssim = metrics_dict['ssim'](predictions, targets)
    
    # Calculate LPIPS (expects images in [0,1] range)
    lpips = metrics_dict['lpips'](predictions, targets)
    
    return mse.item(), mae.item(), psnr.item(), ssim.item(), lpips.item()

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch with video sequences for RGB content prediction"""
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    num_batches = 0
    
    # Create metrics for this epoch
    metrics = create_metrics(device)

    for rgb_sequences, amodal_mask_sequences, amodal_rgb_sequences, amodal_rgb_sequences_full in tqdm(dataloader, desc="Training"):
        # Move data to device
        # rgb_sequences: (B, 2*N_FRAMES+1, C, H, W) -> need to transpose to (B, C, 2*N_FRAMES+1, H, W)
        # amodal_mask_sequences: (B, 2*N_FRAMES+1, H, W) -> need to add channel dim and transpose to (B, 1, 2*N_FRAMES+1, H, W)  
        # amodal_rgb_sequences: (B, 1, 3, H, W) -> need to add temporal dim to (B, 3, 1, H, W)
        
        rgb_sequences = rgb_sequences.permute(0, 2, 1, 3, 4).contiguous().to(device)  # (B, C, 2*N_FRAMES+1, H, W)
        amodal_mask_sequences = amodal_mask_sequences.unsqueeze(2).permute(0, 2, 1, 3, 4).contiguous().to(device)  # (B, 1, 2*N_FRAMES+1, H, W)
        amodal_rgb_sequences = amodal_rgb_sequences.permute(0, 4, 1, 2, 3).contiguous().to(device)  # (B, 3, 1, H, W) - fix permutation

        # Combine RGB and amodal mask as input (4 channels total: 3 RGB + 1 amodal mask)
        inputs = torch.cat([rgb_sequences, amodal_mask_sequences], dim=1)  # (B, 4, 2*N_FRAMES+1, H, W)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)  # (B, 3, 1, H, W) - only center frame
        loss = criterion(outputs, amodal_rgb_sequences)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate metrics using TorchMetrics
        mse, mae, psnr, ssim, lpips = calculate_metrics(outputs.contiguous(), amodal_rgb_sequences.contiguous(), metrics)
        total_loss += loss.item()
        total_mse += mse
        total_mae += mae
        total_psnr += psnr
        total_ssim += ssim
        total_lpips += lpips
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_mse = total_mse / num_batches
    avg_mae = total_mae / num_batches
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    avg_lpips = total_lpips / num_batches

    return avg_loss, avg_mse, avg_mae, avg_psnr, avg_ssim, avg_lpips

def test_epoch(model, dataloader, criterion, device):
    """Test the model for one epoch with video sequences for RGB content prediction"""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    num_batches = 0
    
    # Create metrics for this epoch
    metrics = create_metrics(device)

    with torch.no_grad():
        for rgb_sequences, amodal_mask_sequences, amodal_rgb_sequences, amodal_rgb_sequences_full in tqdm(dataloader, desc="Testing"):
            # Move data to device and rearrange dimensions
            rgb_sequences = rgb_sequences.permute(0, 2, 1, 3, 4).contiguous().to(device)  # (B, C, 2*N_FRAMES+1, H, W)
            amodal_mask_sequences = amodal_mask_sequences.unsqueeze(2).permute(0, 2, 1, 3, 4).contiguous().to(device)  # (B, 1, 2*N_FRAMES+1, H, W)
            amodal_rgb_sequences = amodal_rgb_sequences.permute(0, 4, 1, 2, 3).contiguous().to(device)  # (B, 3, 1, H, W) - fix permutation

            # Combine RGB and amodal mask as input
            inputs = torch.cat([rgb_sequences, amodal_mask_sequences], dim=1)  # (B, 4, 2*N_FRAMES+1, H, W)

            # Forward pass
            outputs = model(inputs)  # (B, 3, 1, H, W) - only center frame
            loss = criterion(outputs, amodal_rgb_sequences)

            # Calculate metrics using TorchMetrics
            mse, mae, psnr, ssim, lpips = calculate_metrics(outputs.contiguous(), amodal_rgb_sequences.contiguous(), metrics)
            total_loss += loss.item()
            total_mse += mse
            total_mae += mae
            total_psnr += psnr
            total_ssim += ssim
            total_lpips += lpips
            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_mse = total_mse / num_batches
    avg_mae = total_mae / num_batches
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    avg_lpips = total_lpips / num_batches

    return avg_loss, avg_mse, avg_mae, avg_psnr, avg_ssim, avg_lpips


print("Training utilities loaded successfully!")


# Training setup and execution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize model, loss, and optimizer for Task 2.2 (RGB content prediction)
model = Lightweight3DUNet(in_channels=4, out_channels=3, n_frames=N_FRAMES).to(device)
criterion = nn.MSELoss()  # MSE loss for RGB regression
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training parameters
num_epochs = EPOCHS
best_test_psnr = 0.0  # Use test PSNR as the best metric for RGB content
train_losses = []
train_mses = []
train_maes = []
train_psnrs = []
train_ssims = []
train_lpips = []

# Test metrics tracking (only recorded every TEST_EPOCH_FREQ epochs)
test_losses = []
test_mses = []
test_maes = []
test_psnrs = []
test_ssims = []
test_lpips = []
test_epochs = []  # Track which epochs we tested on

print(f"Starting training for {num_epochs} epochs with lightweight model for Task 2.2...")
print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print("-" * 50)

    # Training phase
    train_loss_val, train_mse_val, train_mae_val, train_psnr_val, train_ssim_val, train_lpips_val = train_epoch(model, train_dataloader, criterion, optimizer, device)
    train_losses.append(train_loss_val)
    train_mses.append(train_mse_val)
    train_maes.append(train_mae_val)
    train_psnrs.append(train_psnr_val)
    train_ssims.append(train_ssim_val)
    train_lpips.append(train_lpips_val)

    print(f"Train - Loss: {train_loss_val:.4f}, MSE: {train_mse_val:.4f}, MAE: {train_mae_val:.4f}, PSNR: {train_psnr_val:.4f}, SSIM: {train_ssim_val:.4f}, LPIPS: {train_lpips_val:.4f}")

    # Test phase (every TEST_EPOCH_FREQ epochs)
    should_test = (epoch + 1) % TEST_EPOCH_FREQ == 0 or (epoch + 1) == num_epochs
    if should_test:
        print(f"\n{'='*20} TESTING ON TEST SET {'='*20}")
        test_loss_val, test_mse_val, test_mae_val, test_psnr_val, test_ssim_val, test_lpips_val = test_epoch(model, test_dataloader, criterion, device)
        
        # Record test metrics
        test_losses.append(test_loss_val)
        test_mses.append(test_mse_val)
        test_maes.append(test_mae_val)
        test_psnrs.append(test_psnr_val)
        test_ssims.append(test_ssim_val)
        test_lpips.append(test_lpips_val)
        test_epochs.append(epoch + 1)

        # Print test results in a nicely formatted table
        print(f"\n{'Metric':<15} {'Train':<12} {'Test':<12} {'Difference':<12}")
        print("-" * 60)
        print(f"{'Loss (MSE)':<15} {train_loss_val:<12.4f} {test_loss_val:<12.4f} {abs(train_loss_val - test_loss_val):<12.4f}")
        print(f"{'MSE':<15} {train_mse_val:<12.4f} {test_mse_val:<12.4f} {abs(train_mse_val - test_mse_val):<12.4f}")
        print(f"{'MAE':<15} {train_mae_val:<12.4f} {test_mae_val:<12.4f} {abs(train_mae_val - test_mae_val):<12.4f}")
        print(f"{'PSNR (dB)':<15} {train_psnr_val:<12.4f} {test_psnr_val:<12.4f} {abs(train_psnr_val - test_psnr_val):<12.4f}")
        print(f"{'SSIM':<15} {train_ssim_val:<12.4f} {test_ssim_val:<12.4f} {abs(train_ssim_val - test_ssim_val):<12.4f}")
        print(f"{'LPIPS':<15} {train_lpips_val:<12.4f} {test_lpips_val:<12.4f} {abs(train_lpips_val - test_lpips_val):<12.4f}")
        print("-" * 60)

        # Save best model based on test PSNR (higher is better)
        if test_psnr_val > best_test_psnr:
            best_test_psnr = test_psnr_val
            torch.save(model.state_dict(), 'best_lightweight_3dunet_task2_2.pth')
            print(f"🎉 NEW BEST TEST PSNR: {best_test_psnr:.4f} - Model saved!")
        else:
            print(f"Current best test PSNR: {best_test_psnr:.4f}")
        
        print(f"{'='*60}")
    else:
        print(f"(Test evaluation will run on epoch {((epoch + 1) // TEST_EPOCH_FREQ + 1) * TEST_EPOCH_FREQ})")



print("\nTraining completed!")
print(f"Best test PSNR: {best_test_psnr:.4f}")

# Plot training and test curves for RGB content prediction
fig, axes = plt.subplots(3, 3, figsize=(18, 15))

# Loss
axes[0, 0].plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss (MSE)', marker='o')
if len(test_losses) > 0:
    axes[0, 0].plot(test_epochs, test_losses, label='Test Loss (MSE)', marker='s', linestyle='--')
axes[0, 0].set_title('Loss (MSE)')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

# MAE
axes[0, 1].plot(range(1, len(train_maes) + 1), train_maes, label='Train MAE', marker='o')
if len(test_maes) > 0:
    axes[0, 1].plot(test_epochs, test_maes, label='Test MAE', marker='s', linestyle='--')
axes[0, 1].set_title('Mean Absolute Error')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('MAE')
axes[0, 1].legend()
axes[0, 1].grid(True)

# PSNR
axes[0, 2].plot(range(1, len(train_psnrs) + 1), train_psnrs, label='Train PSNR', marker='o')
if len(test_psnrs) > 0:
    axes[0, 2].plot(test_epochs, test_psnrs, label='Test PSNR', marker='s', linestyle='--')
axes[0, 2].set_title('Peak Signal-to-Noise Ratio')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('PSNR (dB)')
axes[0, 2].legend()
axes[0, 2].grid(True)

# SSIM
axes[1, 0].plot(range(1, len(train_ssims) + 1), train_ssims, label='Train SSIM', marker='o')
if len(test_ssims) > 0:
    axes[1, 0].plot(test_epochs, test_ssims, label='Test SSIM', marker='s', linestyle='--')
axes[1, 0].set_title('Structural Similarity Index')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('SSIM')
axes[1, 0].legend()
axes[1, 0].grid(True)

# LPIPS
axes[1, 1].plot(range(1, len(train_lpips) + 1), train_lpips, label='Train LPIPS', marker='o')
if len(test_lpips) > 0:
    axes[1, 1].plot(test_epochs, test_lpips, label='Test LPIPS', marker='s', linestyle='--')
axes[1, 1].set_title('Learned Perceptual Image Patch Similarity')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('LPIPS (lower is better)')
axes[1, 1].legend()
axes[1, 1].grid(True)

# Hide this subplot since we removed FID
axes[1, 2].axis('off')

# Hide the bottom row subplots since we removed IS
axes[2, 0].axis('off')
axes[2, 1].axis('off')
axes[2, 2].axis('off')

plt.tight_layout()
plt.show()


# Final testing phase
print("=" * 50)
print("FINAL TESTING PHASE")
print("=" * 50)

# Load the best model
model.load_state_dict(torch.load('best_lightweight_3dunet_task2_2.pth'))
print("Loaded best 3D model from training")

# Final test on test set
final_test_loss, final_test_mse, final_test_mae, final_test_psnr, final_test_ssim, final_test_lpips = test_epoch(model, test_dataloader, criterion, device)

print(f"\nFinal Test Results:")
print(f"Test Loss: {final_test_loss:.4f}")
print(f"Test MSE: {final_test_mse:.4f}")
print(f"Test MAE: {final_test_mae:.4f}")
print(f"Test PSNR: {final_test_psnr:.4f}")
print(f"Test SSIM: {final_test_ssim:.4f}")
print(f"Test LPIPS: {final_test_lpips:.4f}")

# Create a comparison table
print(f"\nFinal Performance Summary:")
print(f"{'Split':<15} {'Loss':<10} {'MSE':<10} {'MAE':<10} {'PSNR':<10} {'SSIM':<10} {'LPIPS':<10}")
print("-" * 95)
print(f"{'Final Train':<15} {train_losses[-1]:<10.4f} {train_mses[-1]:<10.4f} {train_maes[-1]:<10.4f} {train_psnrs[-1]:<10.4f} {train_ssims[-1]:<10.4f} {train_lpips[-1]:<10.4f}")
if len(test_losses) > 0:
    print(f"{'Best Test':<15} {test_losses[-1]:<10.4f} {test_mses[-1]:<10.4f} {test_maes[-1]:<10.4f} {test_psnrs[-1]:<10.4f} {test_ssims[-1]:<10.4f} {test_lpips[-1]:<10.4f}")
print(f"{'Final Test':<15} {final_test_loss:<10.4f} {final_test_mse:<10.4f} {final_test_mae:<10.4f} {final_test_psnr:<10.4f} {final_test_ssim:<10.4f} {final_test_lpips:<10.4f}")

if len(test_epochs) > 0:
    print(f"\nTest Evaluation Summary:")
    print(f"- Test evaluations performed on epochs: {test_epochs}")
    print(f"- Best test PSNR achieved: {best_test_psnr:.4f}")
    print(f"- Total test evaluations: {len(test_epochs)}")

# Visualize predictions on test set - save multiple different video sequence examples  
total_frames = 2 * N_FRAMES + 1
center_frame_idx = N_FRAMES
print(f"\nSaving multiple organized video sequence visualizations for Task 2.2:")
print(f"  - Layout: {total_frames} rows (frames) x 4 columns (RGB | Amodal Mask | GT RGB | Pred RGB)")
print(f"  - Ground truth amodal RGB content shown for ALL frames")
print(f"  - RGB content predictions shown only on center frame (frame {center_frame_idx+1}/{total_frames})")
print(f"  - Processing {N_FRAMES} frames before + 1 center + {N_FRAMES} frames after = {total_frames} total frames")
print(f"  - Saving to test_video_predictions/ ...")

def save_multiple_predictions(model, dataloader, device, num_images=20, save_dir="test_predictions"):
    """Save multiple individual prediction images with organized temporal layout for RGB content - OPTIMIZED VERSION"""
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    saved_count = 0
    total_frames = 2 * N_FRAMES + 1
    center_frame_idx = N_FRAMES
    
    # Create progress bar for saving predictions
    pbar = tqdm(total=num_images, desc="Saving video predictions", unit="images")
    
    with torch.no_grad():
        for batch_idx, (rgb_sequences, amodal_mask_sequences, amodal_rgb_sequences, amodal_rgb_sequences_full) in enumerate(dataloader):
            if saved_count >= num_images:
                break
                
            batch_size = rgb_sequences.shape[0]
            
            # Rearrange dimensions and move to device
            rgb_sequences = rgb_sequences.permute(0, 2, 1, 3, 4).contiguous().to(device)  # (B, C, 2*N_FRAMES+1, H, W)
            amodal_mask_sequences = amodal_mask_sequences.unsqueeze(2).permute(0, 2, 1, 3, 4).contiguous().to(device)  # (B, 1, 2*N_FRAMES+1, H, W)
            amodal_rgb_sequences = amodal_rgb_sequences.permute(0, 4, 1, 2, 3).contiguous().to(device)  # (B, 3, 1, H, W) - center frame only, fix permutation
            amodal_rgb_sequences_full = amodal_rgb_sequences_full.permute(0, 1, 4, 2, 3).contiguous().to(device)  # (B, 2*N_FRAMES+1, 3, H, W) - all frames, fix permutation
            
            # Combine RGB and amodal mask as input
            inputs = torch.cat([rgb_sequences, amodal_mask_sequences], dim=1)
            
            # Get predictions
            outputs = model(inputs)  # (B, 3, 1, H, W) - only center frame (RGB content)
            predictions = torch.clamp(outputs, 0, 1)  # Clamp to [0,1] range for RGB
            
            rgb_vis = rgb_sequences.permute(0, 2, 1, 3, 4).contiguous().cpu()  # (B, 2*N_FRAMES+1, C, H, W)
            amodal_mask_vis = amodal_mask_sequences.squeeze(1).permute(0, 1, 2, 3).contiguous().cpu()  # (B, 2*N_FRAMES+1, H, W)
            amodal_rgb_vis_full = amodal_rgb_sequences_full.contiguous().cpu()  # (B, 2*N_FRAMES+1, 3, H, W) - all ground truth frames
            pred_center = predictions.squeeze(2).contiguous().cpu()  # (B, 3, H, W) - RGB prediction for center frame only
            
            # Save each sample in the batch as a separate image
            for i in range(batch_size):
                if saved_count >= num_images:
                    break
                
                fig, axes = plt.subplots(total_frames, 4, figsize=(12, 3*total_frames))
                if total_frames == 1:
                    axes = axes.reshape(1, -1)
                
                for frame_idx in range(total_frames):
                    # Column 0: RGB Image
                    img_np = rgb_vis[i, frame_idx].permute(1, 2, 0).numpy()
                    axes[frame_idx, 0].imshow(img_np)
                    axes[frame_idx, 0].set_title(f'RGB {frame_idx+1}', fontsize=10)  # Smaller font
                    axes[frame_idx, 0].axis('off')
                    
                    # Column 1: Amodal Mask
                    amodal_mask = amodal_mask_vis[i, frame_idx].numpy()
                    axes[frame_idx, 1].imshow(amodal_mask, cmap='gray')
                    axes[frame_idx, 1].set_title(f'Amodal Mask {frame_idx+1}', fontsize=10)
                    axes[frame_idx, 1].axis('off')
                    
                    # Column 2: Ground Truth Amodal RGB (only for center frame)
                    if frame_idx == center_frame_idx:  # Center frame
                        amodal_rgb_gt = amodal_rgb_vis_full[i, frame_idx].permute(1, 2, 0).numpy()
                        axes[frame_idx, 2].imshow(amodal_rgb_gt)
                        axes[frame_idx, 2].set_title('GT RGB', fontsize=10)
                        axes[frame_idx, 2].axis('off')
                    else:
                        # Empty column for non-center frames
                        axes[frame_idx, 2].axis('off')
                        axes[frame_idx, 2].set_title('', fontsize=10)
                    
                    # Column 3: Prediction (only on center frame)
                    if frame_idx == center_frame_idx:  # Center frame
                        # Predicted amodal RGB content
                        amodal_pred_rgb = pred_center[i].permute(1, 2, 0).numpy()
                        axes[frame_idx, 3].imshow(amodal_pred_rgb)
                        axes[frame_idx, 3].set_title('Predicted RGB', fontsize=10)
                        axes[frame_idx, 3].axis('off')
                    else:
                        # Empty column for non-center frames
                        axes[frame_idx, 3].axis('off')
                        axes[frame_idx, 3].set_title('', fontsize=10)
                
                fig.suptitle(f'Task 2.2 Sample {saved_count+1} - RGB Content Prediction', fontsize=12)
                plt.tight_layout(rect=(0, 0, 1, 0.97))  # Leave space for title at top
                
                save_path = os.path.join(save_dir, f'video_sequence_{saved_count:03d}.png')
                plt.savefig(save_path, dpi=150)
                plt.close(fig)
                
                saved_count += 1
                pbar.update(1)
    
    pbar.close()
    print(f"Saved {saved_count} optimized video sequence visualizations to {save_dir}/")

# Save prediction examples
save_multiple_predictions(
    model,
    test_dataloader,
    device,
    num_images=IMAGES_TO_SAVE,
    save_dir="test_video_predictions"
)

print("\n" + "=" * 50)
print("TASK 2.2 COMPLETED SUCCESSFULLY!")
print("=" * 50)
print(f"✓ Model Architecture: Lightweight 3D U-Net (Center Frame RGB Content Prediction)")
print(f"✓ Input: RGB Sequences ({2*N_FRAMES+1} frames, 3 channels) + Amodal Mask Sequences ({2*N_FRAMES+1} frames, 1 channel) = 4 channels")
print(f"✓ Output: Amodal RGB Content for Center Frame ONLY (1 frame, 3 channels)")
print(f"✓ Data Structure: Video sequences of {2*N_FRAMES+1} frames ({N_FRAMES} before + 1 center + {N_FRAMES} after)")
print(f"✓ Temporal Processing: 3D Convolutions for spatiotemporal feature learning")
print(f"✓ Prediction Target: Only center frame (frame {N_FRAMES+1} out of {2*N_FRAMES+1})")
print(f"✓ Best Test PSNR: {best_test_psnr:.4f}")
print(f"✓ Final Test MSE: {final_test_mse:.4f}")
print(f"✓ Final Test MAE: {final_test_mae:.4f}")
print(f"✓ Final Test PSNR: {final_test_psnr:.4f}")
print(f"✓ Final Test SSIM: {final_test_ssim:.4f}")
print(f"✓ Final Test LPIPS: {final_test_lpips:.4f}")
print(f"✓ Total Training Sequences: {len(train_dataset)}")
print(f"✓ Total Test Sequences: {len(test_dataset)}")
print(f"✓ Context Frames: {N_FRAMES} before + {N_FRAMES} after = {2*N_FRAMES} context frames")
print("=" * 50)
