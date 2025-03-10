import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import matplotlib.pyplot as plt
import clip
from PIL import Image
import cv2
from torchvision import transforms
from dataset import ReferDataset, ReferDataModule
from refer import REFER
from affordance_clip import CLIPWrapper, FPN, AffordanceCLIP, AffordanceCLIPModule, pixel_text_contrastive_loss

def overlay_heatmap(image, heatmap, alpha=0.6):
    """Overlay a heatmap on an image"""
    # Convert heatmap to colormap
    heatmap_np = heatmap.cpu().numpy()
    heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)
    heatmap_np = (heatmap_np * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Convert image to numpy if it's a PIL image
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # Resize heatmap to match image size if needed
    if heatmap_color.shape[:2] != image_np.shape[:2]:
        heatmap_color = cv2.resize(heatmap_color, (image_np.shape[1], image_np.shape[0]))
    
    # Overlay heatmap on image
    overlay = cv2.addWeighted(image_np, 1-alpha, heatmap_color, alpha, 0)
    return overlay

def load_model(checkpoint_path, device):
    """Load the trained AffordanceCLIP model"""
    # Load CLIP model
    clip_model, preprocess = clip.load("RN101", device=device)
    
    # Create feature extractor and wrapper
    from train import DummyFeatureExtractor
    feature_extractor = DummyFeatureExtractor(clip_model).to(device)
    clip_wrapper = CLIPWrapper(clip_model, feature_extractor).to(device)
    
    # Define FPN parameters
    in_channels_list = [512, 1024, 2048]  # example channels for F1, F2, F3
    clip_dim = 512  # CLIP RN101 output dimension
    mid_channels = 256
    out_channels = clip_dim
    
    # Create FPN
    fpn = FPN(clip_dim=clip_dim, 
              in_channels_list=in_channels_list, 
              mid_channels=mid_channels, 
              out_channels=out_channels).to(device)
    
    # Create the full model
    model = AffordanceCLIPModule(clip_model=clip_wrapper, fpn=fpn)
    
    # Load the trained weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    return model, preprocess

def undo_clip_normalization(processed_image):
    processed_image = processed_image.squeeze(0).cpu().numpy()
    mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
    std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)
    processed_image = processed_image * std + mean
    processed_image = np.clip(processed_image, 0, 1)
    processed_image = (processed_image * 255).astype(np.uint8)
    processed_image = processed_image.transpose(1, 2, 0)
    processed_image = Image.fromarray(processed_image)
    return processed_image


def load_image(index):
    import numpy as np
    import clip
    import matplotlib.pyplot as plt
    model, preprocess = clip.load("RN101", device="cpu")
    refer = REFER("/cluster/scratch/andrye/proj_data", dataset="refcocog", splitBy="umd")
    # refer = REFER("/cluster/scratch/andrye/proj_data", dataset="refcoco", splitBy="unc")
    # refer = REFER("/cluster/scratch/andrye/proj_data", dataset="refcoco+", splitBy="unc")

    
    dataset = ReferDataset(refer, split='train',preprocess=preprocess)
    image, sentence, target = dataset[index]
    
    return image, sentence, target
    
    
def main():
    parser = argparse.ArgumentParser(description="Test AffordanceCLIP model on custom images")
    parser.add_argument("--index", type=int, help="Index of the image to test. If provided, --image_path and --action are ignored.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--image_path", type=str, help="Path to the test image")
    parser.add_argument("--action", type=str, help="Action prompt for affordance (e.g., 'drink from', 'cut with')")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save visualization results")
    parser.add_argument("--alpha", type=float, default=0.6, help="Transparency of heatmap overlay (0-1)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model, preprocess = load_model(args.checkpoint, device)
    
    target = None
    
    # Load and preprocess image
    if args.index is not None:
        print(f"Processing image: {args.index}")
        image, sentence, target = load_image(args.index)
        image = image.unsqueeze(0).to(device)
        sentence = sentence.unsqueeze(0).to(device)
    else:
        print(f"Processing image: {args.image_path}")
        image = Image.open(args.image_path).convert("RGB")
        image = preprocess(image).unsqueeze(0).to(device)
        sentence = clip.tokenize([args.action]).to(device)
    
    # Get model prediction
    print(f"Generating affordance prediction")
    
    with torch.no_grad():
        activation = model(image, sentence)
        
        if target is not None:
            # Create side-by-side comparison
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            
            # Plot activation
            axes[0].imshow(activation.cpu().squeeze(0).numpy())
            axes[0].set_title('Predicted Activation')
            
            # Plot target
            axes[1].imshow(target.cpu().squeeze(0).numpy())
            axes[1].set_title('Ground Truth')
            
            plt.tight_layout()
        else:
            # Just plot activation alone
            plt.figure(figsize=(10, 10))
            plt.imshow(activation.cpu().squeeze(0).numpy())
            plt.axis('off')
            plt.tight_layout()
            
        plt.savefig(os.path.join(args.output_dir, f"activation.jpg"))
        
    
    # # Resize activation to match original image size
    # activation_resized = torch.nn.functional.interpolate(
    #     activation.unsqueeze(1),
    #     size=(image.height, image.width),
    #     mode='bilinear',
    #     align_corners=False
    # ).squeeze(1).squeeze(0)
    
    # # Create visualization
    # overlay = overlay_heatmap(image, activation_resized, alpha=args.alpha)
    
    # # Save the original image, heatmap, and overlay
    # base_filename = os.path.splitext(os.path.basename(args.image_path))[0]
    # action_str = args.action.replace(" ", "_")
    
    # # Save processed image

    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.savefig(os.path.join(args.output_dir, f"{base_filename}_original.jpg"))
    
    # # Save heatmap
    # plt.figure(figsize=(10, 10))
    # plt.imshow(activation_resized.cpu(), cmap='jet')
    # plt.axis('off')
    # plt.tight_layout()
    # plt.savefig(os.path.join(args.output_dir, f"{base_filename}_{action_str}_heatmap.jpg"))
    
    # # Save overlay
    # plt.figure(figsize=(10, 10))
    # plt.imshow(overlay)
    # plt.axis('off')
    # plt.title(f"Action: '{args.action}'")
    # plt.tight_layout()
    # plt.savefig(os.path.join(args.output_dir, f"{base_filename}_{action_str}_overlay.jpg"))

if __name__ == "__main__":
    main()
