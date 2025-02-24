import torch
import torch.nn as nn
import torch.nn.functional as F
import clip  # pip install git+https://github.com/openai/CLIP.git

#############################################
# Helper: Dummy Feature Extractor for CLIP
#############################################
# This module wraps CLIP's visual backbone (here assumed to be a ResNet)
# and returns a tuple: (global_image_feature, [F1, F2, F3]).
# F1, F2, F3 are assumed to be features at progressively lower resolutions.
class DummyFeatureExtractor(nn.Module):
    def __init__(self, clip_model):
        super(DummyFeatureExtractor, self).__init__()
        # Assume clip_model.visual is a ResNet-like backbone.
        # Copy the first layers.
        self.conv1 = clip_model.visual.conv1   # initial conv
        self.bn1   = clip_model.visual.bn1
        self.relu  = clip_model.visual.relu
        self.maxpool = clip_model.visual.maxpool
        
        # Use successive layers as feature stages.
        self.layer1 = clip_model.visual.layer1  # output resolution: H/8, W/8 (highest resolution)
        self.layer2 = clip_model.visual.layer2  # output resolution: H/16, W/16
        self.layer3 = clip_model.visual.layer3  # output resolution: H/32, W/32 (lowest resolution)
        self.layer4 = clip_model.visual.layer4  # final block
        
    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        F1 = self.layer1(x)  # shape (B, C1, H/8, W/8)
        F2 = self.layer2(F1) # shape (B, C2, H/16, W/16)
        F3 = self.layer3(F2) # shape (B, C3, H/32, W/32)
        # Global feature from the final block
        x_global = self.layer4(F3)  # shape (B, C4, H/64, W/64) or similar
        # Global average pooling to get a (B, C4) vector
        global_feat = x_global.mean(dim=[2,3])
        return global_feat, [F1, F2, F3]


#############################################
# Feature Pyramid Network (FPN)
#############################################
# The FPN projects the global image feature and the intermediate features
# into a common embedding space and fuses them in a top–down manner.
class FPN(nn.Module):
    def __init__(self, clip_dim, in_channels_list, mid_channels, out_channels):
        """
        Args:
            clip_dim: Dimension of the CLIP global image feature.
            in_channels_list: List of channel dimensions for [F1, F2, F3].
                (e.g. [256, 512, 1024])
            mid_channels: Common channel dimension for FPN features (e.g. 256).
            out_channels: Final output channels (should match clip_dim, e.g. 512).
        """
        super(FPN, self).__init__()
        # Projection for intermediate features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels_list[0], mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels_list[1], mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels_list[2], mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        # Projection for the global vector.
        # We first reshape the global vector to a spatial map (matching F3 resolution).
        self.proj_global = nn.Sequential(
            nn.Conv2d(clip_dim, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        # Final projection: from mid_channels to out_channels.
        self.out_conv = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, global_feat, feats):
        """
        Args:
            global_feat: (B, clip_dim) global image feature.
            feats: list of feature maps [F1, F2, F3] with shapes:
                   F1: (B, C1, H/8, W/8)
                   F2: (B, C2, H/16, W/16)
                   F3: (B, C3, H/32, W/32)
        Returns:
            A refined spatial feature map of shape (B, out_channels, H/8, W/8).
        """
        B = global_feat.size(0)
        # Get spatial dims of F3 (lowest resolution)
        _, _, H3, W3 = feats[2].shape
        # Reshape global feature to (B, clip_dim, 1, 1) and expand to match F3 spatial dims
        global_feat = global_feat.view(B, -1, 1, 1).expand(-1, -1, H3, W3)
        global_proj = self.proj_global(global_feat)  # (B, mid_channels, H3, W3)
        
        # Project intermediate features
        feat1 = self.conv1(feats[0])  # (B, mid_channels, H/8, W/8)
        feat2 = self.conv2(feats[1])  # (B, mid_channels, H/16, W/16)
        feat3 = self.conv3(feats[2])  # (B, mid_channels, H/32, W/32)
        
        # Build FPN in a top–down manner:
        # Start at the lowest resolution: fuse global info with feat3.
        fpn3 = feat3 + global_proj  # (B, mid_channels, H/32, W/32)
        # Upsample to resolution of feat2 and fuse.
        fpn3_up = F.interpolate(fpn3, size=feat2.shape[2:], mode='nearest')
        fpn2 = feat2 + fpn3_up  # (B, mid_channels, H/16, W/16)
        # Upsample to resolution of feat1 and fuse.
        fpn2_up = F.interpolate(fpn2, size=feat1.shape[2:], mode='nearest')
        fpn1 = feat1 + fpn2_up  # (B, mid_channels, H/8, W/8)
        
        # Final projection to obtain the dense visual feature map.
        out = self.out_conv(fpn1)  # (B, out_channels, H/8, W/8)
        return out


#############################################
# AffordanceCLIP Model
#############################################
# This model uses the frozen CLIP model to extract image and text embeddings,
# applies the FPN to recover spatial details, and computes a dense activation map.
class AffordanceCLIP(nn.Module):
    def __init__(self, clip_model, fpn):
        """
        Args:
            clip_model: A CLIP model instance.
            fpn: An instance of the FPN module.
        """
        super(AffordanceCLIP, self).__init__()
        self.clip_model = clip_model
        self.fpn = fpn
        # Freeze CLIP parameters.
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, image, text):
        """
        Args:
            image: Preprocessed image tensor (B, 3, H, W).
            text: Tokenized text input (B, L).
        Returns:
            activation: Dense activation map (B, H_out, W_out) with values in [0,1].
        """
        # Get image features. Here we assume that image_encoder returns a tuple:
        # (global_feature, [F1, F2, F3]).
        global_feat, feats = self.clip_model.image_encoder(image, return_intermediate=True)
        # Refine features with FPN.
        spatial_feat = self.fpn(global_feat, feats)  # shape: (B, out_channels, H/8, W/8)
        
        # Encode text using CLIP's text encoder.
        text_feat = self.clip_model.encode_text(text)  # shape: (B, clip_dim)
        # Normalize embeddings.
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        # Consider this alternative normalization for spatial features:
        spatial_feat = spatial_feat / (spatial_feat.norm(dim=1, keepdim=True) + 1e-6)  # Add epsilon for numerical stability
        
        # Compute activation map as the dot product between the text embedding and each pixel's feature.
        # Reshape text_feat to (B, clip_dim, 1, 1) for broadcasting.
        B, C, H, W = spatial_feat.shape
        text_feat = text_feat.view(B, C, 1, 1)
        activation = (spatial_feat * text_feat).sum(dim=1)  # (B, H, W)
        # Apply sigmoid to map values between 0 and 1.
        activation = torch.sigmoid(activation)
        return activation


#############################################
# Pixel–Text Contrastive Loss
#############################################
def pixel_text_contrastive_loss(pred, gt_mask):
    """
    Args:
        pred: Predicted activation map (B, H, W) with values in (0,1).
        gt_mask: Binary ground truth mask (B, H, W) with 1 for positive pixels and 0 for negative.
    Returns:
        Loss value (scalar).
    """
    eps = 1e-6
    loss = - gt_mask * torch.log(pred + eps) - (1 - gt_mask) * torch.log(1 - pred + eps)
    return loss.mean()


#############################################
# Example: Putting Everything Together
#############################################
if __name__ == '__main__':
    # Load the CLIP model (and move to device)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("RN101", device=device)  # using ResNet-101 variant
    
    # Wrap the CLIP model with our feature extractor.
    feature_extractor = DummyFeatureExtractor(clip_model).to(device)
    
    # Create a simple wrapper that uses our feature extractor for the image encoder.
    class CLIPWrapper(nn.Module):
        def __init__(self, clip_model, feature_extractor):
            super(CLIPWrapper, self).__init__()
            self.clip_model = clip_model
            self.feature_extractor = feature_extractor

        def image_encoder(self, image, return_intermediate=False):
            if return_intermediate:
                return self.feature_extractor(image)
            else:
                return self.clip_model.encode_image(image)

        def encode_text(self, text):
            return self.clip_model.encode_text(text)

        # Allow access to all parameters (they remain frozen)
        def parameters(self):
            return self.clip_model.parameters()
    
    clip_wrapper = CLIPWrapper(clip_model, feature_extractor).to(device)
    
    # Define FPN parameters.
    # (These channel numbers depend on the backbone; adjust them as needed.)
    in_channels_list = [256, 512, 1024]   # e.g., channels for F1, F2, F3
    clip_dim = 512                        # dimension of CLIP global feature (for RN101)
    mid_channels = 256
    out_channels = clip_dim               # final output channels
    
    fpn = FPN(clip_dim=clip_dim, in_channels_list=in_channels_list,
              mid_channels=mid_channels, out_channels=out_channels).to(device)
    
    # Instantiate the AffordanceCLIP model.
    model = AffordanceCLIP(clip_model=clip_wrapper, fpn=fpn).to(device)
    
    # Dummy inputs: load an image and tokenize a text prompt.
    # In practice, use the provided `preprocess` function from CLIP and a proper tokenizer.
    from PIL import Image
    try:
        image_pil = Image.open("example.jpg").convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        raise
    image_input = preprocess(image_pil).unsqueeze(0).to(device)  # (1, 3, H, W)
    text_input = clip.tokenize(["type on"]).to(device)           # (1, L)
    
    # Forward pass
    with torch.no_grad():
        activation_map = model(image_input, text_input)  # (1, H_out, W_out)
    
    # (Optional) Upsample activation map to original image resolution.
    activation_map_upsampled = F.interpolate(activation_map.unsqueeze(1),
                                             size=image_input.shape[2:],
                                             mode='bilinear', align_corners=False)
    # Save or visualize activation_map_upsampled[0, 0]
    
    #############################################
    # Dummy Training Loop Example
    #############################################
    # Suppose you have a dataloader yielding (image, text, gt_mask) where:
    #   image: preprocessed image tensor (B, 3, H, W)
    #   text: tokenized text prompt (B, L)
    #   gt_mask: binary mask (B, H_out, W_out) with 1 for the target affordance regions.
    #
    # Only the FPN parameters are trainable.
    optimizer = torch.optim.Adam(model.fpn.parameters(), lr=1e-4)
    
    num_epochs = 10  # for example
    for epoch in range(num_epochs):
        for batch in dataloader:  # define your dataloader accordingly
            images, texts, gt_masks = batch
            images = images.to(device)
            texts = texts.to(device)
            gt_masks = gt_masks.to(device).float()  # ensure float for loss computation
            
            optimizer.zero_grad()
            preds = model(images, texts)  # (B, H_out, W_out)
            loss = pixel_text_contrastive_loss(preds, gt_masks)
            loss.backward()
            optimizer.step()
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
