import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

LOG = False

#############################################
# Model Components (CLIP Feature Extractor, FPN, AffordanceCLIP)
#############################################
class DummyFeatureExtractor(nn.Module):
    """
    A simple wrapper to extract intermediate features from CLIP's visual backbone.
    Assumes a ResNet-like structure.
    """
    def __init__(self, clip_model):
        super(DummyFeatureExtractor, self).__init__()
        self.conv1 = clip_model.visual.conv1
        self.conv2 = clip_model.visual.conv2
        self.conv3 = clip_model.visual.conv3
        self.bn1   = clip_model.visual.bn1
        self.bn2   = clip_model.visual.bn2
        self.bn3   = clip_model.visual.bn3
        self.relu1  = clip_model.visual.relu1
        self.relu2  = clip_model.visual.relu2
        self.relu3  = clip_model.visual.relu3
        self.avgpool = clip_model.visual.avgpool
        # Using ResNet blocks as feature stages (adjust channels if needed)
        self.layer1 = clip_model.visual.layer1  # high-res feature: ~H/8, W/8 
        self.layer2 = clip_model.visual.layer2  # ~H/16, W/16
        self.layer3 = clip_model.visual.layer3  # ~H/32, W/32
        self.layer4 = clip_model.visual.layer4  # final block
        
        self.visual = clip_model.visual

    def forward(self, x):
        
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        y = x.type(self.conv1.weight.dtype)
        y = stem(y)
        y = self.layer1(y)
        F1 = self.layer2(y)
        F2 = self.layer3(F1)
        F3 = self.layer4(F2)
        
        global_feat = self.visual(x)
        return global_feat, [F1, F2, F3]

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


class FPN(nn.Module):
    """
    A lightweight Feature Pyramid Network (FPN) that fuses a global CLIP feature with
    intermediate feature maps (F1, F2, F3) to recover spatial details.
    """
    def __init__(self, clip_dim, in_channels_list, mid_channels, out_channels):
        super(FPN, self).__init__()
        # Project each intermediate feature to a common channel dimension.
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
        # Project the global feature and spatially broadcast.
        self.proj_global = nn.Sequential(
            nn.Conv2d(clip_dim, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        # Final projection to match CLIP's feature dimension.
        self.out_conv = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, global_feat, feats):
        B = global_feat.size(0)
        # Get the spatial size from the lowest-resolution feature (F3)
        _, _, H3, W3 = feats[2].shape
        
        # Convert global_feat to same precision as model weights
        global_feat = global_feat.to(self.proj_global[0].weight.dtype)
        global_feat = global_feat.view(B, -1, 1, 1).expand(-1, -1, H3, W3)
        global_proj = self.proj_global(global_feat)
        
        # Convert feature maps to same precision as model weights
        feat1 = self.conv1(feats[0].to(self.conv1[0].weight.dtype))
        feat2 = self.conv2(feats[1].to(self.conv2[0].weight.dtype))
        feat3 = self.conv3(feats[2].to(self.conv3[0].weight.dtype))
        
        # Top-down fusion: start at F3 and upsample progressively.
        fpn3 = feat3 + global_proj
        fpn3_up = F.interpolate(fpn3, size=feat2.shape[2:], mode='nearest')
        fpn2 = feat2 + fpn3_up
        fpn2_up = F.interpolate(fpn2, size=feat1.shape[2:], mode='nearest')
        fpn1 = feat1 + fpn2_up
        out = self.out_conv(fpn1)
        return out

class AffordanceCLIP(nn.Module):
    """
    The overall model that uses a frozen CLIP backbone to obtain image and text embeddings,
    refines the image features via an FPN, and produces a dense activation map via a dot product.
    """
    def __init__(self, clip_model, fpn):
        super(AffordanceCLIP, self).__init__()
        self.clip_model = clip_model
        self.fpn = fpn
        # Freeze all CLIP parameters.
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, image, text):
        # Obtain a global image feature and intermediate features.
        global_feat, feats = self.clip_model.image_encoder(image, return_intermediate=True)

        # Fuse features with FPN.
        spatial_feat = self.fpn(global_feat, feats)
        # Get text embedding.
        text_feat = self.clip_model.encode_text(text)
        
        if LOG:
            # save text_feat as plot (text_feat.shape torch.Size([32, 512]))
            import matplotlib.pyplot as plt
            plt.imshow(text_feat.cpu().numpy())
            plt.savefig("text_feat.png")
            print("spatial_feat.shape", spatial_feat.shape)
            print("text_feat.shape", text_feat.shape)
        
        # # Normalize features.
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        spatial_feat = spatial_feat / spatial_feat.norm(dim=1, keepdim=True)
        # Compute dense activation map by dot-product.
        B, C, H, W = spatial_feat.shape
        text_feat = text_feat.view(B, C, 1, 1)
        activation = (spatial_feat * text_feat).sum(dim=1)
        activation = torch.sigmoid(activation)
        return activation

def pixel_text_contrastive_loss(pred, gt_mask, eps=1e-6):
    loss = - gt_mask * torch.log(pred + eps) - (1 - gt_mask) * torch.log(1 - pred + eps)
    return loss.mean()

#############################################
# PyTorch Lightning Module for Training/Inference
#############################################
class AffordanceCLIPModule(pl.LightningModule):
    def __init__(self, clip_model, fpn, learning_rate=1e-4):
        super().__init__()
        self.model = AffordanceCLIP(clip_model=clip_model, fpn=fpn)
        self.learning_rate = learning_rate

    def forward(self, image, text):
        return self.model(image, text)

    def training_step(self, batch, batch_idx):
        image, text, mask = batch
        preds = self(image, text)
        mask = mask.float()
        loss = pixel_text_contrastive_loss(preds, mask)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, text, mask = batch
        preds = self(image, text)
        mask = mask.float()
        loss = pixel_text_contrastive_loss(preds, mask)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.fpn.parameters(), lr=self.learning_rate)
        return optimizer
