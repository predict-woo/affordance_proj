import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import clip  # pip install git+https://github.com/openai/CLIP.git
import numpy as np
import pycocotools.mask as mask_util

# Import the REFER class (make sure refer.py is in your PYTHONPATH)
from refer import REFER

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
        self.bn1   = clip_model.visual.bn1
        self.relu  = clip_model.visual.relu1
        self.avgpool = clip_model.visual.avgpool
        # Using ResNet blocks as feature stages (adjust channels if needed)
        self.layer1 = clip_model.visual.layer1  # high-res feature: ~H/8, W/8 
        self.layer2 = clip_model.visual.layer2  # ~H/16, W/16
        self.layer3 = clip_model.visual.layer3  # ~H/32, W/32
        self.layer4 = clip_model.visual.layer4  # final block

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.avgpool(x)
        F1 = self.layer1(x)
        F2 = self.layer2(F1)
        F3 = self.layer3(F2)
        x_global = self.layer4(F3)
        global_feat = x_global.mean(dim=[2, 3])
        return global_feat, [F1, F2, F3]

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
        global_feat = global_feat.view(B, -1, 1, 1).expand(-1, -1, H3, W3)
        global_proj = self.proj_global(global_feat)
        
        feat1 = self.conv1(feats[0])  # resolution: H/8, W/8
        feat2 = self.conv2(feats[1])  # resolution: H/16, W/16
        feat3 = self.conv3(feats[2])  # resolution: H/32, W/32
        
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
        # Normalize features.
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
# PyTorch Lightning DataModule for RefCOCO/+/g
#############################################
class RefCOCODataset(Dataset):
    """
    A dataset wrapper for RefCOCO/+/g using the REFER API.
    Each sample returns:
      - image: preprocessed image tensor,
      - text: tokenized referring expression,
      - mask: binary segmentation mask as a tensor.
    """
    def __init__(self, refer, split, image_dir, preprocess):
        self.refer = refer
        self.split = split  # e.g., "train", "val", "test"
        self.image_dir = image_dir
        self.preprocess = preprocess
        # Get reference IDs for the specified split.
        self.ref_ids = self.refer.getRefIds(split=split)
    
    def __len__(self):
        return len(self.ref_ids)
    
    def __getitem__(self, idx):
        ref_id = self.ref_ids[idx]
        ref = self.refer.loadRefs(ref_id)[0]
        ann_id = ref['ann_id']
        file_name = ref['file_name']
        image_path = os.path.join(self.image_dir, file_name)
        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(image)
        # Use the first sentence as the text prompt.
        text = ref['sentences'][0]['sent']
        text_input = clip.tokenize([text]).squeeze(0)
        # Retrieve the annotation to get segmentation info.
        ann = self.refer.anns[ann_id]
        mask = self.decode_mask(ann, image.size)  # using original image size
        mask = torch.tensor(mask, dtype=torch.float32)
        # Resize mask to match the output activation resolution (assume same as preprocess size here).
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0),
                             size=(self.preprocess.transforms[-1].size, self.preprocess.transforms[-1].size),
                             mode='bilinear', align_corners=False)
        mask = mask.squeeze(0).squeeze(0)
        return image, text_input, mask

    def decode_mask(self, ann, img_size):
        # img_size: (width, height)
        width, height = img_size
        segm = ann['segmentation']
        if isinstance(segm, list):
            rles = mask_util.frPyObjects(segm, height, width)
            rle = mask_util.merge(rles)
        elif isinstance(segm['counts'], list):
            rle = mask_util.frPyObjects(segm, height, width)
        else:
            rle = ann['segmentation']
        mask = mask_util.decode(rle)
        return mask

class RefCOCODataModule(pl.LightningDataModule):
    """
    LightningDataModule for loading RefCOCO/+/g.
    Here we use the UMD split for RefCOCOg.
    """
    def __init__(self, data_root, dataset_name='refcocog', splitBy='umd', batch_size=32, num_workers=4):
        super().__init__()
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.splitBy = splitBy
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_dir = os.path.join(data_root, 'images', 'mscoco')
        # Initialize REFER API.
        self.refer = REFER(data_root, dataset=self.dataset_name, splitBy=self.splitBy)
        self.refer.IMAGE_DIR
        # Load CLIP (to get its preprocess function)
        self.clip_model, self.preprocess = clip.load("RN101", device="cpu")
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = RefCOCODataset(self.refer, split='train', image_dir=self.image_dir, preprocess=self.preprocess)
            self.val_dataset = RefCOCODataset(self.refer, split='val', image_dir=self.image_dir, preprocess=self.preprocess)
        if stage == 'test':
            self.test_dataset = RefCOCODataset(self.refer, split='test', image_dir=self.image_dir, preprocess=self.preprocess)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

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
        loss = pixel_text_contrastive_loss(preds, mask)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, text, mask = batch
        preds = self(image, text)
        loss = pixel_text_contrastive_loss(preds, mask)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.fpn.parameters(), lr=self.learning_rate)
        return optimizer

#############################################
# Training and Inference Script
#############################################
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load CLIP and its preprocess.
    clip_model, preprocess = clip.load("RN101", device=device)
    # Create a feature extractor from CLIP.
    feature_extractor = DummyFeatureExtractor(clip_model).to(device)
    
    # Wrap CLIP to allow access to intermediate features.
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
    
    clip_wrapper = CLIPWrapper(clip_model, feature_extractor).to(device)
    
    # Define FPN parameters (adjust channels as needed).
    in_channels_list = [256, 512, 1024]  # example channels for F1, F2, F3
    clip_dim = 512  # CLIP RN101 output dimension
    mid_channels = 256
    out_channels = clip_dim
    fpn = FPN(clip_dim=clip_dim, in_channels_list=in_channels_list, mid_channels=mid_channels, out_channels=out_channels).to(device)
    
    # Instantiate the LightningModule.
    model_module = AffordanceCLIPModule(clip_model=clip_wrapper, fpn=fpn, learning_rate=1e-4)
    
    # Prepare the DataModule for RefCOCOg (UMD split recommended).
    data_module = RefCOCODataModule(
        data_root=args.data_root,
        dataset_name='refcocog',
        splitBy='umd',
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Initialize PyTorch Lightning Trainer.
    trainer = pl.Trainer(max_epochs=args.max_epochs, accelerator='gpu' if device == 'cuda' else 'cpu', devices=args.gpus)
    
    # Train the model.
    trainer.fit(model_module, datamodule=data_module)
    
    # For inference, load a batch from the test dataloader.
    test_loader = data_module.test_dataloader()
    model_module.eval()
    with torch.no_grad():
        for image, text, mask in test_loader:
            image = image.to(device)
            text = text.to(device)
            preds = model_module(image, text)
            # Here you can save or visualize preds (upsample if needed).
            print("Inference activation map shape:", preds.shape)
            break
