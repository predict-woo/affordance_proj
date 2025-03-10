import torch
import pytorch_lightning as pl
import clip  # pip install git+https://github.com/openai/CLIP.git
from pytorch_lightning.loggers import WandbLogger
from dataset import ReferDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from affordance_clip import DummyFeatureExtractor, CLIPWrapper, FPN, AffordanceCLIPModule
#############################################
# Training and Inference Script
#############################################
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/cluster/scratch/andrye/proj_data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=1)

    # wandb
    parser.add_argument("--wandb_project", type=str, default="affordance-clip")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    
    args = parser.parse_args()
    
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=args.wandb_name,
        entity=args.wandb_entity,
        log_model=True
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load CLIP and its preprocess.
    clip_model, preprocess = clip.load("RN101", device=device)
    # Create a feature extractor from CLIP.
    feature_extractor = DummyFeatureExtractor(clip_model).to(device)
    clip_wrapper = CLIPWrapper(clip_model, feature_extractor).to(device)
    
    # Define FPN parameters (adjust channels as needed).
    in_channels_list = [512, 1024, 2048]  # example channels for F1, F2, F3
    clip_dim = 512  # CLIP RN101 output dimension
    mid_channels = 256
    out_channels = clip_dim
    fpn = FPN(clip_dim=clip_dim, in_channels_list=in_channels_list, mid_channels=mid_channels, out_channels=out_channels).to(device)
    
    # Instantiate the LightningModule.
    model_module = AffordanceCLIPModule(clip_model=clip_wrapper, fpn=fpn, learning_rate=1e-4)
    
    # Prepare the DataModule for RefCOCOg (UMD split recommended).
    data_module = ReferDataModule(
        data_root=args.data_root,
        dataset='refcocog',
        splitBy='umd',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        preprocess=preprocess
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        filename='affordance-clip-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss'
    )
    
    # Initialize PyTorch Lightning Trainer.
    trainer = pl.Trainer(
        max_epochs=args.max_epochs, 
        accelerator='gpu' if device == 'cuda' else 'cpu', 
        devices=args.gpus, 
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )
    
    # Train the model.
    trainer.fit(model_module, datamodule=data_module)
