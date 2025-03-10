import torch
import os
import torch
from PIL import Image
from refer import REFER
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import clip
import torch.nn.functional as F
from visualize import overlay_image_and_mask, undo_clip_normalization


def resize_with_aspect_ratio(tensor, target_size, mode='bilinear'):
    """
    Resizes a tensor while maintaining aspect ratio.
    """
    _, _, original_height, original_width = tensor.shape
    aspect_ratio = original_width / original_height

    if original_height < original_width:
        new_size = (target_size, int(target_size * aspect_ratio))
    else:
        new_size = (int(target_size / aspect_ratio), target_size)

    return F.interpolate(tensor, size=new_size, mode=mode, align_corners=False)

def center_crop_tensor(tensor, output_size):
    """
    Performs a center crop on a tensor.

    Args:
        tensor (torch.Tensor): Input tensor of shape [batch_size, channels, height, width].
        output_size (int or tuple): Desired output size (H, W). If int, a square crop is made.

    Returns:
        torch.Tensor: Center-cropped tensor.
    """
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    _, _, h, w = tensor.shape
    crop_h, crop_w = output_size

    # Ensure crop size is not larger than the original size
    crop_h = min(crop_h, h)
    crop_w = min(crop_w, w)

    # Calculate cropping coordinates
    start_y = (h - crop_h) // 2
    start_x = (w - crop_w) // 2

    # Perform the crop
    return tensor[:, :, start_y:start_y + crop_h, start_x:start_x + crop_w]


def transform_mask(mask, n_px):
    mask = resize_with_aspect_ratio(mask, n_px)
    mask = center_crop_tensor(mask, n_px)
    return mask


class ReferDataset(torch.utils.data.Dataset):
    def __init__(self, refer, split='train', transforms=None, return_mask=True, preprocess=None):
        """
        Args:
            refer: REFER object
            split: dataset split ('train', 'val', 'test', 'testA', 'testB', etc.)
            transforms: image transforms
            return_mask: whether to return instance mask or bounding box
            preprocess: clip preprocess
        """
        self.refer = refer
        self.split = split
        self.transforms = transforms
        self.return_mask = return_mask
        self.preprocess = preprocess
        
        # Get all relevant ref_ids for this split
        self.ref_ids = self.refer.getRefIds(split=self.split)
        
        if len(self.ref_ids) == 0:
            raise ValueError(f"No reference ids found for split '{split}'")
            
        print(f"Loaded {len(self.ref_ids)} referring expressions for split '{split}'")
        
    def __len__(self):
        return len(self.ref_ids)
    
    def __getitem__(self, idx):
        ref_id = self.ref_ids[idx]
        ref = self.refer.loadRefs(ref_id)[0]
        
        img_id = ref['image_id']
        img_info = self.refer.loadImgs(img_id)[0]
        img_path = os.path.join(self.refer.IMAGE_DIR, img_info['file_name'])
                
        try:
            image = Image.open(img_path).convert('RGB')
            if self.preprocess is not None:
                image = self.preprocess(image)
                
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a small placeholder image if the actual image can't be loaded
            import numpy as np
            image = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
        
        # Get mask or bbox
        if self.return_mask:
            mask_info = self.refer.getMask(ref)
            mask = mask_info['mask']
            target = torch.as_tensor(mask, dtype=torch.uint8).unsqueeze(0)  # Add channel dimension
        else:
            ann = self.refer.refToAnn[ref_id]
            bbox = torch.as_tensor(ann['bbox'], dtype=torch.float32)  # [x, y, width, height]
            target = bbox
        
        # tranform target in same way as image
        
        # Get referring expressions (all sentences for this reference)
        sentences = [sent['sent'] for sent in ref['sentences']]
        
        # Randomly select one sentence
        import random
        sentence = random.choice(sentences)
        
        # Tokenize the sentence
        sentence = clip.tokenize(sentence)
        
        # Pad the tokenized sentence to a fixed length
        max_length = 77  # CLIP's max token length
        if sentence.size(1) < max_length:
            padding = torch.zeros((1, max_length - sentence.size(1)), dtype=torch.int64)
            sentence = torch.cat((sentence, padding), dim=1)
        
        # Apply transforms if available
        if self.transforms is not None:
            if isinstance(self.transforms, list):
                for t in self.transforms:
                    if hasattr(t, '__call__'):
                        image, target = t(image, target)
            else:
                image, target = self.transforms(image, target)
        
        n_px = image.shape[1]
        target = transform_mask(target.unsqueeze(0), n_px // 8)
        target = target.squeeze(0)
        
        sentence = sentence.squeeze(0)
                
        # return {
        #     'image': image,
        #     'target': target,
        #     'sentence': sentence,
        #     'sentences': sentences,
        #     'ref_id': ref_id,
        #     'img_id': img_id
        # }
        
        return image, sentence, target

class ReferDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        data_root,
        dataset='refcoco',
        splitBy='unc',
        batch_size=32,
        num_workers=4,
        preprocess=None,
        train_transforms=None,
        val_transforms=None,
        test_transforms=None,
        return_mask=True,
    ):
        """
        PyTorch Lightning DataModule for REFER dataset
        
        Args:
            data_root: Path to the data directory
            dataset: Dataset name ('refcoco', 'refcoco+', 'refcocog', etc.)
            splitBy: Split method ('unc', 'google', 'umd', etc.)
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            train_transforms: Transformations to apply to training data
            val_transforms: Transformations to apply to validation data
            test_transforms: Transformations to apply to test data
            return_mask: Whether to return segmentation masks (True) or bounding boxes (False)
        """
        super().__init__()
        self.data_root = data_root
        self.dataset = dataset
        self.splitBy = splitBy
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.return_mask = return_mask
        self.preprocess = preprocess
    
    
    def prepare_data(self):
        """
        Download data if needed. This method is called only from a single process.
        """
        # REFER dataset should be downloaded manually, so we'll just check if it exists
        expected_path = os.path.join(self.data_root, self.dataset)
        if not os.path.exists(expected_path):
            raise FileNotFoundError(
                f"Dataset {self.dataset} not found at {expected_path}. "
                f"Please download the dataset and place it in the correct location."
            )
            
    def setup(self, stage=None):
        """
        Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called on every process when using DDP.
        """
        # Initialize REFER
        self.refer = REFER(self.data_root, dataset=self.dataset, splitBy=self.splitBy)
        
        # Create datasets based on stage
        self.train_dataset = ReferDataset(
            self.refer, 
            split='train', 
            transforms=self.train_transforms,
            preprocess=self.preprocess,
            return_mask=self.return_mask,
        )
        
        self.val_dataset = ReferDataset(
            self.refer, 
            split='val', 
            transforms=self.val_transforms,
            preprocess=self.preprocess,
            return_mask=self.return_mask,
        )
        
        self.test_dataset = ReferDataset(
            self.refer, 
            split='test', 
            transforms=self.test_transforms,
            preprocess=self.preprocess,
            return_mask=self.return_mask,
        )
        
            
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )



class CombinedReferDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, datasets=None, splitBys=None, split='train', transforms=None, return_mask=True, preprocess=None):
        """
        Args:
            data_root: Path to the data directory
            datasets: List of dataset names ['refcoco', 'refcoco+', 'refcocog']
            splitBys: List of split methods ['unc', 'unc', 'umd'] corresponding to datasets
            split: dataset split ('train', 'val', 'test')
            transforms: image transforms
            return_mask: whether to return instance mask or bounding box
            preprocess: clip preprocess
        """
        self.data_root = data_root
        self.datasets = datasets or ['refcoco', 'refcoco+', 'refcocog']
        self.splitBys = splitBys or ['unc', 'unc', 'umd']
        self.split = split
        self.transforms = transforms
        self.return_mask = return_mask
        self.preprocess = preprocess
        
        assert len(self.datasets) == len(self.splitBys), "Number of datasets must match number of splitBys"
        
        # Initialize REFER objects for each dataset
        self.refers = {}
        for dataset, splitBy in zip(self.datasets, self.splitBys):
            self.refers[dataset] = REFER(self.data_root, dataset=dataset, splitBy=splitBy)
        
        # Get all relevant ref_ids for this split, and track which dataset they belong to
        self.ref_ids = []
        self.ref_to_dataset = {}
        
        for dataset, refer_obj in self.refers.items():
            dataset_ref_ids = refer_obj.getRefIds(split=self.split)
            for ref_id in dataset_ref_ids:
                self.ref_ids.append(ref_id)
                self.ref_to_dataset[ref_id] = dataset
        
        if len(self.ref_ids) == 0:
            raise ValueError(f"No reference ids found for split '{split}' in any dataset")
            
        print(f"Loaded {len(self.ref_ids)} referring expressions for split '{split}' across all datasets")
        for dataset in self.datasets:
            count = sum(1 for ref_id in self.ref_ids if self.ref_to_dataset[ref_id] == dataset)
            print(f"  - {dataset}: {count} expressions")
    
    def __len__(self):
        return len(self.ref_ids)
    
    def __getitem__(self, idx):
        ref_id = self.ref_ids[idx]
        dataset = self.ref_to_dataset[ref_id]
        refer = self.refers[dataset]
        
        ref = refer.loadRefs(ref_id)[0]
        
        img_id = ref['image_id']
        img_info = refer.loadImgs(img_id)[0]
        img_path = os.path.join(refer.IMAGE_DIR, img_info['file_name'])
                
        try:
            image = Image.open(img_path).convert('RGB')
            if self.preprocess is not None:
                image = self.preprocess(image)
                
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a small placeholder image if the actual image can't be loaded
            import numpy as np
            image = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
        
        # Get mask or bbox
        if self.return_mask:
            mask_info = refer.getMask(ref)
            mask = mask_info['mask']
            target = torch.as_tensor(mask, dtype=torch.uint8).unsqueeze(0)  # Add channel dimension
        else:
            ann = refer.refToAnn[ref_id]
            bbox = torch.as_tensor(ann['bbox'], dtype=torch.float32)  # [x, y, width, height]
            target = bbox
        
        # Get referring expressions (all sentences for this reference)
        sentences = [sent['sent'] for sent in ref['sentences']]
        
        # Randomly select one sentence
        import random
        sentence = random.choice(sentences)
        print(sentence)
        
        # Tokenize the sentence
        sentence = clip.tokenize(sentence)
        
        # Pad the tokenized sentence to a fixed length
        max_length = 77  # CLIP's max token length
        if sentence.size(1) < max_length:
            padding = torch.zeros((1, max_length - sentence.size(1)), dtype=torch.int64)
            sentence = torch.cat((sentence, padding), dim=1)
        
        # Apply transforms if available
        if self.transforms is not None:
            if isinstance(self.transforms, list):
                for t in self.transforms:
                    if hasattr(t, '__call__'):
                        image, target = t(image, target)
            else:
                image, target = self.transforms(image, target)
        
        n_px = image.shape[1]
        target = transform_mask(target.unsqueeze(0), n_px // 8)
        target = target.squeeze(0)
        
        sentence = sentence.squeeze(0)
        
        return image, sentence, target

class CombinedReferDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        data_root,
        datasets=None,
        splitBys=None,
        batch_size=32,
        num_workers=4,
        preprocess=None,
        train_transforms=None,
        val_transforms=None,
        test_transforms=None,
        return_mask=True,
    ):
        """
        PyTorch Lightning DataModule for combined REFER datasets
        
        Args:
            data_root: Path to the data directory
            datasets: List of dataset names ['refcoco', 'refcoco+', 'refcocog']
            splitBys: List of split methods ['unc', 'unc', 'umd'] corresponding to datasets
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            train_transforms: Transformations to apply to training data
            val_transforms: Transformations to apply to validation data
            test_transforms: Transformations to apply to test data
            return_mask: Whether to return segmentation masks (True) or bounding boxes (False)
        """
        super().__init__()
        self.data_root = data_root
        self.datasets = datasets or ['refcoco', 'refcoco+', 'refcocog']
        self.splitBys = splitBys or ['unc', 'unc', 'umd']
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.return_mask = return_mask
        self.preprocess = preprocess
    
    def prepare_data(self):
        """
        Check if datasets exist.
        """
        for dataset in self.datasets:
            expected_path = os.path.join(self.data_root, dataset)
            if not os.path.exists(expected_path):
                raise FileNotFoundError(
                    f"Dataset {dataset} not found at {expected_path}. "
                    f"Please download the dataset and place it in the correct location."
                )
            
    def setup(self, stage=None):
        """
        Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        """
        # Create combined datasets for each split
        self.train_dataset = CombinedReferDataset(
            self.data_root,
            datasets=self.datasets,
            splitBys=self.splitBys,
            split='train', 
            transforms=self.train_transforms,
            preprocess=self.preprocess,
            return_mask=self.return_mask,
        )
        
        self.val_dataset = CombinedReferDataset(
            self.data_root,
            datasets=self.datasets,
            splitBys=self.splitBys,
            split='val', 
            transforms=self.val_transforms,
            preprocess=self.preprocess,
            return_mask=self.return_mask,
        )
        
        self.test_dataset = CombinedReferDataset(
            self.data_root,
            datasets=self.datasets,
            splitBys=self.splitBys,
            split='test', 
            transforms=self.test_transforms,
            preprocess=self.preprocess,
            return_mask=self.return_mask,
        )
            
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


if __name__ == '__main__':
    
    import numpy as np
    import clip
    import matplotlib.pyplot as plt
    model, preprocess = clip.load("RN101", device="cpu")
    refer = REFER("/cluster/scratch/andrye/proj_data", dataset="refcocog", splitBy="umd")
    # refer = REFER("/cluster/scratch/andrye/proj_data", dataset="refcoco", splitBy="unc")
    # refer = REFER("/cluster/scratch/andrye/proj_data", dataset="refcoco+", splitBy="unc")

    
    dataset = ReferDataset(refer, split='train',preprocess=preprocess)
    image, sentence, target = dataset[10]
    
    print(image.shape, target.shape)
    image = image.permute(1, 2, 0).numpy()
    target = target.permute(1, 2, 0).numpy()
    
    image = undo_clip_normalization(image)
    
    image = overlay_image_and_mask(image, target)
    
    plt.imsave("image.png", image)
    
    
    
    
    # import matplotlib.pyplot as plt
    # # Convert PyTorch tensor to numpy array and transpose dimensions
    # plt.imshow(image.permute(1, 2, 0).numpy())
    # plt.savefig("image.png")
    
    # plt.figure()  # Create a new figure
    # plt.imshow(target.squeeze().numpy())  # Remove channel dimension for mask
    # plt.savefig("target.png")
    
    # datamodule = ReferDataModule(
    #     data_root="/cluster/scratch/andrye/proj_data",
    #     dataset="refcocog",
    #     splitBy="umd",
    #     batch_size=32,
    #     num_workers=4
    # )
    # datamodule.setup()
    # train_loader = datamodule.train_dataloader()
    # val_loader = datamodule.val_dataloader()
    # test_loader = datamodule.test_dataloader()
    
    # print(len(train_loader))
    # print(len(val_loader))
    # print(len(test_loader))