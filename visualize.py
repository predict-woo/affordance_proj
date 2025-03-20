import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from scipy.ndimage import zoom


def undo_clip_normalization(processed_image):
    '''
    Undo the normalization applied by CLIP.
    processed_image: (H, W, 3)
    returns: (H, W, 3)
    '''
    mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(1, 1, 3)
    std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(1, 1, 3)
    processed_image = processed_image * std + mean
    processed_image = np.clip(processed_image, 0, 1)
    return processed_image
    

def overlay_image_and_mask(image, mask, alpha=0.5):
    '''
    Overlay the mask on the image with a given alpha value.
    image: (H1, W1, 3)
    mask: (H2, W2, 1)
    alpha: float
    '''
    print(image.shape, mask.shape)
    zoom_factor = image.shape[0] / mask.shape[0]
    mask = zoom(mask, (zoom_factor, zoom_factor, 1))
    print(mask.shape)
    
    
    # repeat mask to 3 channels
    mask = np.repeat(mask, 3, axis=2)
    overlay = image * (1 - alpha) + mask * alpha

    return overlay

def visualize_feature_map(feature_map, save_path, title=None):
    '''
    Visualize the feature map.
    feature_map: (1, C, H, W)
    '''
    # reset matplotlib
    plt.clf()
    print(f"visualize_feature_map ({title})", feature_map.shape)
    detached_feature_map = feature_map.detach().cpu().numpy()
    
    # pick the first one in the batch
    detached_feature_map = detached_feature_map[0]
    
    # average over the channels
    detached_feature_map = np.mean(detached_feature_map, axis=0)
    plt.imshow(detached_feature_map, vmin=0)
    plt.colorbar()
    if title:
        plt.title(title)
    plt.savefig(save_path)