'''
Code to visualize the intermediate features of CLIP model
'''


from affordance_clip import CLIPWrapper, FPN, AffordanceCLIP, AffordanceCLIPModule, pixel_text_contrastive_loss, DummyFeatureExtractor
from visualize import visualize_feature_map
import clip
from test import load_image
import torch


def test_feature(device):
    """Load the trained AffordanceCLIP model"""
    # Load CLIP model
    clip_model, preprocess = clip.load("RN101", device=device)
    
    # Create feature extractor and wrapper
    feature_extractor = DummyFeatureExtractor(clip_model).to(device)
    clip_wrapper = CLIPWrapper(clip_model, feature_extractor).to(device)
    
    # load image
    image, sentence, target = load_image(10)
    
    image = image.unsqueeze(0).to(device)
    sentence = sentence.unsqueeze(0).to(device)
    
    # get feature map
    feature_map, intermediate_features = clip_wrapper.image_encoder(image, return_intermediate=True)
    
    for feature in intermediate_features:
        print(feature.shape)
    
    # visualize intermediate features
    for i, feature in enumerate(intermediate_features):
        visualize_feature_map(feature, f"intermediate_feature_{i}.png", title=f"intermediate_feature_{i}")
    
    
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_feature(device)
