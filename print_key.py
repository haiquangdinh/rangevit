import torch

# Load the pretrained model and print its keys
pretrained = torch.load('pretrain/model_nuscenes_cs_init.pth')
if isinstance(pretrained, dict) and 'model' in pretrained:
    pretrained = pretrained['model']  # Some checkpoints store the model within a dict
print("Available keys in pretrained model:")
print([k for k in pretrained.keys() if 'pos_embed' in k or 'embed' in k])  # Look for position embedding keys