import torch
import torchvision.models as models
import torch.nn as nn

def load_model(model_path, num_classes):
    model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    model_ft.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model_ft
