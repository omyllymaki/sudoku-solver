import torch
from torchvision import transforms

from digit_classifier import DigitClassifier

TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def predict_digit(cell, model):
    cell_tensor = TRANSFORMS(cell)
    cell_tensor = cell_tensor.unsqueeze(0)
    output = model(cell_tensor)
    predicted_digit = output.argmax(dim=1)[0].item()
    probability = round(torch.exp(output.max()).item(), 2)
    return predicted_digit, probability


def load_model(path):
    model = DigitClassifier()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
