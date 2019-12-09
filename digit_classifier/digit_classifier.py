import torch
from torchvision import transforms

from constants import MNIST_DATASET_STDEV, MNIST_DATASET_MEAN
from digit_classifier.net import Net


class DigitClassifier:
    TRANSFORMS = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_DATASET_MEAN,), (MNIST_DATASET_STDEV,))
    ])

    def __init__(self, path, min_n_white_pixels=10, threshold=250):
        self.model = Net()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        self.min_n_white_pixels = min_n_white_pixels
        self.threshold = threshold

    def predict(self, image):
        n_white_pixels = sum(sum(image > self.threshold))
        if n_white_pixels > self.min_n_white_pixels:
            predicted_digit, probability = self._predict_digit(image)
        else:
            predicted_digit = None
            probability = 1
        return predicted_digit, probability

    def _predict_digit(self, image):
        tensor = self.TRANSFORMS(image)
        tensor = tensor.unsqueeze(0)
        output = self.model(tensor)
        predicted_digit = output.argmax(dim=1)[0].item()
        probability = torch.exp(output.max()).item()
        return predicted_digit, probability
