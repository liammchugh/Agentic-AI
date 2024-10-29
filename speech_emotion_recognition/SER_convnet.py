import torch
import torch.nn as nn
import torchvision.models as models

class SERCONV(nn.Module):
    def __init__(self, model_type, input_shape, reg_strength=0.01, dropout_rate=0.5):
        super(SERCONV, self).__init__()

        if model_type == 'vgg':
            self.backbone = models.vgg16(weights=None)
            self.backbone.features[0] = nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=1, padding=1)
        elif model_type == 'resnet':
            self.backbone = models.resnet50(weights=None)
            self.backbone.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            raise ValueError("Unsupported model type. Choose 'vgg' or 'resnet'.")

        self.backbone.fc = nn.Identity()  # Remove the last layer (for ResNet)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.backbone.fc.in_features if model_type == 'resnet' else 512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.output = nn.Linear(4096, 5)

        # Regularization and dropout
        self.regularizer = nn.Parameter(torch.Tensor([reg_strength]), requires_grad=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        x = nn.functional.relu(self.fc1(x)) + self.regularizer * torch.sum(self.fc1.weight ** 2)
        x = self.dropout(x)
        x = nn.functional.relu(self.fc2(x)) + self.regularizer * torch.sum(self.fc2.weight ** 2)
        x = self.dropout(x)
        x = self.output(x)

        return nn.functional.softmax(x, dim=1)

if __name__ == "__main__":
    model_type = 'vgg'  # or 'resnet'
    input_shape = (3, 224, 224)
    model = SERCONV(model_type=model_type, input_shape=input_shape, reg_strength=0.01, dropout_rate=0.5)
