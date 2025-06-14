
import torch
import torch.nn as nn
import torch.nn.functional as F

class alexNet(nn.Module):
    def __init__(self):
        super(alexNet, self).__init__()
        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels=96, stride=4, kernel_size=11, padding=0)
        self.lrn1 = nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # 2nd Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels = 96, out_channels=256, stride=1, kernel_size=5, padding=2)
        self.lrn2 = nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # 3rd Convolutional Layer
        self.conv3 = nn.Conv2d(in_channels = 256, out_channels=384, stride=1, kernel_size=3, padding=1)

        # 4th Convolutional Layer
        self.conv4 = nn.Conv2d(in_channels = 384, out_channels=384, stride=1, kernel_size=3, padding=1)

        # 5th Convolutional Layer        
        self.conv5 = nn.Conv2d(in_channels = 384, out_channels=256, stride=1, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        # 6th- Fully Connected Layer
        self.dropout1 = nn.Dropout()
        self.fc1 = nn.Linear(256*6*6, 4096)

        # 7th Fully Connected Layer
        self.dropout2 = nn.Dropout()
        self.fc2 = nn.Linear(4096, 4096)

        # 8th Fully Connected Layer
        self.fc3 = nn.Linear(4096, 1000)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.lrn1(x)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.lrn2(x)
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool3(x)

        x = x.view(x.size(0), 256*6*6)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def save_model(model, filepath="alexnet_state_dict.pth"):
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath="alexnet_state_dict.pth"):
    model = alexNet()
    model.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
    model.eval()  # Set to evaluation mode
    print(f"Model loaded from {filepath}")
    return model

if __name__=="__main__":
    pass