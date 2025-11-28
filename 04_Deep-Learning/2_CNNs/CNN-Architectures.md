# CNN Architectures

> **Evolution of CNNs** - From LeNet to EfficientNet

---

## 🎯 LeNet-5 (1998)

```python
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 16*4*4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

---

## 📊 AlexNet (2012)

**Key innovations:**
- ReLU activation
- Dropout
- Data augmentation
- GPU training

---

## 🎯 VGG (2014)

**Principle:** Deeper is better with small filters

```python
# VGG block
def vgg_block(in_channels, out_channels, num_convs):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)
```

---

## 📈 ResNet (2015)

**Skip connections solve vanishing gradients**

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        out = torch.relu(out)
        return out
```

---

## 🎯 EfficientNet (2019)

**Compound scaling:** Width × Depth × Resolution

```python
from torchvision.models import efficientnet_b0

model = efficientnet_b0(pretrained=True)
```

---

**CNN architectures: standing on the shoulders of giants!**
