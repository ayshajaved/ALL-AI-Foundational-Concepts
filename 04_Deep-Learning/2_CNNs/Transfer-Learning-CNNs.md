# Transfer Learning for CNNs

> **Leverage pretrained models** - Fine-tuning for your task

---

## 🎯 Transfer Learning Strategies

### 1. Feature Extraction
```python
from torchvision import models

# Load pretrained ResNet
model = models.resnet50(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Only train final layer
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
```

### 2. Fine-Tuning
```python
# Unfreeze last few layers
for param in model.layer4.parameters():
    param.requires_grad = True

# Different learning rates
optimizer = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
])
```

---

## 📊 Pretrained Models

```python
# PyTorch
from torchvision import models

resnet = models.resnet50(pretrained=True)
vgg = models.vgg16(pretrained=True)
efficientnet = models.efficientnet_b0(pretrained=True)

# TensorFlow
from tensorflow.keras.applications import ResNet50

model = ResNet50(weights='imagenet', include_top=False)
```

---

## 🎯 Domain Adaptation

**When source ≠ target domain**

```python
# Gradual unfreezing
for epoch in range(num_epochs):
    if epoch == 5:
        # Unfreeze layer 4
        for param in model.layer4.parameters():
            param.requires_grad = True
    if epoch == 10:
        # Unfreeze layer 3
        for param in model.layer3.parameters():
            param.requires_grad = True
```

---

**Transfer learning: don't train from scratch!**
