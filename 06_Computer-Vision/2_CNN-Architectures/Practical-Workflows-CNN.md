# Practical Workflows: Transfer Learning

> **Standing on the shoulders of giants** - Fine-tuning ResNet on CIFAR-100

---

## 🛠️ The Strategy

1.  **Load Pre-trained Model:** Use weights trained on ImageNet (1M images).
2.  **Freeze Feature Extractor:** Don't update the early layers (they detect edges/textures).
3.  **Replace Head:** Swap the final Linear layer (1000 classes) with a new one (100 classes).
4.  **Fine-tune:** Train the head (high LR) and maybe unfreeze the body later (low LR).

---

## 💻 Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms

# 1. Data Augmentation (Crucial for Transfer Learning)
transform = transforms.Compose([
    transforms.Resize(224), # ResNet expects 224x224
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

# 2. Load Model
model = models.resnet18(pretrained=True)

# 3. Freeze Body
for param in model.parameters():
    param.requires_grad = False

# 4. Replace Head
# ResNet stores the classifier in 'fc'
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 100) # 100 classes for CIFAR-100

model = model.cuda()

# 5. Train
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001) # Optimize ONLY the head

for epoch in range(5):
    for inputs, labels in train_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch} complete.")
```

---

## 🧠 Advanced Tricks

1.  **Differential Learning Rates:**
    - Head: `1e-3`
    - Body: `1e-5` (Unfreeze after 5 epochs).
2.  **Test Time Augmentation (TTA):**
    - During inference, predict on the image, the flipped image, and the zoomed image. Average the results.

---

**Transfer Learning: How to look like a genius with 100 images!**
