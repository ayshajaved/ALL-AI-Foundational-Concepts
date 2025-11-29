# PyTorch Mastery

> **The framework of research** - Essential patterns for PyTorch development

---

## 📦 Custom Datasets

Inherit from `Dataset` and implement `__len__` and `__getitem__`.

```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label
```

---

## 🏗️ Custom Modules

Inherit from `nn.Module`. Define layers in `__init__`, logic in `forward`.

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        
    def forward(self, x):
        return torch.relu(self.layer1(x))
```

---

## 🔄 Training Loop Boilerplate

```python
def train_step(model, batch, optimizer, criterion, device):
    model.train()
    inputs, targets = batch
    inputs, targets = inputs.to(device), targets.to(device)
    
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

---

## 💾 Saving and Loading

```python
# Save weights only (Recommended)
torch.save(model.state_dict(), "model.pth")

# Load
model = MyModel()
model.load_state_dict(torch.load("model.pth"))
model.eval() # Don't forget this for inference!
```

---

**PyTorch: Flexible and Pythonic!**
