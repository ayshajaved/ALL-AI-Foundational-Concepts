# Practical Workflows - Neural Network Basics

> **Building your first neural networks** - PyTorch and TensorFlow essentials

---

## 🎯 PyTorch Basics

### Complete Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. Data
X_train = torch.randn(1000, 20)
y_train = torch.randint(0, 2, (1000,))

dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 2. Model
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleNN()

# 3. Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training Loop
for epoch in range(10):
    for batch_x, batch_y in dataloader:
        # Forward
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 5. Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_train)
    accuracy = (predictions.argmax(1) == y_train).float().mean()
    print(f"Accuracy: {accuracy:.4f}")
```

---

## 📊 TensorFlow/Keras Basics

```python
import tensorflow as tf
from tensorflow import keras

# 1. Data
X_train = tf.random.normal((1000, 20))
y_train = tf.random.uniform((1000,), maxval=2, dtype=tf.int32)

# 2. Model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(20,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])

# 3. Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Train
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=10,
    validation_split=0.2
)

# 5. Evaluate
loss, accuracy = model.evaluate(X_train, y_train)
print(f"Accuracy: {accuracy:.4f}")
```

---

## 🎯 Debugging Tips

### Check Shapes
```python
# PyTorch
print(f"Input shape: {X_train.shape}")
print(f"Output shape: {model(X_train[:1]).shape}")

# TensorFlow
model.summary()
```

### Monitor Gradients
```python
# PyTorch
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")
```

### Overfit Small Batch
```python
# Test if model can learn
small_batch = X_train[:10]
small_labels = y_train[:10]

for i in range(100):
    outputs = model(small_batch)
    loss = criterion(outputs, small_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i % 10 == 0:
        print(f"Loss: {loss.item():.4f}")
```

---

**Start building neural networks today!**
