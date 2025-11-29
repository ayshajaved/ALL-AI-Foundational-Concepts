# Advanced Training Techniques

> **Scaling and stabilizing training** - Mixed Precision, Gradient Accumulation, and Checkpointing

---

## ⚡ Mixed Precision Training (FP16)

Standard training uses **FP32** (32-bit float).
**FP16** (16-bit float) uses half the memory and is 2-4x faster on Tensor Cores (NVIDIA GPUs).

**Challenge:** FP16 has limited range (underflow/overflow).
**Solution:** Gradient Scaling.

### PyTorch AMP (Automatic Mixed Precision)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for inputs, targets in dataloader:
    optimizer.zero_grad()
    
    # Runs forward pass in FP16
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    # Scales loss (to prevent underflow) and calls backward
    scaler.scale(loss).backward()
    
    # Unscales gradients and updates weights
    scaler.step(optimizer)
    scaler.update()
```

---

## 📦 Gradient Accumulation

**Problem:** GPU memory limits batch size (e.g., can only fit batch=8), but you need batch=64 for stable convergence.
**Solution:** Accumulate gradients over multiple small steps before updating weights.

```python
accumulation_steps = 8  # Effective batch size = 8 * real_batch_size

for i, (inputs, targets) in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Normalize loss
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 💾 Gradient Checkpointing

**Problem:** Storing all intermediate activations for backprop consumes huge memory ($O(N)$ layers).
**Solution:** Don't store intermediates. Recompute them during backward pass.
**Tradeoff:** Saves 4-5x memory, but 20-30% slower.

```python
from torch.utils.checkpoint import checkpoint

class LargeModel(nn.Module):
    def forward(self, x):
        x = self.layer1(x)
        # Checkpoint layer 2
        x = checkpoint(self.layer2, x)
        x = self.layer3(x)
        return x
```

---

## 🎓 Interview Focus

1.  **What is the benefit of Mixed Precision?**
    - Faster training (Tensor Cores) and lower memory usage (allows larger batch sizes).

2.  **Why do we need Gradient Scaling in FP16?**
    - Gradients can be very small ($< 2^{-24}$), causing them to vanish to zero in FP16. Scaling shifts them into the representable range.

3.  **When to use Gradient Accumulation?**
    - When your GPU memory is too small to fit the desired batch size.

---

**Training at Scale: Doing more with less!**
