# Practical Workflows - Optimization

> **Hands-on optimization in Python** - SciPy, PyTorch, JAX

---

## 🛠️ SciPy Optimization

```python
from scipy.optimize import minimize, minimize_scalar

# Unconstrained optimization
def f(x):
    return (x[0] - 1)**2 + (x[1] - 2.5)**2

result = minimize(f, x0=[0, 0], method='BFGS')
print(f"Optimal: {result.x}")

# With gradient
def grad_f(x):
    return np.array([2*(x[0]-1), 2*(x[1]-2.5)])

result = minimize(f, x0=[0, 0], jac=grad_f, method='BFGS')

# Constrained
from scipy.optimize import LinearConstraint, NonlinearConstraint

constraints = [{'type': 'ineq', 'fun': lambda x: x[0] + x[1] - 1}]
result = minimize(f, x0=[0, 0], constraints=constraints)
```

---

## 🔥 PyTorch Optimizers

```python
import torch
import torch.optim as optim

# Model parameters
model = torch.nn.Linear(10, 1)

# Optimizers
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    loss = compute_loss(model, data)
    loss.backward()
    optimizer.step()

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
```

---

## ⚡ JAX Optimization

```python
import jax
import jax.numpy as jnp
from jax import grad, jit

# Define function
def f(x):
    return jnp.sum(x**2)

# Gradient
grad_f = grad(f)

# Optimize
x = jnp.array([1.0, 2.0, 3.0])
lr = 0.1

for i in range(100):
    g = grad_f(x)
    x = x - lr * g

# Using optax
import optax

optimizer = optax.adam(learning_rate=0.001)
opt_state = optimizer.init(x)

for i in range(100):
    g = grad_f(x)
    updates, opt_state = optimizer.update(g, opt_state)
    x = optax.apply_updates(x, updates)
```

---

## 📊 Monitoring and Debugging

```python
# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Gradient accumulation
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for batch in dataloader:
    with autocast():
        loss = model(batch)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

**Master these tools for efficient ML optimization!**
