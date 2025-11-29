# Practical Workflows: Neural Style Transfer

> **The "Hello World" of Generative Vision** - Painting like Van Gogh

---

## 🎨 The Goal

Combine the **Content** of one image (Photo) with the **Style** of another (Painting).

---

## 🧠 The Algorithm (Gatys et al., 2015)

We don't train a new network. We optimize the **input image** pixels.
Using a pre-trained VGG-19 network as a feature extractor.

1.  **Content Loss:**
    - The activations of high-level layers (e.g., `conv4_2`) should match the Content Image.
    - $L_{content} = MSE(F_{generated}, F_{content})$.

2.  **Style Loss (Gram Matrix):**
    - Style is correlation between features.
    - **Gram Matrix:** $G = F \cdot F^T$. (Dot product of feature maps).
    - The Gram Matrices of multiple layers should match the Style Image.
    - $L_{style} = \sum MSE(G_{generated}, G_{style})$.

3.  **Total Loss:**
    - $L = \alpha L_{content} + \beta L_{style}$.

---

## 💻 PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image

# 1. Load VGG
vgg = models.vgg19(pretrained=True).features.eval()

# 2. Load Images
loader = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
original_img = load_image("content.jpg")
style_img = load_image("style.jpg")
generated = original_img.clone().requires_grad_(True)

# 3. Optimization Loop
optimizer = optim.Adam([generated], lr=0.01)

for step in range(3000):
    # Extract features
    gen_features = get_features(generated, vgg)
    orig_features = get_features(original_img, vgg)
    style_features = get_features(style_img, vgg)
    
    # Calculate Loss
    content_loss = torch.mean((gen_features['conv4_2'] - orig_features['conv4_2'])**2)
    
    style_loss = 0
    for layer in style_layers:
        gen_gram = gram_matrix(gen_features[layer])
        style_gram = gram_matrix(style_features[layer])
        style_loss += torch.mean((gen_gram - style_gram)**2)
        
    total_loss = content_weight * content_loss + style_weight * style_loss
    
    # Update Image
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

---

## 🚀 Fast Style Transfer

Optimization takes time (1 min per image).
**Fast Style Transfer:** Train a Feed-Forward Network (CNN) to apply a *specific* style instantly.
- **Loss:** Same as above.
- **Inference:** 0.1s.

---

**Style Transfer: Making algorithms artistic!**
