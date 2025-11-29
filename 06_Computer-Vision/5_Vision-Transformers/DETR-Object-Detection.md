# DETR (Detection Transformer)

> **End-to-End Object Detection** - Goodbye NMS and Anchors

---

## 🤯 The Revolution

Traditional Detection (YOLO, Faster R-CNN):
- Anchors, NMS, IoU Thresholds.
- Many heuristics.

**DETR (DEtection TRansformer):**
- **Input:** Image.
- **Output:** Set of Boxes.
- **No Anchors. No NMS.**

---

## 🏗️ Architecture

1.  **Backbone (ResNet):** Extract features ($H/32 \times W/32$).
2.  **Transformer Encoder:** Refine features with global attention.
3.  **Object Queries:**
    - A fixed set of learnable vectors (e.g., $N=100$).
    - Each query asks the Decoder: "Is there an object in my region?"
4.  **Transformer Decoder:**
    - Takes Object Queries and Encoder Memory.
    - Outputs $N$ box predictions.
5.  **FFN Heads:**
    - Class Head: "Cat", "Dog", or "No Object" ($\emptyset$).
    - Box Head: $x, y, w, h$.

---

## 🤝 Bipartite Matching (Hungarian Loss)

The model outputs 100 boxes. The image has 3 objects.
How do we calculate loss?

**Hungarian Algorithm:**
Finds the optimal 1-to-1 assignment between Predictions and Ground Truth that minimizes cost.
- Pred 1 $\leftrightarrow$ Object A
- Pred 5 $\leftrightarrow$ Object B
- Pred 99 $\leftrightarrow$ Object C
- All others $\leftrightarrow$ $\emptyset$ (No Object)

**Loss:**
Classification Loss + Box L1 Loss + GIoU Loss.

---

## 💻 PyTorch Implementation

```python
import torch
from torch import nn
from torchvision.models import resnet50

class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        # Backbone
        self.backbone = resnet50(pretrained=True)
        self.conv = nn.Conv2d(2048, hidden_dim, 1) # Project to hidden_dim
        
        # Transformer
        self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        
        # Prediction Heads
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1) # +1 for 'No Object'
        self.linear_bbox = nn.Linear(hidden_dim, 4)
        
        # Object Queries
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        x = self.backbone(inputs)
        h = self.conv(x)
        H, W = h.shape[-2:]
        
        # Positional Encoding (Sine/Cosine)
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # Transformer
        h = self.transformer(pos + h.flatten(2).permute(2, 0, 1), self.query_pos.unsqueeze(1))
        
        return self.linear_class(h), self.linear_bbox(h).sigmoid()
```

---

## 🎓 Interview Focus

1.  **Why is DETR slow to converge?**
    - The attention maps start uniform. It takes a long time for the object queries to learn where to look.
    - **Deformable DETR** fixes this by attending only to a few key points (sparse attention), converging 10x faster.

2.  **What are Object Queries?**
    - Learnable slots that specialize. One query might learn to look for "small objects in the corner", another for "large objects in the center".

---

**DETR: Treating detection as a set prediction problem!**
