# Faster R-CNN: Two-Stage Object Detection from First Principles

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Paper](https://img.shields.io/badge/Paper-NeurIPS_2015-blue)](https://arxiv.org/abs/1506.01497)
[![Implementation](https://img.shields.io/badge/Implementation-From_Scratch-orange)]()

A **from-scratch** PyTorch implementation of Faster R-CNN, designed to deeply understand the mathematics and engineering behind two-stage object detection.

> **Original Paper**: [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497) (Ren et al., NeurIPS 2015)

---

## 🔬 Research Motivation

**The Evolution:**
- **R-CNN** (2014): Extract ~2000 region proposals with Selective Search → CNN features → SVM classifier. **Bottleneck**: Selective Search is slow.
- **Fast R-CNN** (2015): Share CNN computation across proposals. **Bottleneck**: Still uses Selective Search.
- **Faster R-CNN** (2015): Replace Selective Search with a learned **Region Proposal Network (RPN)**. **Result**: End-to-end trainable, 250ms/image.

**Key Insight:** Region proposals can be learned from the same CNN features used for classification, making detection fully differentiable.

---

## 🧮 Mathematical Formulations

### Anchor Box Generation

At each feature map location $(x, y)$, generate $k$ anchor boxes with:
- **Scales**: $s \in \{2, 4, 6\}$ (relative to feature map)
- **Aspect Ratios**: $r \in \{0.5, 1.0, 1.5\}$

Each anchor is defined as:

$$\text{anchor} = \left(x - \frac{w}{2}, y - \frac{h}{2}, x + \frac{w}{2}, y + \frac{h}{2}\right)$$

where $w = s \cdot r$ and $h = s$.

**Total anchors**: For a $20 \times 15$ feature map with 9 anchors each = **2,700 proposals/image**.

### Intersection over Union (IoU)

$$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{\text{Intersection Area}}{\text{Union Area}}$$

**Anchor assignment:**
- **Positive**: IoU > 0.7 with any GT box, OR highest IoU for a GT box
- **Negative**: IoU < 0.3 with all GT boxes
- **Ignore**: 0.3 ≤ IoU ≤ 0.7 (not used in training)

### Bounding Box Regression

Transform from anchor $A = (A_x, A_y, A_w, A_h)$ to ground truth $G = (G_x, G_y, G_w, G_h)$:

$$t_x = \frac{G_x - A_x}{A_w}, \quad t_y = \frac{G_y - A_y}{A_h}$$

$$t_w = \log\left(\frac{G_w}{A_w}\right), \quad t_h = \log\left(\frac{G_h}{A_h}\right)$$

**Why log for width/height?** Ensures predictions are always positive and handles large scale variations.

**Inverse transform (inference):**

$$P_x = A_x + t_x \cdot A_w, \quad P_y = A_y + t_y \cdot A_h$$

$$P_w = A_w \cdot e^{t_w}, \quad P_h = A_h \cdot e^{t_h}$$

### Loss Functions

**RPN Classification Loss (Binary Cross-Entropy):**

$$\mathcal{L}_{cls} = -\frac{1}{N_{cls}} \sum_i \left[ y_i \log(p_i) + (1-y_i) \log(1-p_i) \right]$$

**RPN Regression Loss (Smooth L1):**

$$\mathcal{L}_{reg} = \frac{1}{N_{reg}} \sum_i \text{smooth}_{L1}(t_i - t_i^*)$$

where:

$$\text{smooth}_{L1}(x) = \begin{cases} 
0.5x^2 & \text{if } |x| < 1 \\
|x| - 0.5 & \text{otherwise}
\end{cases}$$

**Why Smooth L1?** More robust to outliers than L2, while having stable gradients near zero.

**Total Loss:**

$$\mathcal{L} = \mathcal{L}_{cls} + \lambda \mathcal{L}_{reg}$$

where $\lambda = 5$ balances the two losses.

### Non-Maximum Suppression (NMS)

```
1. Sort proposals by confidence score
2. Select highest-scoring box, add to output
3. Remove all boxes with IoU > threshold (0.7) with selected box
4. Repeat until no boxes remain
```

---

## 🏗️ Architecture

```
Input Image (H×W×3)
        │
        ▼
┌─────────────────────────────────────────────────┐
│   ResNet-50 Backbone (layers 1-4)               │
│   Output: (B, 2048, H/32, W/32)                 │
│   Purpose: Extract rich semantic features       │
└─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────┐
│   Region Proposal Network (RPN)                 │
│   ┌─────────────────────────────────────────┐   │
│   │ 3×3 Conv (512 channels)                 │   │
│   │ ├── Objectness Head: 1×1 Conv → A       │   │
│   │ │   (Is there an object?)               │   │
│   │ └── Regression Head: 1×1 Conv → 4A      │   │
│   │     (Box offsets: tx, ty, tw, th)       │   │
│   └─────────────────────────────────────────┘   │
│   ↓                                             │
│   Generate proposals + NMS → Top-N boxes        │
└─────────────────────────────────────────────────┘
        │ Proposals (~300 boxes)
        ▼
┌─────────────────────────────────────────────────┐
│   ROI Pooling                                   │
│   Extract fixed 7×7 features for each proposal  │
│   (Handles variable-sized boxes)                │
└─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────┐
│   Classification Head                           │
│   AvgPool → FC(512) → FC(N_classes)            │
│   Output: Class probabilities                   │
└─────────────────────────────────────────────────┘
```

---

## 📦 Implementation Details

### Core Components

| Module | Parameters | Key Design Choices |
|--------|------------|-------------------|
| `FeatureExtractor` | ResNet-50 (23M) | Layers 1-4, all trainable |
| `ProposalModule` | ~2.4M | 512 hidden dim, dropout=0.3 |
| `ClassificationModule` | ~1.1M | 7×7 ROI, avg pool |

### Anchor Box Visualization

```
Feature Map Position (x, y):

         r=0.5    r=1.0    r=1.5
        ┌─┐      ┌──┐     ┌───┐
s=2     │ │      │  │     │   │
        └─┘      └──┘     └───┘

        ┌──┐     ┌────┐   ┌─────┐
s=4     │  │     │    │   │     │
        └──┘     └────┘   └─────┘

        ┌───┐    ┌──────┐ ┌───────┐
s=6     │   │    │      │ │       │
        └───┘    └──────┘ └───────┘
```

### Memory-Efficient IoU Computation

```python
# Vectorized IoU for (N anchors × M GT boxes)
def get_iou_mat(anc_boxes, gt_bboxes):
    # anc_boxes: (N, 4), gt_bboxes: (M, 4)
    # Returns: (N, M) IoU matrix
    return ops.box_iou(anc_boxes, gt_bboxes)  # O(NM) but vectorized
```

---

## 💡 Insights from Implementation

### What I Learned

1. **Anchor assignment is critical**: The IoU thresholds (0.7/0.3) create a "gray zone" that prevents noisy gradients from ambiguous boxes.

2. **Balanced sampling matters**: Negative anchors vastly outnumber positives (~100:1). Random sampling of negatives to match positives prevents the model from predicting "no object" everywhere.

3. **Spatial scale alignment**: `roi_pool` requires `spatial_scale=1/32` to match ResNet's downsampling factor. Wrong scale = garbage features.

4. **Gradient flow through proposals**: During training, proposals are detached (`proposals.detach()`) to prevent gradients from flowing back through NMS (non-differentiable).

### Challenges Encountered

| Challenge | Solution |
|-----------|----------|
| Memory explosion with all anchors | Sample 256 anchors/image (128 pos + 128 neg) |
| NMS is non-differentiable | Detach proposals before classification |
| Coordinate system confusion | Consistent use of `xyxy` format, project with scale factors |
| Class imbalance | Balanced sampling + focal loss (future work) |

### Debugging Tips

```python
# Sanity check: visualize anchors
fig, ax = plt.subplots()
display_bbox(anchors[:100], fig, ax, color='blue')  # Sample anchors
display_bbox(gt_boxes, fig, ax, color='red')        # Ground truth
```

---

## 🔬 Ablation Study Design

### Planned Experiments

| Experiment | Variable | Expected Result |
|------------|----------|-----------------|
| **Anchor scales** | {64, 128, 256} vs {2, 4, 6} | Larger scales for larger objects |
| **IoU thresholds** | 0.5/0.1 vs 0.7/0.3 | Lower thresholds → more positives but noisier |
| **Loss weight λ** | 1 vs 5 vs 10 | Higher λ → tighter boxes but slower convergence |
| **NMS threshold** | 0.5 vs 0.7 vs 0.9 | Lower → fewer proposals, higher precision |
| **ROI size** | 5×5 vs 7×7 vs 14×14 | Larger → more detail but slower |

### Baseline Comparisons

| Method | mAP (VOC) | Speed |
|--------|-----------|-------|
| Selective Search + CNN | 58.5% | ~47s/image |
| Fast R-CNN | 66.9% | ~2s/image |
| **Faster R-CNN** | **69.9%** | **0.2s/image** |

---

## 📊 Training Pipeline

```
Epoch Loop:
│
├── For each batch:
│   ├── Extract features (backbone)
│   ├── Generate anchors (grid × scales × ratios)
│   ├── Compute IoU matrix (anchors × GT)
│   ├── Assign labels (pos/neg/ignore)
│   ├── Sample 256 anchors (balanced)
│   ├── RPN forward → conf scores, offsets
│   ├── RPN loss = BCE + Smooth L1
│   ├── Generate proposals (anchor + offset)
│   ├── NMS → top-N proposals
│   ├── ROI pooling → fixed-size features
│   ├── Classification head → class scores
│   ├── Classification loss = CrossEntropy
│   └── Total loss = RPN loss + Cls loss
│
└── Backprop and optimize
```

---

## 🔧 Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `pos_thresh` | 0.7 | Standard from paper |
| `neg_thresh` | 0.3 | Creates clear separation |
| `λ (loss weight)` | 5 | Regression needs more weight |
| `n_anchors` | 9 | 3 scales × 3 ratios |
| `nms_thresh` | 0.7 | Balances recall/precision |
| `conf_thresh` | 0.5 | Filter low-confidence proposals |

---

## 📚 Citation

```bibtex
@inproceedings{ren2015faster,
  title={Faster {R-CNN}: Towards Real-Time Object Detection 
         with Region Proposal Networks},
  author={Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
  booktitle={NeurIPS},
  pages={91--99},
  year={2015}
}
```

---

## 🔮 Future Directions

1. **Feature Pyramid Network (FPN)**: Multi-scale detection for small objects
2. **Focal Loss**: Address class imbalance more elegantly
3. **Deformable Convolutions**: Better handle geometric variations
4. **Cascade R-CNN**: Progressive refinement for tighter boxes
5. **ROI Align**: Replace ROI Pool for pixel-perfect alignment

---

## 📄 License

MIT License
