# Face Mask Detection with Faster R-CNN

A face mask detection system built with Faster R-CNN to classify three categories — proper mask wearing, no mask, and incorrectly worn mask. This was built as a **learning project** to understand two-stage object detection on a small real-world dataset.

---

![Image](https://github.com/user-attachments/assets/65e52fc7-c6bc-47d8-a4ed-4f4edcbf76a1)

## Results

| Metric | Value |
|--------|-------|
| **mAP@0.5** | **86.5%** |
| Training Epochs | 15 (early stopped) |
| Best Val Loss | 0.2803 |
| Final Train Loss | 0.0959 |
| Dataset Size | 853 images |
| Train/Val/Test Split | 597/127/129 (70/15/15) |

---

## Dataset Information

**Source:** [Kaggle Face Mask Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)

- **Total Images:** 853
- **Total Bounding Boxes:** 4,072
- **Format:** PASCAL VOC XML annotations
- **Classes:** 3 (with_mask, without_mask, mask_weared_incorrect)

### Class Distribution

```
with_mask:              3,232 (79.37%)
without_mask:             717 (17.61%)
mask_weared_incorrect:    123 ( 3.02%)
```
<img width="989" height="590" alt="Image" src="https://github.com/user-attachments/assets/1fdec436-1bbc-47d8-8051-2e3b45ae3810" />


**Note:** The dataset has a natural class imbalance. The minority class `mask_weared_incorrect` (3%) is the weakest performer in the model. We relied on COCO-pretrained transfer learning to partially handle this, but collecting more examples of this class is the proper long-term fix.

---

## Why Faster R-CNN?

| Decision | Reasoning |
|----------|-----------|
| **Two-stage detector** | Chosen specifically to learn and understand how two-stage object detection works — RPN + classification head |
| **ResNet50-FPN backbone** | FPN handles objects at multiple scales well. However in hindsight, ResNet18-FPN would have been a better fit for a dataset this small — less complexity, less overfitting risk |
| **Transfer learning from COCO** | Pretrained weights reduced training time and helped the model generalize despite the small dataset |
| **Proven architecture** | Well documented, easier to debug and understand compared to newer methods |

---

## Training Configuration

### Key Hyperparameters

- **Batch Size:** 4 (limited by GPU memory — 6GB VRAM)
- **Learning Rate:** 5e-4 with ReduceLROnPlateau scheduler
- **Optimizer:** Adam
- **Early Stopping:** Patience of 7 epochs
- **Random Seed:** 42 (for reproducibility)

### Data Augmentation
```python
RandomHorizontalFlip(0.5)        # Faces can appear mirrored
ColorJitter(brightness=0.2,      # Handle different lighting conditions
            contrast=0.2,
            saturation=0.1)
```
Augmentation was important here because 853 images is a very small dataset. These transforms increase effective variety without collecting new data.

### Why These Choices?

**Batch Size 4:**
- Tested 2, 4, and 8
- Size 8 caused GPU out-of-memory errors
- Size 2 caused unstable gradients
- 4 was the stable middle ground

**Learning Rate 5e-4:**
- 1e-3 — loss diverged (jumped around, never settled)
- 5e-4 — stable convergence
- 1e-4 — converged but very slowly
- Scheduler automatically reduced to 2.5e-4 at epoch 12

**70/15/15 Split:**
- Separate test set ensures unbiased final evaluation
- Fixed seed guarantees same split every run
- No data leakage between sets

---




## Training Curve

### What Happened During Training

**Epochs 1–8:** Both train and val loss decreasing steadily
**Epoch 8:** Best validation loss achieved (0.2803)
**Epochs 9–11:** Val loss starts rising while train loss keeps dropping → overfitting begins
**Epoch 12:** Scheduler triggers, LR reduced from 5e-4 to 2.5e-4
**Epochs 12–15:** No improvement even with lower LR
**Epoch 15:** Early stopping triggered

---

## Overfitting Analysis

### Evidence

```
Train Loss: 0.0959  ← very low
Val Loss:   0.3171  ← 3.3x higher
```

The model memorized the training data rather than learning features that generalize.

### Root Causes

1. **Dataset too small** — 853 images is not enough variety for a model this size. Object detectors typically need 5,000+ images
2. **Model too complex** — ResNet50 has too many parameters for this dataset. ResNet18 would have been a better backbone choice
3. **Limited data diversity** — mostly frontal faces, similar lighting, similar contexts

### Where the Model Fails

- **Unusual angles** — trained mostly on frontal faces, struggles with side profiles and tilted heads
- **False positives** — sometimes detects masks on ears, hair, or background objects
- **Occlusions** — hands covering face or sunglasses combined with mask confuse the model
- **Minority class** — `mask_weared_incorrect` has very few examples, model underperforms on it

---

## How to Improve

### Short-term
- Replace ResNet50 with ResNet18 backbone — reduces model complexity and overfitting risk
- Add dropout layers for regularization
- More aggressive augmentation (rotations, perspective changes, synthetic occlusions)

### Long-term
- Collect 5,000+ diverse images with varied angles, lighting, and demographics
- Add focal loss to better handle the minority class imbalance
- Once dataset is larger, explore more powerful architectures

---

## Installation

```bash
git clone https://github.com/Pooja-Vachhad/face-mask-detection.git
cd face-mask-detection
pip install -r requirements.txt
```

---

## Usage

### Train
```bash
python train.py
```
- Max 30 epochs with early stopping
- Best model saved to `best_model.pth`
- ~1.5 hours on T4 GPU

### Inference
```bash
python test.py
```

---

## What I Learned

### Technical
1. **Model size must match dataset size** — ResNet50 was overkill for 853 images
2. **Transfer learning helps a lot on small datasets** — COCO pretraining gave a strong starting point
3. **Early stopping is necessary but not sufficient** — it limits overfitting but doesn't fix the root cause
4. **Train loss alone is misleading** — always watch validation loss

### General
1. **Start with the simplest model that makes sense** — don't go heavy by default
2. **Document what went wrong honestly** — it shows deeper understanding than hiding problems
3. **mAP on test set ≠ production ready** — edge cases matter in real deployment

---

## Technical Notes

### Why Not YOLO?
YOLO is significantly faster but this project was specifically about learning how two-stage detection works. Understanding the RPN, anchor boxes, and ROI pooling in Faster R-CNN was the goal — not achieving maximum inference speed.

### Why 3 Classes?
Binary mask/no-mask misses a real common case — mask worn with nose exposed. Three classes gives more actionable output and reflects what you actually see in practice.

### Why No Class Weights?
We used COCO-pretrained transfer learning as the primary strategy for handling imbalance. The minority class `mask_weared_incorrect` still underperforms — this is acknowledged as a known limitation. The real fix is more data, not reweighting.
