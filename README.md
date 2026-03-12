# Gender Detection — Multi-Branch Gated Fusion Network

> **Problem solved:** Standard gender classifiers fail on CCTV footage when faces are occluded, blurry, or not visible. This model handles both cases — face visible and face not visible — using a learned gating mechanism.

---

## 🧠 The Problem With Existing Approaches

| Approach | Failure Mode |
|---|---|
| Face-only model | Fails completely when face not visible on CCTV |
| Body-only model | Lower accuracy when face IS clearly visible |
| Simple ensemble | Treats both branches equally — ignores face visibility |

**Real-world CCTV is messy.** Cameras are overhead, faces are small, people turn away. A model that only works on clean face crops is useless in production.

---

## 💡 Solution: Gated Multi-Branch Architecture

Instead of choosing one approach, this model runs **two branches simultaneously** and uses a **learnable gate** to dynamically weight their outputs based on whether the face is visible.

```
Input Frame
    │
    ├──► Full-Body Branch (EfficientNet-B0) ──► Body Logits
    │                                                │
    └──► Face Branch (EfficientNet-B0)     ──► Face Logits
              │                                      │
              └──► Gating Module ──► alpha ──► Fusion
                   (body_feat + face_visible)
                                              │
                                    Final Prediction
```

**Gate logic:**
```
final_logits = alpha * face_logits + (1 - alpha) * body_logits
```

- When face is clearly visible → gate learns to push `alpha` → 1 (trust face branch)
- When face is occluded/absent → gate learns to push `alpha` → 0 (trust body branch)
- Gate is jointly trained — not a hard rule, a learned behaviour

---

## 🏗️ Architecture Details

```python
MultiBranchGenderNet
├── backbone         EfficientNet-B0  (full body, 224×224 input)
├── body_head        Linear(1280→128→2)
├── face_backbone    EfficientNet-B0  (face crop, 112×112 input)
├── face_head        Linear(1280→64→2)
└── gate             Linear(1281→128→1) + Sigmoid
                     input = [body_features ; face_visible_flag]
```

---

## 📦 Dataset Format

Each training sample is a folder:

```
dataset/
└── sample_00001/
    ├── full.jpg      # full body crop (person bounding box)
    ├── face.jpg      # face crop (zeros if face not visible)
    └── meta.json     # {"gender": 0/1, "face_visible": 0/1}
```

- `gender`: 0 = female, 1 = male
- `face_visible`: 1 if face crop is valid, 0 if face not detected

---

## 🔁 Training Strategy

### Data Augmentation
- Full body: `RandomResizedCrop` + `ColorJitter` + synchronized horizontal flip
- Face: Lighter `ColorJitter` + same horizontal flip (synchronized with body)
- Zero-face images kept untouched when `face_visible = 0`

### Loss Function
```python
L_total = L_final + 0.5 * L_body + 0.5 * L_face_weighted + 0.1 * L_gate
```

| Loss | Purpose |
|---|---|
| `L_final` | Main classification loss on fused output |
| `L_body` | Keeps body branch independently trained |
| `L_face_weighted` | Face loss weighted by `face_visible` — only penalizes when face is present |
| `L_gate` | Supervises gate to correctly learn face visibility signal |

### Training Config
```
Backbone:     EfficientNet-B0 (pretrained ImageNet)
Optimizer:    Adam (lr=3e-4)
Batch size:   64
Epochs:       25
Mixed precision: torch.amp.autocast (faster GPU training)
Best model:   saved by validation accuracy
```

---

## 🚀 How to Run

### Install
```bash
pip install torch torchvision timm opencv-python pillow tqdm
```

### Prepare dataset
```
dataset/
├── sample_00001/
│   ├── full.jpg
│   ├── face.jpg
│   └── meta.json
├── sample_00002/
...
```

### Train
```bash
python train.py
```

### Output
```
best_model.pth   # saved when validation accuracy improves
last_model.pth   # saved at end of training
```

---

## 🔍 Why This Works Better on CCTV

| Scenario | Behaviour |
|---|---|
| Customer walks facing camera | Gate → high alpha → face branch dominates |
| Person walking away / side view | Gate → low alpha → body branch dominates |
| Overhead camera (no face visible) | face_visible=0 → gate suppresses face branch completely |
| Mixed batch during training | Both branches stay trained — no branch collapses |

---

## 📊 Key Design Decisions

**Why synchronized flip?**
Body and face crops must flip together — otherwise the model sees a left-facing body with a right-facing face, which never occurs in reality.

**Why supervise the gate with `L_gate`?**
Without gate supervision, the gate can learn to ignore `face_visible` and find other shortcuts. `L_gate` anchors its behaviour to the actual visibility signal.

**Why separate face backbone?**
Face and body features have very different spatial characteristics. A shared backbone would need to learn both simultaneously — separate backbones let each specialize.

---

## 🛠️ Stack
`PyTorch` `timm` `EfficientNet-B0` `OpenCV` `Mixed Precision Training` `Multi-task Loss`

---

## 👤 Author
Built as part of production CCTV analytics work — solving real failure cases in deployed gender detection systems for surveillance and retail applications.
