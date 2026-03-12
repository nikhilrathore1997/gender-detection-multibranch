import os
import json
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import random

# ----------------------
#  CONFIG
# ----------------------
DATA_DIR    = "dataset"        # folder with sample_XXXXX subfolders
VAL_DIR     = "val"            # validation folder (same structure)
BATCH_SIZE  = 64
NUM_EPOCHS  = 25
LR          = 3e-4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_BEST   = "best_model.pth"
SAVE_LAST   = "last_model.pth"


# ----------------------
#  DATASET
# ----------------------
class GenderMultiBranchDataset(Dataset):
    """
    Each sample is a folder:
        sample_XXXXX/
            full.jpg      (full body crop, any size — resized in __getitem__)
            face.jpg      (face crop, or black image if face not visible)
            meta.json     {"gender": 0 or 1, "face_visible": 0 or 1}
    """

    def __init__(self, root):
        self.samples = sorted([
            os.path.join(root, d)
            for d in os.listdir(root)
            if d.startswith("sample_")
        ])
        if len(self.samples) == 0:
            raise RuntimeError(f"No sample folders found in: {root}")

        # Full body augmentations
        self.full_aug = transforms.Compose([
            transforms.RandomResizedCrop((224, 224), scale=(0.7, 1.0)),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
        ])

        # Face augmentations (lighter)
        self.face_aug = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        ])

        self.face_resize = transforms.Resize((112, 112))

        self.to_tensor = transforms.ToTensor()

        # ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder = self.samples[idx]

        # --- Load images ---
        full_path = os.path.join(folder, "full.jpg")
        face_path = os.path.join(folder, "face.jpg")

        full_bgr = cv2.imread(full_path)
        face_bgr = cv2.imread(face_path)

        if full_bgr is None:
            raise FileNotFoundError(f"Missing full.jpg in {folder}")
        if face_bgr is None:
            raise FileNotFoundError(f"Missing face.jpg in {folder}")

        full_img = Image.fromarray(cv2.cvtColor(full_bgr, cv2.COLOR_BGR2RGB))
        face_img = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))

        # --- Load metadata ---
        meta_path = os.path.join(folder, "meta.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)

        gender       = int(meta["gender"])        # 0 or 1
        face_visible = int(meta["face_visible"])  # 0 or 1

        # --- Synchronized horizontal flip ---
        if random.random() < 0.5:
            full_img = transforms.functional.hflip(full_img)
            face_img = transforms.functional.hflip(face_img)

        # --- Full body augmentation ---
        full_img = self.full_aug(full_img)

        # --- Face augmentation (only if face is visible) ---
        if face_visible == 1:
            face_img = self.face_aug(face_img)
        else:
            # Zero out face image when face not visible
            face_img = self.face_resize(face_img)
            face_img = Image.fromarray(
                torch.zeros(3, 112, 112).permute(1, 2, 0).byte().numpy()
            )

        # --- To tensor + normalize ---
        full_tensor = self.normalize(self.to_tensor(full_img))
        face_tensor = self.normalize(self.to_tensor(face_img))

        return (
            full_tensor,                                        # [3, 224, 224]
            face_tensor,                                        # [3, 112, 112]
            torch.tensor(face_visible, dtype=torch.float32),   # scalar
            torch.tensor(gender,       dtype=torch.long),      # scalar
        )


# ----------------------
#  VALIDATION DATASET
# ----------------------
class GenderMultiBranchValDataset(GenderMultiBranchDataset):
    """
    Same structure as training dataset but NO augmentation.
    """

    def __getitem__(self, idx):
        folder = self.samples[idx]

        full_bgr = cv2.imread(os.path.join(folder, "full.jpg"))
        face_bgr = cv2.imread(os.path.join(folder, "face.jpg"))

        if full_bgr is None:
            raise FileNotFoundError(f"Missing full.jpg in {folder}")
        if face_bgr is None:
            raise FileNotFoundError(f"Missing face.jpg in {folder}")

        full_img = Image.fromarray(cv2.cvtColor(full_bgr, cv2.COLOR_BGR2RGB))
        face_img = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))

        with open(os.path.join(folder, "meta.json"), "r") as f:
            meta = json.load(f)

        gender       = int(meta["gender"])
        face_visible = int(meta["face_visible"])

        # Resize only — no augmentation
        full_img = transforms.functional.resize(full_img, (224, 224))
        face_img = transforms.functional.resize(face_img, (112, 112))

        if face_visible == 0:
            face_img = Image.fromarray(
                torch.zeros(3, 112, 112).permute(1, 2, 0).byte().numpy()
            )

        full_tensor = self.normalize(self.to_tensor(full_img))
        face_tensor = self.normalize(self.to_tensor(face_img))

        return (
            full_tensor,
            face_tensor,
            torch.tensor(face_visible, dtype=torch.float32),
            torch.tensor(gender,       dtype=torch.long),
        )


# ----------------------
#  MODEL
# ----------------------
class MultiBranchGenderNet(nn.Module):
    """
    Two-branch gender classifier:
        - Body branch:  EfficientNet-B0 on full person crop (224x224)
        - Face branch:  EfficientNet-B0 on face crop        (112x112)
        - Gate:         learned from body features + face_visible flag
                        decides how much to trust each branch per sample
    """

    def __init__(self, backbone_name="efficientnet_b0"):
        super().__init__()

        # Body backbone
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0, global_pool="avg"
        )
        backbone_dim = self.backbone.num_features  # 1280 for efficientnet_b0

        self.body_head = nn.Sequential(
            nn.Linear(backbone_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

        # Face backbone (separate — face and body need different feature spaces)
        self.face_backbone = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=0, global_pool="avg"
        )
        face_dim = self.face_backbone.num_features  # 1280

        self.face_head = nn.Sequential(
            nn.Linear(face_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

        # Gate: inputs are body features + face_visible flag (1 value)
        # Output: single logit → sigmoid → alpha (0=trust body, 1=trust face)
        self.gate = nn.Sequential(
            nn.Linear(backbone_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            # No sigmoid here — we apply it manually so we can
            # also use BCEWithLogitsLoss for gate supervision
        )

    def forward(self, full_img, face_img, face_visible):
        # Body branch
        body_feat   = self.backbone(full_img)           # [B, 1280]
        body_logits = self.body_head(body_feat)         # [B, 2]

        # Face branch
        face_feat   = self.face_backbone(face_img)      # [B, 1280]
        face_logits = self.face_head(face_feat)         # [B, 2]

        # Gate
        gate_in      = torch.cat([body_feat, face_visible.unsqueeze(1)], dim=1)  # [B, 1281]
        alpha_logits = self.gate(gate_in)               # [B, 1]  — raw logit
        alpha        = torch.sigmoid(alpha_logits)      # [B, 1]  — 0..1

        # Weighted fusion
        final_logits = alpha * face_logits + (1 - alpha) * body_logits  # [B, 2]

        return final_logits, body_logits, face_logits, alpha, alpha_logits


# ----------------------
#  VALIDATION
# ----------------------
@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    correct = 0
    total   = 0

    # Per-group tracking
    face_correct = face_total = 0
    noface_correct = noface_total = 0

    for full_img, face_img, face_visible, gender in val_loader:
        full_img     = full_img.to(device)
        face_img     = face_img.to(device)
        face_visible = face_visible.to(device)
        gender       = gender.to(device)

        final_logits, _, _, _, _ = model(full_img, face_img, face_visible)
        preds = torch.argmax(final_logits, dim=1)

        correct += (preds == gender).sum().item()
        total   += gender.size(0)

        # Face-visible group
        fv_mask = face_visible.bool()
        if fv_mask.sum() > 0:
            face_correct += (preds[fv_mask] == gender[fv_mask]).sum().item()
            face_total   += fv_mask.sum().item()

        # Face-not-visible group
        nv_mask = ~fv_mask
        if nv_mask.sum() > 0:
            noface_correct += (preds[nv_mask] == gender[nv_mask]).sum().item()
            noface_total   += nv_mask.sum().item()

    metrics = {
        "accuracy":        correct / total if total > 0 else 0,
        "face_accuracy":   face_correct / face_total if face_total > 0 else 0,
        "noface_accuracy": noface_correct / noface_total if noface_total > 0 else 0,
        "face_total":      face_total,
        "noface_total":    noface_total,
    }
    return metrics


# ----------------------
#  TRAINING
# ----------------------
def train():
    print(f"Device: {DEVICE}")

    # --- Datasets ---
    train_dataset = GenderMultiBranchDataset(DATA_DIR)
    val_dataset   = GenderMultiBranchValDataset(VAL_DIR)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # --- Model ---
    model     = MultiBranchGenderNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # --- Loss functions ---
    # FIX: use reduction='none' so we can mask per-sample before averaging
    criterion_mean = nn.CrossEntropyLoss()                      # for L_final, L_body
    criterion_none = nn.CrossEntropyLoss(reduction='none')      # for L_face (masked)
    criterion_gate = nn.BCEWithLogitsLoss()                     # for gate supervision

    scaler       = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None
    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()

        total_loss    = 0.0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]")

        for full_img, face_img, face_visible, gender in pbar:
            full_img     = full_img.to(DEVICE)
            face_img     = face_img.to(DEVICE)
            face_visible = face_visible.to(DEVICE)   # [B]  float 0/1
            gender       = gender.to(DEVICE)         # [B]  long

            optimizer.zero_grad()

            use_amp = (DEVICE.type == "cuda")

            with torch.amp.autocast("cuda", enabled=use_amp):
                final_logits, body_logits, face_logits, alpha, alpha_logits = \
                    model(full_img, face_img, face_visible)

                # --- L_final: main classification loss ---
                L_final = criterion_mean(final_logits, gender)

                # --- L_body: keep body branch independently trained ---
                L_body = criterion_mean(body_logits, gender)

                # --- L_face: only penalize when face IS visible ---
                # criterion_none returns [B] per-sample losses
                # multiply by face_visible to zero-out invisible-face samples
                # then mean over full batch (denominator = B, not num_visible)
                # This is intentional: rare visible faces still contribute meaningfully
                per_sample_face_loss = criterion_none(face_logits, gender)  # [B]
                L_face = (per_sample_face_loss * face_visible).mean()       # scalar

                # --- L_gate: supervise gate to learn face_visible signal ---
                # Target: alpha should be 1 when face visible, 0 when not
                gate_target = face_visible.unsqueeze(1)          # [B, 1]
                L_gate      = criterion_gate(alpha_logits, gate_target)

                # --- Total loss ---
                # Weights: L_final dominates, body/face auxiliary, gate lightly supervised
                loss = L_final + 0.5 * L_body + 0.5 * L_face + 0.1 * L_gate

            # --- Backward ---
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # --- Metrics ---
            total_loss    += loss.item()
            preds          = torch.argmax(final_logits, dim=1)
            total_correct += (preds == gender).sum().item()
            total_samples += gender.size(0)

            avg_loss = total_loss / (pbar.n + 1)
            train_acc = total_correct / total_samples
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{train_acc:.4f}"})

        scheduler.step()

        # --- Validation ---
        val_metrics = validate(model, val_loader, DEVICE)
        val_acc = val_metrics["accuracy"]

        print(
            f"\nEpoch {epoch+1} Summary:"
            f"\n  Train Acc : {train_acc:.4f}"
            f"\n  Val Acc   : {val_acc:.4f}"
            f"\n  Face Acc  : {val_metrics['face_accuracy']:.4f}  (n={val_metrics['face_total']})"
            f"\n  NoFace Acc: {val_metrics['noface_accuracy']:.4f}  (n={val_metrics['noface_total']})"
            f"\n  LR        : {scheduler.get_last_lr()[0]:.6f}"
        )

        # --- Save best ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_BEST)
            print(f"  ✓ Saved best model (val_acc={best_val_acc:.4f})")

    # --- Save last ---
    torch.save(model.state_dict(), SAVE_LAST)
    print(f"\n✓ Training complete. best_val_acc={best_val_acc:.4f}")
    print(f"  Saved: {SAVE_BEST}, {SAVE_LAST}")


if __name__ == "__main__":
    train()
