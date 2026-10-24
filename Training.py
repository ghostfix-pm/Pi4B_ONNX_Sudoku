import os, random, numpy as np
import torch as tr
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.export import Dim
import onnxruntime as ort

from digitnet_model import DigitNet

# Configuration
random.seed(0); np.random.seed(0); tr.manual_seed(0)
DEVICE = "cpu"           # keep CPU for portability
DATA_DIR = "data_v4"        # data/train/0..9, data/val/0..9
ARTI_DIR = "artifacts_v4"
os.makedirs(ARTI_DIR, exist_ok=True)

BATCH = 256
EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.05

# Datasets (force grayscale to match model)
tfm_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),   # force 1-channel input
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.15),
    transforms.RandomAffine(degrees=5, translate=(0.02,0.02), scale=(0.95,1.05)),
])
tfm_val = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),   # force 1-channel input
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=tfm_train)
val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=tfm_val)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)

def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    with tr.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total   += y.numel()
    return correct / total if total else 0.0

def main():
    # Sanity: verify loader channels
    xb, _ = next(iter(train_loader))
    print("Train batch shape:", xb.shape)  # expect [B, 1, 28, 28]

    model = DigitNet(num_classes=10).to(DEVICE)
    opt = tr.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_acc, best_state = 0.0, None
    for ep in range(1, EPOCHS + 1):
        model.train()
        for x, y in tqdm(train_loader, desc=f"epoch {ep}/{EPOCHS}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=LABEL_SMOOTH)
            loss.backward()
            opt.step()

        acc = evaluate(model, val_loader)
        print(f"val acc: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best val acc: {best_acc:.4f}")

    # Save PyTorch weights (optional)
    tr.save(model.state_dict(), os.path.join(ARTI_DIR, "digitnet_best.pth"))

    # Export to ONNX with dynamic batch using the Dynamo API
    model.eval()
    dummy = tr.randn(1, 1, 28, 28)
    onnx_path = os.path.join(ARTI_DIR, "sudoku_digitnet.onnx")

    tr.onnx.export(
        model,                       # model
        dummy,                       # input tensor corresponds to arg name 'x'
        onnx_path,                   # output .onnx
        input_names=["input"],       # ONNX *display* name (independent of arg name)
        output_names=["logits"],
        opset_version=19,
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
        dynamo=True,
        dynamic_shapes={"x": {0: Dim("N")}}
    )

    print("Exported:", onnx_path)

    # quick parity with a big batch (proves dynamic batch works)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    x = tr.randn(512, 1, 28, 28)
    onnx_logits = sess.run(["logits"], {"input": x.numpy().astype(np.float32)})[0]
    pt_logits   = model(x).detach().numpy()
    print("Parity MAE:", np.mean(np.abs(onnx_logits - pt_logits)))

if __name__ == "__main__":
    main()
