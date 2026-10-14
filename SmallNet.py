import torch, torch.nn as nn
import onnx, onnxruntime as ort
import numpy as np

class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(8, 10)
        )
    def forward(self, x): return self.seq(x)

m = SmallNet().eval()
dummy = torch.randn(1, 1, 28, 28)

torch.onnx.export(
    m, dummy, "smallnet.onnx",
    input_names=["input"], output_names=["logits"],
    opset_version=19,
    do_constant_folding=True,
    keep_initializers_as_inputs=False,
    dynamo=True
)

# Quick sanity run
sess = ort.InferenceSession("smallnet.onnx", providers=["CPUExecutionProvider"])
out = sess.run(["logits"], {"input": dummy.numpy()})[0]
print("Output shape:", out.shape, "Providers:", sess.get_providers())
