"""Unit 14: a multilayer perceptron in PyTorch for handwritten digits.

The 8 x 8 digit images ship with scikit-learn (1,797 images). The
larger 28 x 28 MNIST set is on Kaggle: hojjatk/mnist-dataset.
Runs on the CPU in under a minute; no GPU is needed.
"""
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn

torch.manual_seed(465)
X, y = load_digits(return_X_y=True)
X = X / 16.0                                   # pixel values to [0, 1]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          stratify=y, random_state=465)
X_tr, X_te = (torch.tensor(a, dtype=torch.float32) for a in (X_tr, X_te))
y_tr, y_te = torch.tensor(y_tr), torch.tensor(y_te)

model = nn.Sequential(                         # 64 -> 64 -> 32 -> 10
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 32), nn.ReLU(),
    nn.Linear(32, 10))                         # one score per digit
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
print("trainable parameters:",
      sum(p.numel() for p in model.parameters()))

for epoch in range(1, 31):
    model.train()
    total = 0.0
    for idx in torch.randperm(len(X_tr)).split(64):   # mini-batches
        opt.zero_grad()
        loss = loss_fn(model(X_tr[idx]), y_tr[idx])
        loss.backward()                        # backpropagation
        opt.step()                             # update the weights
        total += loss.item() * len(idx)
    if epoch in (1, 5, 10, 30):
        model.eval()
        with torch.no_grad():
            acc = (model(X_te).argmax(dim=1) == y_te).float().mean()
        print(f"epoch {epoch:2d}: training loss {total / len(X_tr):.4f}, "
              f"test accuracy {acc:.3f}")
