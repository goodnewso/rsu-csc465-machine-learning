"""Unit 14: PyTorch tensors and automatic differentiation (autograd)."""
import torch

torch.manual_seed(465)
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print("tensor:\n", x, "\nshape:", tuple(x.shape), "dtype:", x.dtype)
print("matrix product x @ x:\n", x @ x)

# autograd: PyTorch records operations and computes derivatives for us
w = torch.tensor(2.0, requires_grad=True)
loss = (3 * w - 12) ** 2                     # minimum at w = 4
loss.backward()
print(f"loss {loss.item():.1f}, d(loss)/dw = {w.grad.item():.1f}"
      "  (by hand: 2*(3w-12)*3 = -36)")

# the same gradient-descent loop as Unit 4, written with autograd
w = torch.tensor(0.0, requires_grad=True)
opt = torch.optim.SGD([w], lr=0.02)
for step in range(1, 51):
    opt.zero_grad()
    loss = (3 * w - 12) ** 2
    loss.backward()
    opt.step()
    if step in (1, 10, 50):
        print(f"step {step:2d}: w = {w.item():.4f}")
device = ("cuda" if torch.cuda.is_available() else
          "mps" if torch.backends.mps.is_available() else "cpu")
print("PyTorch", torch.__version__.split("+")[0], "| device available:",
      device)
