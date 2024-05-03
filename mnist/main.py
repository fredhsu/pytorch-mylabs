import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import optim


# Build the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            # nn.Linear(128, 128),
            # nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        # x = torch.flatten(x)
        logits = self.linear_relu_stack(x)
        return F.log_softmax(logits, dim=1)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
model = NeuralNetwork().to(device)

transform = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST("../data", train=True, download=True, transform=transform)
validation_ds = datasets.MNIST("../data", train=False, transform=transform)

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True)
validation_dl = torch.utils.data.DataLoader(validation_ds, batch_size=256, shuffle=True)

optimizer = optim.SGD(model.parameters(), lr=0.1)

for x, y in train_dl:
    x = x.to(device)
    y = y.to(device)
    optimizer.zero_grad()
    output = model(x)
    loss = F.nll_loss(output, y).backward()
    optimizer.step()

with torch.no_grad():
    for x, y in validation_dl:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred_probab = nn.Softmax(dim=1)(logits)
        y_pred = pred_probab.argmax(1)
        # print((y_pred == y))
        acc = (y_pred == y).float().mean()
        print(f"Accuracy: {acc.item() * 100}%")

    # print(f"Predicted class: {y_pred}")
# X = torch.rand(1, 28, 28, device=device)
# logits = model(X)
