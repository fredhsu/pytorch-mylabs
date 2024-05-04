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
            # Output layer has 10 classes
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
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

# Move the model to GPU
model = NeuralNetwork().to(device)

transform = transforms.Compose([transforms.ToTensor()])

train_ds = datasets.MNIST("../data", train=True, download=True, transform=transform)
validation_ds = datasets.MNIST("../data", train=False, transform=transform)

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True)
validation_dl = torch.utils.data.DataLoader(validation_ds, batch_size=256, shuffle=True)

optimizer = optim.SGD(model.parameters(), lr=0.1)

for x, y in train_dl:
    # Need to move the data to GPU
    x = x.to(device)
    y = y.to(device)
    optimizer.zero_grad()

    # Running the model with input data
    output = model(x)

    # Using NLL as it is well suited for classification problems with C classes
    # Also solved the dimension mismatch issue
    loss = F.nll_loss(output, y).backward()
    optimizer.step()

# turn off gradient calculation for inference
with torch.no_grad():
    for x, y in validation_dl:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        # Softmax because we are doing classification
        pred_probab = nn.Softmax(dim=1)(logits)
        # Choose the class with highest probability
        y_pred = pred_probab.argmax(1)
        acc = (y_pred == y).float().mean()
        print(f"Accuracy: {acc.item() * 100}%")
