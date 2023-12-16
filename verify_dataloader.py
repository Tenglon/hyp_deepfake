

                
import torch
import torch.nn as nn
import torch.optim as optim
from timm import utils
import time
from ffppc23.dataloader import Ffplusplusc23DatasetFactory

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x

# Set the hyperparameters
input_size = 768
hidden_size = 1024
output_size = 998

# Create an instance of the MLP model
model = MLP(input_size, hidden_size, output_size)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Load the dataset using the Ffplusplusc23DatasetFactory
dataset_factory = Ffplusplusc23DatasetFactory()
train_loader, test_loader = dataset_factory.create_train_loaders(batch_size=256)

# Training loop
for epoch in range(100):
    epoch_start = time.time()
    model.train()
    for idx, (input, target) in enumerate(train_loader):
        input, target = input.cuda(), target.cuda()
        # import pdb;pdb.set_trace()
        model = model.to('cuda')  # Move the model to GPU

        output = model(input)

        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # import matplotlib.pyplot as plt

    # Initialize an empty list to store the losses
    losses = []

    # Validate on the test dataset
    metrics = {
        "losses": utils.AverageMeter(),
        "top1": utils.AverageMeter(),
        "top5": utils.AverageMeter(),
    }

    model.eval()
    with torch.no_grad():
        for input, target in test_loader:
            input, target = input.cuda(), target.cuda()
            output = model(input)

            loss = criterion(output, target)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            metrics["losses"].update(loss.data.item(), input.size(0))
            metrics["top1"].update(acc1.item(), output.size(0))
            metrics["top5"].update(acc5.item(), output.size(0))

            predicted_labels = torch.argmax(output, dim=-1)
            predicted_labels = torch.squeeze(predicted_labels)

            # Append the loss to the list
            losses.append(loss.item())

    epoch_end = time.time()
    print(f"Epoch: {epoch+1}, Loss: {metrics['losses'].avg:.4f}, Top-1 Accuracy: {metrics['top1'].avg:.2f}%, Top-5 Accuracy: {metrics['top5'].avg:.2f}%, Time: {epoch_end - epoch_start:.2f}s")

    # Plot the loss curve
    # plt.plot(losses)
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.title('Loss Curve')
    # plt.show()




