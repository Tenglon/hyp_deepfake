import os
import time
import torch
import torchvision
import argparse
from timm import utils

from torchvision.models import resnet18, ResNet18_Weights
from frame_loader import RealVideoFrameDataset

parser = argparse.ArgumentParser()

parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--epochs", type=int, default=20)

parser.add_argument("--opt", type=str, default="adam")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--save", action="store_const", const=True)
parser.add_argument("--num-workers", type=int, default=8)

parser.add_argument("--geometry", type=str, default="euc")
parser.add_argument("--emb_dim", type=int, default=512)
parser.add_argument("--T", type=float, default=1.0)
parser.add_argument("--c", type=float, default=1.0)

args = parser.parse_args()

root = '/ssd/ff++23/tlong_all_frames/original'

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224), antialias=True),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = RealVideoFrameDataset(os.path.join(root, 'train'), transform=transform)
testset = RealVideoFrameDataset(os.path.join(root, 'test'), transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

model0 = resnet18(weights=ResNet18_Weights.DEFAULT)
backbone = torch.nn.Sequential(*list(model0.children())[:-1])
head = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(512, args.emb_dim),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(args.emb_dim, 1000),
    torch.nn.Softmax(dim=-1)
)

model = torch.nn.Sequential(backbone, head).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = torch.nn.CrossEntropyLoss()

best_avg_metrics = {}

for epoch in range(args.epochs):

    epoch_start = time.time()
    model.train()

    metrics = {"losses": utils.AverageMeter(), "top1": utils.AverageMeter()}

    for i, (x, y) in enumerate(trainloader):
        x = x.cuda()
        y = y.cuda()

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        acc1, acc5 = utils.accuracy(logits, y, topk=(1, 5))
        metrics["losses"].update(loss.data.item(), x.size(0))
        metrics["top1"].update(acc1.item(), logits.size(0))

        if i % 50 == 0:
            print(f"Epoch {epoch} Iter {i}:  Time: {time.time() - epoch_start:.3f}  Loss: {metrics['losses'].avg:>7.4f}  Acc@1: {metrics['top1'].avg:>7.4f}")

    metrics = {"losses": utils.AverageMeter(), "top1": utils.AverageMeter()}

    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(testloader):
            x = x.cuda()
            y = y.cuda()

            logits = model(x)
            loss = criterion(logits, y)

            acc1, acc5 = utils.accuracy(logits, y, topk=(1, 5))
            metrics["losses"].update(loss.data.item(), x.size(0))
            metrics["top1"].update(acc1.item(), logits.size(0))

        if (not best_avg_metrics or metrics['top1'].avg > best_avg_metrics['top1']):
            best_avg_metrics = {k: metrics[k].avg for k in metrics}
            best_model_state = model.state_dict()

    print(f"Epoch {epoch}:  Time: {time.time() - epoch_start:.3f}  Loss: {metrics['losses'].avg:>7.4f}  Acc@1: {metrics['top1'].avg:>7.4f}")

exp_dir = os.path.join("frame_experiments", args.geometry ,f"emb_dim_{args.emb_dim}_T_{args.T}_c_{args.c}")
if args.save:
    torch.save(best_model_state, os.path.join(exp_dir, "model.pt"))

