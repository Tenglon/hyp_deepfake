import argparse
import json
import os
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from timm import utils
from lt_loader import create_loaders
from pmath import pair_wise_hyp, pair_wise_eud

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="ffppc23")
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=50)


parser.add_argument("--criterion", type=str, default="losses", choices=["losses", "top1"])
parser.add_argument("--data_dir", type=str, default="/ssd2/data/ffppc23/real/split_20231215/")

parser.add_argument("--opt", type=str, default="adam")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--save", action="store_const", const=True)
parser.add_argument("--num-workers", type=int, default=0)

parser.add_argument("--geometry", type=str, default="hyp")
parser.add_argument("--emb_dim", type=int, default=32)
parser.add_argument("--T", type=float, default=1.0)
parser.add_argument("--c", type=float, default=1.0)

args = parser.parse_args()

class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc0 = torch.nn.Linear(input_size, hidden_size)
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc0(x)
        x = torch.nn.ReLU()(x)
        x = self.fc1(x)
        x = torch.nn.ReLU()(x)
        x = self.fc2(x)
        x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-3)
        return x

# Create some strings for file management
now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
dir_path = os.path.dirname(os.path.realpath(__file__))
exp_dir = os.path.join(dir_path, args.geometry, args.dataset, now)
os.makedirs(exp_dir)

classes = 1000
prototype_embs = torch.randn(classes, args.emb_dim).cuda()
prototype_embs = prototype_embs / (torch.norm(prototype_embs, dim=-1, keepdim=True) + 1e-3)

dist_func = pair_wise_hyp if args.geometry == "hyp" else pair_wise_eud

# Create dataloaders
train_loader, test_loader = create_loaders(data_dir=args.data_dir ,batch_size=args.batch_size, num_workers=args.num_workers)

# Create model
model = MLP(768, 1024, args.emb_dim).cuda()
if args.opt == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.opt == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

loss_fn = torch.nn.CrossEntropyLoss()

best_avg_metrics = {}

for epoch in range(args.epochs):
    epoch_start = time.time()

    model.train()

    for idx, (input, target) in enumerate(train_loader):
        input, target = input.cuda(), target.cuda()
        output = model(input) 

        logits = -dist_func(output, prototype_embs ,c = args.c) / args.T
        loss = F.cross_entropy(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    metrics = {"losses": utils.AverageMeter(), "top1": utils.AverageMeter()}

    model.eval()

    with torch.no_grad():
        for input, target in test_loader:
            input, target = input.cuda(), target.cuda()
            output = model(input)

            logits = -dist_func(output, prototype_embs ,c = args.c) / args.T
            loss = F.cross_entropy(logits, target)

            acc1, acc5 = utils.accuracy(logits, target, topk=(1, 5))

            metrics["losses"].update(loss.data.item(), input.size(0))
            metrics["top1"].update(acc1.item(), logits.size(0))

    if (not best_avg_metrics or metrics[args.criterion].avg > best_avg_metrics[args.criterion]):
        best_avg_metrics = {k: metrics[k].avg for k in metrics}
        best_model_state = model.state_dict()
        
    print(f"Epoch {epoch}:  Time: {time.time() - epoch_start:.3f}  Loss: {metrics['losses'].avg:>7.4f}  Acc@1: {metrics['top1'].avg:>7.4f}")

if args.save:
    torch.save(best_model_state, os.path.join(exp_dir, "model.pt"))
    torch.save(prototype_embs, os.path.join(exp_dir, "prototype_embs.pt")
    with open(os.path.join(exp_dir, "args.json"), "w") as f:
        json.dump(vars(args), f)