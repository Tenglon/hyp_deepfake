import argparse
import json
import os
import time
from datetime import datetime

import torch
from timm import utils

from ffppc23.dataloader import Ffplusplusc23DatasetFactory
from models.optimizers import initialize_optimizer
from models.resnets import parse_model_from_name

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, default="euclidean-8-16-768-resnet-32")
parser.add_argument("--dataset", type=str, default="ffppc23")
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--opt", type=str, default="sgd")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--save", action="store_const", const=True, default=True)
parser.add_argument("--criterion", type=str, default="top1", choices=["losses", "top1", "top5"])
parser.add_argument("--num-workers", type=int, default=8)


def main():
    args = parser.parse_args()

    # Create some strings for file management
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_path = os.path.dirname(os.path.realpath(__file__))
    exp_dir = os.path.join(dir_path, "runs", args.dataset, args.model, now)
    os.makedirs(exp_dir)

    dataset_factory = Ffplusplusc23DatasetFactory
    classes = 1000

    # Create dataloaders
    train_loader, test_loader = dataset_factory.create_train_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Create model
    model = parse_model_from_name(args.model, classes).cuda()

    # Create optimizers
    optimizer = initialize_optimizer(
        model=model,
        args=args,
    )

    print(f"Using optimizer: {optimizer}")

    loss_fn = torch.nn.CrossEntropyLoss()

    best_avg_metrics = {}

    for epoch in range(args.epochs):
        epoch_start = time.time()

        model.train()

        for idx, (input, target) in enumerate(train_loader):
            input, target = input.cuda(), target.cuda()
            output = model(input) #torch.Size([128, 998])
            # import pdb;pdb.set_trace()
            loss = loss_fn(output, target)#torch.Size([128])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

                loss = loss_fn(output, target)
                acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

                metrics["losses"].update(loss.data.item(), input.size(0))
                metrics["top1"].update(acc1.item(), output.size(0))
                metrics["top5"].update(acc5.item(), output.size(0))

        if (
            not best_avg_metrics
            or metrics[args.criterion].avg > best_avg_metrics[args.criterion]
        ):
            best_avg_metrics = {k: metrics[k].avg for k in metrics}
            best_model_state = model.state_dict()

        print(
            f"Epoch {epoch}:  "
            f"Time: {time.time() - epoch_start:.3f}  "
            f"Loss: {metrics['losses'].avg:>7.4f}  "
            f"Acc@1: {metrics['top1'].avg:>7.4f}  "
            f"Acc@5: {metrics['top5'].avg:>7.4f}"
        )

    output_dict = {
        "best_model_state": best_model_state,
        "best_avg_metrics": best_avg_metrics,
        "last_model_state": model.state_dict(),
        "last_avg_metrics": {k: metrics[k].avg for k in metrics},
    }

    # Store model weights
    if args.save:
        torch.save(
            output_dict["last_model_state"],
            os.path.join(exp_dir, f"{args.model}_weights.pth"),
        )

        weights_dir = os.path.join(dir_path, "weights", args.dataset, args.model)
        os.makedirs(weights_dir)
        torch.save(
            output_dict["last_model_state"],
            os.path.join(weights_dir, f"{args.model}_weights.pth"),
        )

    # Store metrics
    with open(f"{exp_dir}/metrics.json", "w") as file:
        json.dump(output_dict["last_avg_metrics"], file, indent=4)


if __name__ == "__main__":
    main()
