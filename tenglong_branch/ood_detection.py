import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from lt_loader import create_loaders, Fakedataset
from pmath import pair_wise_hyp, pair_wise_eud

from ood_utils.display_results import (
    get_measures,
    print_measures,
    print_measures_with_std,
    show_performance,
)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="ffppc23", help="Choose dataset.")
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--num_to_avg", type=int, default=1, help="Average measures across num_to_avg runs.")
parser.add_argument("--geometry", type=str, default="hyp")
parser.add_argument("--emb_dim", type=int, default=32)
parser.add_argument("--c", type=float, default=1.0)
parser.add_argument("--T", type=float, default=1.0, help="Temperature for softmax.")
parser.add_argument("--num_workers", type=int, default=2, help="Pre-fetching threads.")

real_dir = '/ssd2/data/ffppc23/real/split_20231215/'
fake_dirs = {'face2face': '/ssd2/data/ffppc23/face2face/all', 
             'deepfakes': '/ssd2/data/ffppc23/deepfakes/all', 
             'faceswap': '/ssd2/data/ffppc23/faceswap/all', 
             'neuraltextures': '/ssd2/data/ffppc23/neuraltextures/all'}

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

def get_and_print_results(ood_loader, model, prototype_embs, num_to_avg=args.num_to_avg):
    aurocs, auprs, fprs = [], [], []

    for _ in range(num_to_avg):
        out_score = get_ood_scores(ood_loader, model, prototype_embs)
        measures = get_measures(-in_score, -out_score)
        aurocs.append(measures[0])
        auprs.append(measures[1])
        fprs.append(measures[2])

    # print(in_score[:3], out_score[:3])
    auroc = np.mean(aurocs)
    aupr = np.mean(auprs)
    fpr = np.mean(fprs)

    if num_to_avg >= 5:
        print_measures_with_std(aurocs, auprs, fprs)
    else:
        print_measures(auroc, aupr, fpr)

    return auroc, aupr, fpr


def get_ood_scores(loader, model, prototype_embs, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // args.batch_size and in_dist is False:
                break

            data = data.cuda()

            output = model(data)
            if args.geometry == "hyp":
                output = -pair_wise_hyp(output, prototype_embs, c=args.c) / args.T
            else:
                output = -pair_wise_eud(output, prototype_embs) / args.T
                
            smax = to_np(F.softmax(output, dim=1))

            _score.append(-to_np((args.T * torch.logsumexp(output / args.T, dim=1))))

            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)

                _right_score.append(-np.max(smax[right_indices], axis=1))
                _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return (
            concat(_score).copy(),
            concat(_right_score).copy(),
            concat(_wrong_score).copy(),
        )
    else:
        return concat(_score)[:ood_num_examples].copy()


# Step0 : Load the model
model = MLP(768, 1024, args.emb_dim).cuda()

runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.geometry, args.dataset)
last_run = sorted(os.listdir(runs_dir))[-1]
last_run = '2023-12-30_07-02-02' # TODO: remove this line

weights_path = os.path.join(runs_dir, last_run, "model.pt")
emb_path = os.path.join(runs_dir, last_run, "prototype_embs.pt")

prototype_embs = torch.load(emb_path)

state_dict = torch.load(weights_path)
model.load_state_dict(state_dict)
model.eval()

# Step 1: Get in-distribution scores on test set
_, test_loader = create_loaders(data_dir=real_dir, batch_size=args.batch_size, num_workers=args.num_workers)
ood_num_examples = len(test_loader) * args.batch_size // 5

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()

auroc_list, aupr_list, fpr_list = [], [], []

in_score, right_score, wrong_score = get_ood_scores(test_loader, model, prototype_embs, in_dist=True)

num_right = len(right_score)
num_wrong = len(wrong_score)
print("Error Rate {:.2f}".format(100 * num_wrong / (num_wrong + num_right)))

show_performance(wrong_score, right_score)


# Step 2: Get out-of-distribution scores on all OOD datasets
for key, value in fake_dirs.items():

    ood_data = Fakedataset(root_dir=value)
    ood_loader = torch.utils.data.DataLoader(
        ood_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    print("{} Detection".format(key))
    auroc, aupr, fpr = get_and_print_results(ood_loader, model, prototype_embs)

    out_score = get_ood_scores(ood_loader, model, prototype_embs)
    score_dict = {'in_score': -in_score, 'out_score': -out_score}
    torch.save(score_dict, os.path.join(runs_dir, last_run, '{}_score.pt'.format(key)))

    auroc_list.append(auroc), aupr_list.append(aupr), fpr_list.append(fpr)


# Mean Results
print("\nMean Test Results!!!!!")
print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list))

