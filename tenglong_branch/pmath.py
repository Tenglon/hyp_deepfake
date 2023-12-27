import torch


class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        res = (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)
    
def artanh(x):
    return Artanh.apply(x)

def _mobius_addition_batch(x, y, c):
    
    xy = torch.einsum('ij,kj->ik', (x, y))  # B x C
    x2 = x.pow(2).sum(-1, keepdim=True)  # B x 1
    y2 = y.pow(2).sum(-1, keepdim=True)  # C x 1
    num = (1 + 2 * c * xy + c * y2.permute(1, 0))  # B x C
    num = num.unsqueeze(2) * x.unsqueeze(1) # B x C x 1 * B x 1 x D = B x C x D
    num = num + (1 - c * x2).unsqueeze(2) * y  # B x C x D + B x 1 x 1 = B x C x D
    denom_part1 = 1 + 2 * c * xy  # B x C
    denom_part2 = c ** 2 * x2 * y2.permute(1, 0) # B x 1 * 1 x C = B x C
    denom = denom_part1 + denom_part2
    res = num / (denom.unsqueeze(2) + 1e-5)
    return res

def pair_wise_hyp(x, y, c=1.0):
    sqrt_c = c ** 0.5
    return 2 / sqrt_c * artanh(sqrt_c * torch.norm(_mobius_addition_batch(-x, y, c=c), dim=-1))

def pair_wise_eud(x, y, c=1.0):
    # input: 
    # x, m x d
    # y, n x d
    # output: m x n
    m = x.size(0)
    n = y.size(0)
    d = x.size(1)
    assert(x.size(1) == y.size(1))
    xx = x.pow(2).sum(-1, keepdim = True)
    yy = y.pow(2).sum(-1, keepdim = True)
    xy = torch.einsum('ij,kj->ik', (x, y))

    result = xx - 2*xy + yy.permute(1, 0)
    return result