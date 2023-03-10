import torch

if __name__ == '__main__':
    pred = torch.rand([2, 16, 4])
    gt = torch.tensor([1, 2])
    for i in range(2):
        gt_ = pred[i, :, gt[i]]
        other = torch.cat(pred[i, :, :gt[i]], pred[i, :, gt[i]+1:]) if gt[i] != 3 else pred[i, :, :3]
        max_index = torch.max(other)

