import torch

def get_mask_from_lengths(lengths, max_len=None):
    if not max_len:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device, dtype=torch.long)
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def get_mask(lengths, max_len=None):
    if not max_len:
        max_len = torch.max(lengths).item()
    lens = torch.arange(max_len)
    mask = lens[:max_len].unsqueeze(0) < lengths.unsqueeze(1)
    return mask



if __name__ == "__main__":
    lens = torch.tensor([2, 3, 7, 5, 4])
    print(get_mask_from_lengths(lens))
    print(get_mask(lens))
    print(get_mask(lens).shape)
    mask  = get_mask(lens)
    print(mask.unsqueeze(1).shape)
    print(mask.shape)
