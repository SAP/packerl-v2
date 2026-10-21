import torch

if __name__ == '__main__':

    # “Source” observers and their values (unsorted, may have extras)
    observers = torch.tensor([10, 42, 17, 8, 99], dtype=torch.long)
    print("observers:         ", observers)

    values    = torch.tensor([1.5, 3.2, 4.8, 2.1, 0.7], dtype=torch.float)
    print("values:            ", values)

    # The subset of observer‑IDs we care about (unsorted)
    relevant_observers = torch.tensor([99, 17, 42, 123], dtype=torch.long)
    print("relevant_observers:", relevant_observers)

    # Build an (N × R) boolean mask
    # mask[i,j] == True iff observers[i] == relevant_observers[j]
    mask = observers.unsqueeze(1) == relevant_observers.unsqueeze(0)
    print("mask:             ", mask)

    target_values = mask.to(values.dtype).T @ values
    print("target_values:     ", target_values)
