import time

import torch
from torch_scatter import scatter_max
from torch.distributions.categorical import Categorical


def loop_sample_with_index(probs, indices):
    """
    Loop-based sampling per group.

    For each group (unique value in indices), it masks out the corresponding probabilities,
    samples an index using torch.distributions.Categorical, and returns the sampled probability
    and the original index.
    """
    num_groups = int(indices.max().item()) + 1
    sampled_probs = torch.zeros(num_groups, device=probs.device)
    sampled_indices = torch.zeros(num_groups, dtype=torch.long, device=probs.device)

    for i in range(num_groups):
        mask = indices == i
        cur_probs = probs[mask]
        if cur_probs.nelement() == 0:
            raise ValueError("This should not happen: group {} is empty.".format(i))
        s = Categorical(probs=cur_probs).sample()
        sampled_probs[i] = cur_probs[s]
        cur_position_candidates = torch.nonzero(mask, as_tuple=False).view(-1)
        sampled_indices[i] = cur_position_candidates[s]
    return sampled_probs, sampled_indices


def scatter_sample_categorical(probs, indices):
    """
    Vectorized categorical sampling per group using the Gumbel–Max trick.

    Args:
        probs (torch.Tensor): 1-D tensor of probability values for items.
                                These need not be normalized and should be > 0.
        indices (torch.Tensor): 1-D tensor of integer group ids (same shape as probs).
                                Groups are assumed to be in {0, ..., num_groups-1}.

    Returns:
        sampled_probs (torch.Tensor): 1-D tensor (length = num_groups) containing the
                                      sampled probability value (from the original input)
                                      for each group.
        sampled_indices (torch.Tensor): 1-D tensor (length = num_groups) containing the
                                        indices (with respect to the input tensor) corresponding
                                        to the sampled probabilities.
    """
    # Compute the number of groups.
    num_groups = int(indices.max().item()) + 1

    # The Gumbel–Max trick:
    #   sample u ~ Uniform(0, 1) for each element,
    #   compute g = -log(-log(u)) (Gumbel noise),
    #   then compute score = log(prob) + g.
    # The argmax of these scores per group gives the desired sample.
    probs_double = probs.to(torch.double)
    u = torch.rand_like(probs_double)
    gumbel_noise = -torch.log(-torch.log(u))
    scores = torch.log(probs_double) + gumbel_noise

    # For each group, compute the index of the maximum score.
    # scatter_max returns (max_values, argmax_indices) per group.
    _, group_argmax = scatter_max(scores, indices, dim=0, dim_size=num_groups)

    # These indices refer to positions in the original 'probs' tensor.
    sampled_indices = group_argmax
    sampled_probs = probs[sampled_indices]

    return sampled_probs, sampled_indices


def exp_verify_sample(sample_func, num_samples=100000):
    # Define human-readable probabilities and group assignments.
    # - Group 0: two options with probabilities 0.2 and 0.8
    # - Group 1: three options with probabilities 0.3, 0.3, and 0.4
    # - Group 2: two options with probabilities 0.5 and 0.5
    probs = torch.tensor([0.8, 0.3, 0.4, 0.2, 0.5, 0.3, 0.5])
    indices = torch.tensor([0, 1, 1, 0, 2, 1, 2])
    num_options = probs.size(0)

    group_0_indices = torch.nonzero(indices == 0, as_tuple=True)[0]
    group_0_probs = probs[group_0_indices]
    group_1_indices = torch.nonzero(indices == 1, as_tuple=True)[0]
    group_1_probs = probs[group_1_indices]
    group_2_indices = torch.nonzero(indices == 2, as_tuple=True)[0]
    group_2_probs = probs[group_2_indices]

    # Create a count tensor for each original option.
    counts = torch.zeros(num_options, dtype=torch.long)

    # Run the sampling procedure num_samples times.
    for _ in range(num_samples):
        # The sampler returns one sample per group (3 samples per call).
        _, sampled_indices = sample_func(probs, indices)
        # Count the selections for each option.
        for idx in sampled_indices.tolist():
            counts[idx] += 1

    # Print the detailed results.
    print(f"Results with seed {seed}:")

    # Group 0
    group0_total = torch.sum(counts[group_0_indices])
    print(f" Group 0 (expected: {group_0_probs}):")
    for i, (idx, prob) in enumerate(zip(group_0_indices, group_0_probs)):
        print(f"  - Option at original index {idx.item()} (prob = {prob.item():.2f}): "
              f"{counts[idx].item()} selections ({counts[idx].item() / group0_total:.3f})")

    # Group 1
    group1_total = torch.sum(counts[group_1_indices])
    print(f" Group 1 (expected: {group_1_probs}):")
    for i, (idx, prob) in enumerate(zip(group_1_indices, group_1_probs)):
        print(f"  - Option at original index {idx.item()} (prob = {prob.item():.2f}): "
              f"{counts[idx].item()} selections ({counts[idx].item() / group1_total:.3f})")

    # Group 2
    group2_total = torch.sum(counts[group_2_indices])
    print(f" Group 2 (expected: {group_2_probs}):")
    for i, (idx, prob) in enumerate(zip(group_2_indices, group_2_probs)):
        print(f"  - Option at original index {idx.item()} (prob = {prob.item():.2f}): "
              f"{counts[idx].item()} selections ({counts[idx].item() / group2_total:.3f})")

    print("-" * 50, "\n")


def _exp_timing(probs, indices, sample_fn, n_iter):
    start_time = time.time()
    for _ in range(n_iter):
        sample_fn(probs, indices)
    return time.time() - start_time


def exp_timing(total_entries=64, n_iter=1000):

    if total_entries < 16:
        raise ValueError("total_entries must be at least 16.")
    if total_entries % 16 != 0:
        raise ValueError("total_entries must be a multiple of 16.")

    all_num_edge_dest_idx = [1, 2, 4, 8]
    s_ = 16
    while total_entries // s_ > 1:
        all_num_edge_dest_idx.append(s_)
        s_ *= 16
    if total_entries > 16:
        all_num_edge_dest_idx.append(total_entries)

    print("\n", "-" * 50, "\n")
    for num_groups in all_num_edge_dest_idx:

        num_entries_per_group = total_entries // num_groups
        indices = torch.arange(num_groups).repeat(num_entries_per_group)
        probs = torch.rand(total_entries)

        print(f"Total entries: {total_entries}, "
              f"Num groups: {num_groups}, "
              f"Num entries per group: {num_entries_per_group}, "
              f"Iterations: {n_iter}:")

        scatter_time = _exp_timing(probs, indices, scatter_sample_categorical, n_iter)


        print("  Scatter Gumbel-based approach runtime: {:.3f} seconds".format(scatter_time))
    print("\n", "-" * 50, "\n")


if __name__ == '__main__':
    #seeds = [0, 42, 2025]
    seeds = [0]
    for seed in seeds:
        torch.manual_seed(seed)
        #exp_verify_sample(scatter_sample_categorical)
        #exp_timing(total_entries=64, n_iter=200)
        exp_timing(total_entries=65536, n_iter=100)
        exp_timing(total_entries=16777216, n_iter=10)
