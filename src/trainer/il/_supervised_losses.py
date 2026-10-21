import torch
from torch import Tensor
from torch.nn import functional as F


def dist_matrix_loss(student_dist_matrix: Tensor, expert_dist_matrix: Tensor, eps=1e-8) -> Tensor:
    """
    Global-geometry loss between pairwise distance matrices via centered Gram matrices.

    Returns:
        Scalar tensor: geometry alignment loss (0 means perfect up-to-scale match).
    """
    Ds = student_dist_matrix
    Dt = expert_dist_matrix

    # Basic checks
    assert Ds.ndim == 2 and Dt.ndim == 2 and Ds.shape == Dt.shape and Ds.shape[0] == Ds.shape[1], \
        "Distance matrices must be square and same shape"

    # Symmetrize, zero diag, clamp
    Ds = 0.5 * (Ds + Ds.T)
    Dt = 0.5 * (Dt + Dt.T)
    Ds = Ds - torch.diag_embed(torch.diagonal(Ds))
    Dt = Dt - torch.diag_embed(torch.diagonal(Dt))
    Ds = Ds.clamp_min(0.0)
    Dt = Dt.clamp_min(0.0)

    Ds2, Dt2 = Ds.square(), Dt.square()

    # Double-centering: B = -0.5 * (D2 - row_mean - col_mean + grand_mean)
    rs, cs, ms = Ds2.mean(1, True), Ds2.mean(0, True), Ds2.mean()
    rt, ct, mt = Dt2.mean(1, True), Dt2.mean(0, True), Dt2.mean()
    Bs = -0.5 * (Ds2 - rs - cs + ms)
    Bt = -0.5 * (Dt2 - rt - ct + mt)

    # Cosine distance (explicit form to control eps)
    num = (Bs * Bt).sum()
    den = Bs.norm().clamp_min(eps) * Bt.norm().clamp_min(eps)
    return 1.0 - num / den


def link_weight_mse_loss(student_link_weights: Tensor, expert_link_weights: Tensor) -> Tensor:
    """
    Mean-squared error loss between student and expert link weights. Simple variant.
    """
    if student_link_weights.shape != expert_link_weights.shape:
        raise ValueError("Link weight tensors must have the same shape")
    c = expert_link_weights.mean() / student_link_weights.mean()
    s = student_link_weights
    e = expert_link_weights / c
    return F.mse_loss(s, e)


def link_weight_mse_loss_dual_normalized(student_link_weights: Tensor, expert_link_weights: Tensor) -> Tensor:
    if student_link_weights.shape != expert_link_weights.shape:
        raise ValueError("Link weight tensors must have the same shape")
    s = student_link_weights / (student_link_weights.mean(dim=-1, keepdim=True) + 1e-8)
    e = student_link_weights / (expert_link_weights.mean(dim=-1, keepdim=True) + 1e-8)
    return F.mse_loss(s, e)


def link_weight_mse_loss_dual_normalized_v2(student_link_weights: Tensor, expert_link_weights: Tensor) -> Tensor:
    if student_link_weights.shape != expert_link_weights.shape:
        raise ValueError("Link weight tensors must have the same shape")
    s = student_link_weights / (student_link_weights.mean(dim=-1, keepdim=True) + 1e-8)
    e = expert_link_weights / (expert_link_weights.mean(dim=-1, keepdim=True) + 1e-8)
    return F.mse_loss(s, e)


def sedi_sparse_ce_loss(student_sedi_soft_sparse: Tensor,
                        expert_sedi_onehot_sparse: Tensor,
                        edge2src: Tensor) -> Tensor:
    """
    Cross-entropy loss between FL's prediction edge-dest values and the sparse target sedi.

    pred_flat:   [E*N]   float logits, flattened so index = edge_idx * N + dest_idx
    target_flat: [E*N]   binary mask (0/1), flattened the same way
    input_obs:   Observation, must contain edge_index [2, E] tensor

    Returns:
      scalar: average cross-entropy over all (source s, dest j) pairs
    """
    N = student_sedi_soft_sparse.numel() // edge2src.numel()  # infer number of destinations

    total_loss = torch.tensor(0.0, device=student_sedi_soft_sparse.device)
    total_count = 0

    # Precompute a [N] tensor of destination offsets 0,1,…,N–1
    dest_range = torch.arange(N, device=student_sedi_soft_sparse.device)

    # Now only loop over each source node once
    for s in torch.unique(edge2src):
        # which edges come from s?
        edges_s = (edge2src == s).nonzero(as_tuple=True)[0]  # shape [deg_s]
        deg_s = edges_s.size(0)
        if deg_s == 0:
            continue

        # Build a [deg_s × N] index matrix into the flat tensors:
        #   row r, col j  →  flat_idx = edges_s[r] * N + j
        idx = edges_s.unsqueeze(1) * N + dest_range.unsqueeze(0)  # → [deg_s, N]

        # Gather both logits and binary targets for all dests at once
        logits_mat = student_sedi_soft_sparse[idx]  # [deg_s, N]
        targ_mat = expert_sedi_onehot_sparse[idx]  # [deg_s, N]

        # Find which destinations actually have exactly one positive among these edges
        valid_dests = (targ_mat.sum(dim=0) == 1)  # [N] boolean mask
        if not valid_dests.any():
            continue

        # Restrict to just those columns (dests)
        logits_sel = logits_mat[:, valid_dests]  # [deg_s, M]
        targ_sel = targ_mat[:, valid_dests]  # [deg_s, M]

        # Within each of these M dests, find the local index (0…deg_s–1) of the positive
        pos_local = targ_sel.argmax(dim=0)  # [M]

        # Prepare a mini-batch of size M with C=deg_s classes
        # – inputs: [M, deg_s]
        # – targets: [M]
        batch_logits = logits_sel.t()  # [M, deg_s]
        batch_labels = pos_local  # [M]

        # Sum the per-sample losses, we’ll divide by total_count at the end
        total_loss += F.cross_entropy(batch_logits, batch_labels, reduction='sum')
        total_count += batch_logits.size(0)

    if total_count == 0:
        # no valid (s,j) pairs at all → zero loss
        return torch.tensor(0.0, device=student_sedi_soft_sparse.device)

    return total_loss / total_count
