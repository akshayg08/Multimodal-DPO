import torch 
from torch.nn import functional as F

def get_logprobs(labels, logits):
    
    logprobs = logits.log_softmax(-1)
    loss_mask = labels > 0
    labels[loss_mask == False] = 0

    sampled_logprobs = torch.gather(logprobs, dim=2, index=labels.unsqueeze(-1))
    sampled_logprobs = sampled_logprobs.squeeze(-1)
    sampled_logprobs[loss_mask == False] = 0

    final_lp_sum = sampled_logprobs.sum(axis=-1)
    final_lp_mean = sampled_logprobs.mean(axis=-1)

    return final_lp_sum, final_lp_mean

def dpo_loss(policy_accept_lps, policy_reject_lps, ref_accept_lps, ref_reject_lps, beta=0.5, label_smoothing=0.2):

    pi_logratios = policy_accept_lps - policy_reject_lps
    ref_logratios = ref_accept_lps - ref_reject_lps

    logits = pi_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
    loss = losses.mean()

    chosen_rewards = beta * (policy_accept_lps - ref_accept_lps).detach()
    rejected_rewards = beta * (policy_reject_lps - ref_reject_lps).detach()

    return loss, chosen_rewards, rejected_rewards
