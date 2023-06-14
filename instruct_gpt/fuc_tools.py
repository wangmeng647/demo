import torch
import torch.nn.functional as F


def generate_for_reward_training(gpt_generator, prompts):
    prompts_ids = []
    ans_ids = []
    for prom in prompts:
        query, ans = gpt_generator.generate(prom, only_id=True)
        prompts_ids.append(query)
        ans_ids.append(ans)
    return prompts_ids, ans_ids


def top_k_top_p_generate(logits, top_k=0, top_p=0.):

    if top_k == 0 and top_p == 0:
        raise Exception('input top_k or top_p for one pattern')
    if top_k != 0 and top_p != 0:
        raise Exception('can only choose one of the generate model')

    if top_k != 0:
        top_k = min(top_k, logits.size(-1))
        idx_mask = logits < torch.topk(logits, top_k)[0][-1]
        logits[idx_mask] = -float('inf')
        next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
    else:
        sorted_value, sorted_idx = torch.sort(logits, descending=True)
        cum_p = torch.cumsum(torch.softmax(sorted_value, dim=-1), dim=-1)
        idx_mask = cum_p > top_p
        idx_mask[idx_mask.nonzero()[0]] = False
        idx_to_remove = sorted_idx[idx_mask]
        logits[idx_to_remove] = -float('inf')
        next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)

    return next_token


def top_k_hierarchy_generate(logits, top_k=5, hierarchy=2):

    top_k = min(top_k, logits.size(-1))
    right_top = top_k * hierarchy
    left_top = top_k * (hierarchy - 1)
    right_mask = logits < torch.topk(logits, right_top)[0][-1]
    if left_top == 0:
        left_mask = logits < float('inf')
    else:
        left_mask = logits < torch.topk(logits, left_top)[0][-1]
    mask = right_mask + ~left_mask
    logits[mask] = -float('inf')
    next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)

    return next_token


def warmup(warmup_steps, total_steps):

    def fuc(step):
        if step < warmup_steps:
            return 1 / warmup_steps * (step + 1)
        else:
            return 1 - 1 / (total_steps - warmup_steps) * (step - warmup_steps + 1)
    return fuc
