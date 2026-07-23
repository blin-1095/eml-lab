import torch
import copy

from typing import Dict

def l1_structured_pruning(state_dict: Dict, prune_ratio: float) -> Dict:

    state_dict = copy.deepcopy(state_dict)

    # only 1-8 because we dont want to pruned the ninth layer
    for i in range(1,9):

        channel_norms = torch.sum(torch.abs(state_dict[f"conv{i}.weight"]), dim=(1,2,3))
        threshold = torch.quantile(channel_norms, prune_ratio)
        mask = (channel_norms >= threshold).float()
        state_dict[f"conv{i}.weight"] = state_dict[f"conv{i}.weight"] * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # prune batchnorm layers as well
        state_dict[f"bn{i}.weight"] = state_dict[f"bn{i}.weight"] * mask
        state_dict[f"bn{i}.bias"] = state_dict[f"bn{i}.bias"] * mask
        state_dict[f"bn{i}.running_mean"] = state_dict[f"bn{i}.running_mean"] * mask
        state_dict[f"bn{i}.running_var"] = state_dict[f"bn{i}.running_var"] * mask

    return state_dict


def densify_state_dict(state_dict: Dict) -> Dict:

    state_dict = copy.deepcopy(state_dict)

    for i in range(1,10):

        filter_norm = torch.sum(torch.abs(state_dict[f"conv{i}.weight"]), dim=(1,2,3))

        mask = (filter_norm > 0)
        state_dict[f"conv{i}.weight"] = state_dict[f"conv{i}.weight"][mask, :, :, :]

        if i != 9:
            state_dict[f"conv{i + 1}.weight"] = state_dict[f"conv{i + 1}.weight"][:, mask, :, :]

        bn_key = f"bn{i}"
        if f"{bn_key}.weight" in state_dict:
            for param in ['weight', 'bias', 'running_mean', 'running_var']:
                full_key = f"{bn_key}.{param}"
                state_dict[full_key] = state_dict[full_key][mask]

    return state_dict