import copy

import torch
import torch.nn as nn


def tree_level_elements(tree, level, return_subtrees=False):
    return _tree_level_elements(tree, 0, level, return_subtrees=return_subtrees)


def _tree_level_elements(tree, current_level, desired_level, return_subtrees=False):

    if current_level == desired_level:

        if isinstance(tree, dict):
            if return_subtrees:
                subtrees = []
                for key, sub_tree in tree.items():
                    subtrees.append((key, _tree_level_elements(sub_tree, current_level + 1, desired_level + 1, return_subtrees=False)))
                return subtrees
            else:
                return list(tree.keys())
        elif isinstance(tree, (list, tuple)):
            return list(tree)
        else:
            return tree
    else:

        subtree_keys = []

        for _, sub_tree in tree.items():
            subtree_keys.extend(_tree_level_elements(sub_tree, current_level + 1, desired_level, return_subtrees=return_subtrees))

        return subtree_keys


def get_nesting_depth(d):
    if not isinstance(d, dict) or not d:
        return 0
    return 1 + get_nesting_depth(next(iter(d.values())))


def get_explanation_level_mapping(higher_level, lower_level, label_hierarchy):

    assert higher_level <= lower_level
    assert len(label_hierarchy) >= lower_level - 1

    if higher_level == lower_level:
        return {i: [i] for i in range(len(label_hierarchy[higher_level]))}

    label_mapping = copy.deepcopy(label_hierarchy[higher_level])

    while higher_level != lower_level - 1:
        higher_level += 1

        one_level_remapping = copy.deepcopy(label_hierarchy[higher_level])

        for higher_class, mapped_labels in label_mapping.items():

            new_targets = []

            for mapped_label in mapped_labels:

                new_mapped_labels = one_level_remapping[mapped_label]
                new_targets.extend(new_mapped_labels)

            label_mapping[higher_class] = new_targets

    return label_mapping


def parse_label_hierarchy(label_hierarchy):

    hierarchy_depth = get_nesting_depth(label_hierarchy) + 1

    explanations_per_level = [tree_level_elements(label_hierarchy, l, return_subtrees=False) for l in range(hierarchy_depth)]
    explanations_per_level_remapping = [tree_level_elements(label_hierarchy, l, return_subtrees=True) for l in range(hierarchy_depth - 1)]
    explanations_per_level_number_mapping = [{exp: i for i, exp in enumerate(explanations, 1)} for explanations in explanations_per_level]

    number_remapping_levels = []

    for level, remap_dict in enumerate(explanations_per_level_remapping):

        # Background is always mapped to background
        remapping = {0: [0]}

        for general_exp, fine_grained_exps in remap_dict:

            number_general_exp = explanations_per_level_number_mapping[level][general_exp]
            numbers_fine_grained_exps = [explanations_per_level_number_mapping[level + 1][exp] for exp in fine_grained_exps]

            remapping[number_general_exp] = numbers_fine_grained_exps

        number_remapping_levels.append(remapping)

    return explanations_per_level, explanations_per_level_number_mapping, explanations_per_level_remapping, number_remapping_levels


def remap_label_levels(mask, number_remapping):

    aggregated_slices = [torch.sum(mask[:, number_remapping[i], ...], dim=1) for i in range(len(number_remapping))]
    return torch.stack(aggregated_slices, dim=1)


def generate_label_hierarchy(mask, number_remappings, start_level=2):

    masks = [mask]

    for level in range(start_level - 1, -1, -1):
        number_remapping = number_remappings[level]
        # for number_remapping in number_remappings[-start_level::-1]:
        masks.append(remap_label_levels(masks[-1], number_remapping))

    return masks[::-1]


class TreeLoss(nn.Module):

    __name__ = "TreeLoss"

    def __init__(self, level_weights, loss_functions_per_level, numeric_label_remappings=None, **kwargs):
        super().__init__()

        self.remappings = numeric_label_remappings
        self.weights = level_weights
        self.loss_functions_per_level = loss_functions_per_level
        self.num_classes_per_level = self._get_num_classes_per_level(numeric_label_remappings)

    def _get_num_classes_per_level(self, mapping):
        if mapping is None:
            return None
        num_classes_per_level = [(len(r.keys()), len(set(sum(list(r.values()), [])))) for r in mapping]
        num_classes_per_level = [ncpl[0] for ncpl in num_classes_per_level] + [num_classes_per_level[-1][1]]
        return num_classes_per_level

    def _get_level_of_input(self, input):

        for lvl, num_classes in enumerate(self.num_classes_per_level):
            if num_classes == input.shape[1]:
                return lvl

        raise RuntimeError(f"No matching level found with {input.shape[1]} number of classes.")

    def forward(self, input: torch.Tensor, target: torch.Tensor):

        lvl_of_input = self._get_level_of_input(input)

        input_lvls = generate_label_hierarchy(input, self.remappings, lvl_of_input)
        target_lvls = generate_label_hierarchy(target, self.remappings, lvl_of_input)

        losses = []
        total_loss = 0.0

        for input_lvl, target_lvl, loss_functions_lvl, lvl_weight in zip(input_lvls, target_lvls, self.loss_functions_per_level, self.weights):
            lvl_losses = []
            lvl_total_loss = 0.0

            for weigth, loss_function in loss_functions_lvl:
                l = loss_function(input_lvl, target_lvl)
                lvl_total_loss += weigth * l
                lvl_losses.append(l)

            losses.append(lvl_losses)
            total_loss += lvl_weight * lvl_total_loss

        return total_loss

    def init_runtime(self, numeric_label_remappings, **kwargs):
        self.remappings = numeric_label_remappings
        self.num_classes_per_level = self._get_num_classes_per_level(numeric_label_remappings)
