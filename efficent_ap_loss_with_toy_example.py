import torch
import time
import numpy as np
import os
# import psutil
import sys
import torch.nn.functional as F
import math


def AP_loss(logits, targets):
    total_memory = 0
    st = time.time()

    delta = 0.0
    grad = torch.zeros(logits.shape)  # .cuda()

    metric = torch.zeros(1)  # .cuda()

    if torch.max(targets) <= 0:
        return grad, metric

    labels_p = (targets == 1)
    fg_logits = logits[labels_p]
    threshold_logit = torch.min(fg_logits) - delta

    ######## Ignore those negative j that satisfy (L_{ij}=0 for all positive i), to accelerate the AP-loss computation.
    valid_labels_n = ((targets == 0) & (logits >= threshold_logit))
    valid_bg_logits = logits[valid_labels_n]
    valid_bg_grad = torch.zeros(len(valid_bg_logits))  # .cuda()
    ########

    fg_num = len(fg_logits)
    total_memory += sys.getsizeof(fg_num)
    prec = torch.zeros(fg_num)  # .cuda()
    order = torch.argsort(fg_logits)
    max_prec = 0

    for ii in order:
        tmp1 = fg_logits - fg_logits[ii]
        if delta > 0:
            tmp1 = torch.clamp(tmp1 / (2 * delta) + 0.5, min=0, max=1)
        else:
            tmp1 = (tmp1 >= 0).float()

        tmp2 = valid_bg_logits - fg_logits[ii]
        a = 0
        if delta > 0:
            tmp2 = torch.clamp(tmp2 / (2 * delta) + 0.5, min=0, max=1)
            a += 0.5
        else:
            tmp2 = (tmp2 >= 0).float()

        a += torch.sum(tmp1)
        b = torch.sum(tmp2)
        tmp2 /= (a + b)
        current_prec = a / (a + b)
        '''
        if (max_prec <= current_prec):
            max_prec = current_prec
        else:
            tmp2 *= ((1 - max_prec) / (1 - current_prec))
        '''
        valid_bg_grad += tmp2  # .cuda()
        prec[ii] = current_prec

    grad[valid_labels_n] = valid_bg_grad
    grad[labels_p] = (1 - prec)

    fg_num = max(fg_num, 1)

    grad /= fg_num

    metric = torch.sum(prec, dim=0, keepdim=True) / fg_num
    et = time.time()

    print("Elapsed Time:", (et - st))

    grad_positive_sum = torch.sum(grad[labels_p], axis=0)
    grad_negative_sum = torch.sum(grad[valid_labels_n], axis=0)
    print("Difference between sum of positive grads and sum of negative grads:",
          torch.sub(grad_positive_sum, grad_negative_sum))
    total_memory += sys.getsizeof(delta)
    total_memory += sys.getsizeof(grad)
    total_memory += grad.nelement() * grad.element_size()
    total_memory += sys.getsizeof(metric)
    total_memory += metric.nelement() * metric.element_size()
    total_memory += sys.getsizeof(labels_p)
    total_memory += sys.getsizeof(fg_logits)
    total_memory += fg_logits.nelement() * fg_logits.element_size()
    total_memory += sys.getsizeof(threshold_logit)
    total_memory += threshold_logit.nelement() * threshold_logit.element_size()
    total_memory += sys.getsizeof(valid_labels_n)
    total_memory += valid_labels_n.nelement() * valid_labels_n.element_size()
    total_memory += sys.getsizeof(valid_bg_logits)
    total_memory += valid_bg_logits.nelement() * valid_bg_logits.element_size()
    total_memory += sys.getsizeof(valid_bg_grad)
    total_memory += valid_bg_grad.nelement() * valid_bg_grad.element_size()
    total_memory += sys.getsizeof(prec)
    total_memory += prec.nelement() * prec.element_size()
    total_memory += sys.getsizeof(order)
    total_memory += order.nelement() * order.element_size()
    total_memory += sys.getsizeof(max_prec)
    tmp1 = fg_logits - fg_logits[0]
    tmp2 = valid_bg_logits - fg_logits[0]
    total_memory += sys.getsizeof(tmp1)
    total_memory += tmp1.nelement() * tmp1.element_size()
    total_memory += sys.getsizeof(tmp2)
    total_memory += tmp2.nelement() * tmp2.element_size()
    a = torch.sum(tmp1)
    b = torch.sum(tmp2)
    current_prec = a / (a + b)
    total_memory += sys.getsizeof(a)
    total_memory += sys.getsizeof(b)
    total_memory += sys.getsizeof(current_prec)
    print("Total Memory: ", total_memory / 1024 ** 2)
    return grad, 1 - metric


def Efficent_AP_loss(logits, targets):
    st = time.time()
    total_memory = 0
    delta = 0.0

    grad = torch.zeros(logits.shape)  # .cuda()
    metric = torch.zeros(1)  # .cuda()

    if torch.max(targets) <= 0:
        return grad, metric

    labels_p = (targets == 1)
    fg_logits = logits[labels_p]
    threshold_logit = torch.min(fg_logits) - delta

    ######## Ignore those negative j that satisfy (L_{ij}=0 for all positive i), to accelerate the AP-loss computation.
    valid_labels_n = ((targets == 0) & (logits >= threshold_logit))
    valid_bg_logits = logits[valid_labels_n]
    ########

    fg_num = len(fg_logits)

    bg_relations = valid_bg_logits[:, None] - fg_logits[None, :]
    fg_relations = fg_logits[:, None] - fg_logits[None, :]

    if delta > 0:
        fg_relations = torch.clamp(fg_relations / (2 * delta) + 0.5, min=0, max=1)
        bg_relations = torch.clamp(bg_relations / (2 * delta) + 0.5, min=0, max=1)
    else:
        fg_relations = (fg_relations >= 0).float()
        bg_relations = (bg_relations >= 0).float()

    rank_pos = torch.sum(fg_relations, axis=0)
    if delta > 0:
        rank_pos += 0.5
    FP_num = torch.sum(bg_relations, axis=0)

    rank = rank_pos + FP_num

    ranking_error = FP_num / rank

    grad[labels_p] = ranking_error
    FP_num[(FP_num == 0)] = 1
    grad[valid_labels_n] = torch.sum((bg_relations * (ranking_error / FP_num)), axis=1)

    grad /= fg_num

    metric = torch.sum(1 - ranking_error, axis=0) / fg_num

    et = time.time()

    print("Elapsed Time:", (et - st))
    grad_positive_sum = torch.sum(grad[labels_p], axis=0)
    grad_negative_sum = torch.sum(grad[valid_labels_n], axis=0)

    total_memory += sys.getsizeof(delta)
    total_memory += sys.getsizeof(grad)
    total_memory += grad.nelement() * grad.element_size()
    total_memory += metric.nelement() * metric.element_size()
    total_memory += sys.getsizeof(metric)
    total_memory += sys.getsizeof(labels_p)
    total_memory += labels_p.nelement() * labels_p.element_size()
    total_memory += sys.getsizeof(fg_logits)
    total_memory += fg_logits.nelement() * fg_logits.element_size()
    total_memory += sys.getsizeof(threshold_logit)
    total_memory += threshold_logit.nelement() * threshold_logit.element_size()
    total_memory += sys.getsizeof(valid_labels_n)
    total_memory += valid_labels_n.nelement() * valid_labels_n.element_size()
    total_memory += sys.getsizeof(valid_bg_logits)
    total_memory += valid_bg_logits.nelement() * valid_bg_logits.element_size()
    total_memory += sys.getsizeof(fg_num)
    size_in_bytes = bg_relations.nelement() * bg_relations.element_size()
    total_memory += sys.getsizeof(bg_relations)
    total_memory += size_in_bytes
    total_memory += sys.getsizeof(fg_relations)
    total_memory += fg_relations.nelement() * fg_relations.element_size()
    print("Total Memory: ", total_memory / 1024 ** 2)
    return grad, 1 - metric




def bucketed_ap_loss(logits, targets):
    st = time.time()
    total_memory = 0
    delta = 0.0

    logits, indices = torch.sort(logits, descending=True)
    targets = targets[indices]
    et = time.time()
    print("Elapsed Time for Sorting:", (et - st))
    fg_logits = logits[targets == 1]
    bg_logits = logits[targets == 0]
    irrelevant_fg_logits = fg_logits[(fg_logits >= bg_logits[0] + delta)]
    irrelevant_fg_index = len(irrelevant_fg_logits)
    irrelevant_bg_indices = torch.nonzero((logits + delta < fg_logits[-1]))
    irrelevant_bg_index = len(logits)
    if len(irrelevant_bg_indices) > 0:
        irrelevant_bg_index = irrelevant_bg_indices[0]

    relevant_logits = logits[irrelevant_fg_index:irrelevant_bg_index]
    relevant_targets = targets[irrelevant_fg_index: irrelevant_bg_index]

    relevant_fg_indices = torch.nonzero(relevant_targets == 1)
    relevant_bg_indices = torch.nonzero(relevant_targets == 0)

    # Bucketing of negative logits #

    bucket_sizes_neg = torch.sub(relevant_fg_indices[1:], relevant_fg_indices[:len(relevant_fg_indices) - 1]) - 1
    bucket_sizes_neg = torch.cat((torch.Tensor([[relevant_fg_indices[0]]]), bucket_sizes_neg))
    bucket_sizes_neg = bucket_sizes_neg[(bucket_sizes_neg != 0)]
    bucket_sizes_neg = bucket_sizes_neg.reshape((len(bucket_sizes_neg), 1))

    l_bucket_sizes_neg = [int(x) for x in torch.Tensor.tolist(bucket_sizes_neg.flatten())]
    relevant_bg_buckets = torch.split(relevant_logits[relevant_bg_indices], tuple(l_bucket_sizes_neg))
    bg_bucket_mean = torch.Tensor([torch.mean(bucket) for bucket in list(relevant_bg_buckets)])

    # Bucketing Positive Logits #

    bucket_sizes_pos = torch.sub(relevant_bg_indices[1:], relevant_bg_indices[:len(relevant_bg_indices) - 1]) - 1
    pos_last_bucket = torch.FloatTensor([[len(relevant_logits) - relevant_bg_indices[-1] - 1]])
    bucket_sizes_pos = torch.cat((bucket_sizes_pos, pos_last_bucket))
    bucket_sizes_pos = bucket_sizes_pos[(bucket_sizes_pos != 0)]
    bucket_sizes_pos = bucket_sizes_pos.reshape((len(bucket_sizes_pos), 1))
    l_bucket_sizes_pos = [int(x) for x in torch.Tensor.tolist(bucket_sizes_pos.flatten())]
    relevant_fg_buckets = torch.split(relevant_logits[relevant_fg_indices], tuple(l_bucket_sizes_pos))
    fg_bucket_mean = torch.Tensor([torch.mean(bucket) for bucket in list(relevant_fg_buckets)])

    # Merging all buckets #

    all_buckets = torch.cat((fg_bucket_mean, bg_bucket_mean))
    all_bucket_sizes = torch.cat((bucket_sizes_pos, bucket_sizes_neg))

    all_buckets, indices = torch.sort(all_buckets, descending=True)
    all_bucket_sizes = all_bucket_sizes[indices]

    new_targets = torch.ones(len(all_buckets))
    new_targets[0::2] = 0

    grad = torch.zeros(targets.shape)  # .cuda()
    metric = torch.zeros(1)  # .cuda()

    if torch.max(new_targets) <= 0:
        return grad, metric

    labels_p = (new_targets == 1)
    allabels_p = torch.nonzero((targets == 1))[irrelevant_fg_index:].flatten()

    labels_n = torch.nonzero((targets == 0)[:irrelevant_bg_index]).flatten()

    fg_logits = all_buckets[labels_p]

    threshold_logit = torch.min(fg_logits) - delta

    # Ignore those negative j that satisfy (L_{ij}=0 for all positive i), to accelerate the AP-loss computation.
    valid_labels_n = ((new_targets == 0) & (all_buckets >= threshold_logit))

    valid_bg_logits = all_buckets[valid_labels_n]

    ########

    bg_relations = valid_bg_logits[:, None] - fg_logits[None, :]

    fg_relations = fg_logits[:, None] - fg_logits[None, :]

    if delta > 0:
        fg_relations = torch.clamp(fg_relations / (2 * delta) + 0.5, min=0, max=1)
        bg_relations = torch.clamp(bg_relations / (2 * delta) + 0.5, min=0, max=1)
    else:
        fg_relations = (fg_relations >= 0).float()
        bg_relations = (bg_relations >= 0).float()

    fg_num = len(torch.nonzero((targets == 1)).flatten())

    multiplication_bg = torch.mul(bg_relations, bucket_sizes_neg)
    multiplication_fg = torch.mul(fg_relations, bucket_sizes_pos)

    FP_num = torch.sum(multiplication_bg, axis=0)
    rank_pos = torch.sum(multiplication_fg, axis=0) + irrelevant_fg_index

    if delta > 0:
        rank_pos += 0.5

    rank = rank_pos + FP_num
    ranking_error = FP_num / rank
    FP_num[(FP_num == 0).nonzero()] = 1

    bucket_grads = torch.sum(torch.mul(multiplication_bg, ranking_error) / FP_num, axis=1) / bucket_sizes_neg.flatten()
    duplication_bg = bucket_grads.repeat_interleave(bucket_sizes_neg.flatten().type(torch.LongTensor))
    duplication_fg = (ranking_error / bucket_sizes_pos.flatten()).repeat_interleave(
        bucket_sizes_pos.flatten().type(torch.LongTensor))

    grad[labels_n] = duplication_bg  # .cuda()
    grad[allabels_p] = duplication_fg

    grad /= fg_num
    ranking_error_of_irrelevant_fg = torch.zeros(irrelevant_fg_index)
    ranking_error = torch.cat((ranking_error_of_irrelevant_fg, ranking_error))
    metric = torch.sum(1 - ranking_error, dim=0, keepdim=True) / fg_num
    et_2 = time.time()

    print("Elapsed Time:", (et_2 - st))
    grad_positive_sum = torch.sum(grad[allabels_p], axis=0)
    grad_negative_sum = torch.sum(grad[labels_n], axis=0)

    # print(all_bucket_sizes.flatten().tolist())

    total_memory += sys.getsizeof(delta)
    total_memory += sys.getsizeof(fg_logits)
    total_memory += fg_logits.nelement() * fg_logits.element_size()
    total_memory += sys.getsizeof(irrelevant_fg_logits)
    total_memory += irrelevant_fg_logits.nelement() * irrelevant_fg_logits.element_size()
    total_memory += sys.getsizeof(irrelevant_fg_index)
    total_memory += sys.getsizeof(irrelevant_bg_index)
    total_memory += sys.getsizeof(relevant_logits)
    total_memory += relevant_logits.nelement() * relevant_logits.element_size()
    total_memory += sys.getsizeof(relevant_targets)
    total_memory += relevant_targets.nelement() * relevant_targets.element_size()
    total_memory += sys.getsizeof(relevant_fg_indices)
    total_memory += relevant_fg_indices.nelement() * relevant_fg_indices.element_size()
    total_memory += sys.getsizeof(relevant_bg_indices)
    total_memory += relevant_bg_indices.nelement() * relevant_bg_indices.element_size()
    total_memory += sys.getsizeof(bucket_sizes_neg)
    total_memory += bucket_sizes_neg.nelement() * bucket_sizes_neg.element_size()
    total_memory += sys.getsizeof(l_bucket_sizes_neg)
    total_memory += sys.getsizeof(relevant_bg_buckets)
    total_memory += sys.getsizeof(bg_bucket_mean)
    total_memory += bg_bucket_mean.nelement() * bg_bucket_mean.element_size()
    total_memory += sys.getsizeof(bucket_sizes_pos)
    total_memory += bucket_sizes_pos.nelement() * bucket_sizes_pos.element_size()
    total_memory += sys.getsizeof(pos_last_bucket)
    total_memory += pos_last_bucket.nelement() * pos_last_bucket.element_size()
    total_memory += sys.getsizeof(l_bucket_sizes_pos)
    total_memory += sys.getsizeof(relevant_fg_buckets)
    total_memory += sys.getsizeof(fg_bucket_mean)
    total_memory += fg_bucket_mean.nelement() * fg_bucket_mean.element_size()
    total_memory += sys.getsizeof(all_buckets)
    total_memory += all_buckets.nelement() * all_buckets.element_size()
    total_memory += sys.getsizeof(all_bucket_sizes)
    total_memory += all_bucket_sizes.nelement() * all_bucket_sizes.element_size()
    total_memory += sys.getsizeof(indices)
    total_memory += indices.nelement() * indices.element_size()
    total_memory += sys.getsizeof(new_targets)
    total_memory += new_targets.nelement() * new_targets.element_size()
    total_memory += sys.getsizeof(grad)
    total_memory += grad.nelement() * grad.element_size()
    total_memory += sys.getsizeof(indices)
    total_memory += metric.nelement() * metric.element_size()
    total_memory += sys.getsizeof(metric)
    total_memory += labels_p.nelement() * labels_p.element_size()
    total_memory += sys.getsizeof(labels_p)
    total_memory += allabels_p.nelement() * allabels_p.element_size()
    total_memory += sys.getsizeof(allabels_p)
    total_memory += labels_n.nelement() * labels_n.element_size()
    total_memory += sys.getsizeof(labels_n)
    total_memory += fg_logits.nelement() * fg_logits.element_size()
    total_memory += sys.getsizeof(fg_logits)
    total_memory += sys.getsizeof(threshold_logit)
    total_memory += valid_labels_n.nelement() * valid_labels_n.element_size()
    total_memory += sys.getsizeof(valid_labels_n)
    total_memory += valid_bg_logits.nelement() * valid_bg_logits.element_size()
    total_memory += sys.getsizeof(valid_bg_logits)
    total_memory += bg_relations.nelement() * bg_relations.element_size()
    total_memory += sys.getsizeof(bg_relations)
    total_memory += fg_relations.nelement() * fg_relations.element_size()
    total_memory += sys.getsizeof(fg_relations)
    total_memory += sys.getsizeof(fg_num)
    total_memory += multiplication_bg.nelement() * multiplication_bg.element_size()
    total_memory += sys.getsizeof(multiplication_bg)
    total_memory += multiplication_fg.nelement() * multiplication_fg.element_size()
    total_memory += sys.getsizeof(multiplication_fg)
    total_memory += FP_num.nelement() * FP_num.element_size()
    total_memory += sys.getsizeof(FP_num)
    total_memory += rank_pos.nelement() * rank_pos.element_size()
    total_memory += sys.getsizeof(rank_pos)
    total_memory += rank.nelement() * rank.element_size()
    total_memory += sys.getsizeof(rank)
    total_memory += ranking_error.nelement() * ranking_error.element_size()
    total_memory += sys.getsizeof(ranking_error)
    total_memory += bucket_grads.nelement() * bucket_grads.element_size()
    total_memory += sys.getsizeof(bucket_grads)
    total_memory += duplication_bg.nelement() * duplication_bg.element_size()
    total_memory += sys.getsizeof(duplication_bg)
    total_memory += duplication_fg.nelement() * duplication_fg.element_size()
    total_memory += sys.getsizeof(duplication_fg)
    total_memory += ranking_error_of_irrelevant_fg.nelement() * ranking_error_of_irrelevant_fg.element_size()
    total_memory += sys.getsizeof(ranking_error_of_irrelevant_fg)

    print("Total Memory: ", total_memory / 1024 ** 2)
    return grad, 1 - metric


def RankSort(logits, targets, delta_RS=0.0, eps=1e-10):
    total_memory = 0
    st = time.time()
    classification_grads = torch.zeros(logits.shape)

    # Filter fg logits
    fg_labels = (targets > 0.)
    fg_logits = logits[fg_labels]
    fg_targets = targets[fg_labels]
    fg_num = len(fg_logits)

    # Do not use bg with scores less than minimum fg logit
    # since changing its score does not have an effect on precision
    threshold_logit = torch.min(fg_logits) - delta_RS
    relevant_bg_labels = ((targets == 0) & (logits >= threshold_logit))

    relevant_bg_logits = logits[relevant_bg_labels]
    relevant_bg_grad = torch.zeros(len(relevant_bg_logits))
    sorting_error = torch.zeros(fg_num)
    ranking_error = torch.zeros(fg_num)
    fg_grad = torch.zeros(fg_num)
    fg_grad_temp = torch.zeros(fg_num)
    # sort the fg logits
    order = torch.argsort(fg_logits)
    target_rank = []
    target_s = []
    # Loops over each positive following the order
    for ii in order:
        # Difference Transforms (x_ij)
        fg_relations = fg_logits - fg_logits[ii]
        bg_relations = relevant_bg_logits - fg_logits[ii]

        if delta_RS > 0:
            fg_relations = torch.clamp(fg_relations / (2 * delta_RS) + 0.5, min=0, max=1)
            bg_relations = torch.clamp(bg_relations / (2 * delta_RS) + 0.5, min=0, max=1)
        else:
            fg_relations = (fg_relations >= 0).float()
            bg_relations = (bg_relations >= 0).float()

        # Rank of ii among pos and false positive number (bg with larger scores)
        rank_pos = torch.sum(fg_relations)
        FP_num = torch.sum(bg_relations)

        # Rank of ii among all examples
        rank = rank_pos + FP_num

        # Ranking error of example ii. target_ranking_error is always 0. (Eq. 7)
        ranking_error[ii] = FP_num / rank

        # Current sorting error of example ii. (Eq. 7)
        current_sorting_error = torch.sum(fg_relations * (1 - fg_targets)) / rank_pos

        # Find examples in the target sorted order for example ii
        iou_relations = (fg_targets >= fg_targets[ii])
        target_sorted_order = iou_relations * fg_relations

        # The rank of ii among positives in sorted order
        rank_pos_target = torch.sum(target_sorted_order)
        target_rank.append(rank_pos_target)
        # Compute target sorting error. (Eq. 8)
        # Since target ranking error is 0, this is also total target error
        target_sorting_error = torch.sum(target_sorted_order * (1 - fg_targets)) / rank_pos_target

        target_s.append(target_sorting_error)
        # Compute sorting error on example ii
        sorting_error[ii] = current_sorting_error - target_sorting_error

        # Identity Update for Ranking Error
        if FP_num > eps:
            # For ii the update is the ranking error
            fg_grad[ii] -= ranking_error[ii]
            fg_grad_temp[ii] -= ranking_error[ii]
            # For negatives, distribute error via ranking pmf (i.e. bg_relations/FP_num)
            relevant_bg_grad += (bg_relations * (ranking_error[ii] / FP_num))

        # Find the positives that are misranked (the cause of the error)
        # These are the ones with smaller IoU but larger logits
        missorted_examples = (~ iou_relations) * fg_relations


        # Denominotor of sorting pmf
        sorting_pmf_denom = torch.sum(missorted_examples)

        # Identity Update for Sorting Error
        if sorting_pmf_denom > eps:
            # For ii the update is the sorting error
            fg_grad[ii] -= sorting_error[ii]

            # For positives, distribute error via sorting pmf (i.e. missorted_examples/sorting_pmf_denom)
            fg_grad += (missorted_examples * (sorting_error[ii] / sorting_pmf_denom))

    # Normalize gradients by number of positives
    classification_grads[fg_labels] = (fg_grad / fg_num)
    classification_grads[relevant_bg_labels] = (relevant_bg_grad / fg_num)
    et_2 = time.time()
    print("Elapsed Time:", (et_2 - st))
    total_memory += sys.getsizeof(classification_grads)
    total_memory += classification_grads.nelement() * classification_grads.element_size()
    total_memory += sys.getsizeof(fg_labels)
    total_memory += fg_labels.nelement() * fg_labels.element_size()
    total_memory += sys.getsizeof(fg_logits)
    total_memory += fg_logits.nelement() * fg_logits.element_size()
    total_memory += sys.getsizeof(fg_targets)
    total_memory += fg_targets.nelement() * fg_targets.element_size()
    total_memory += sys.getsizeof(fg_num)
    total_memory += sys.getsizeof(threshold_logit)
    total_memory += sys.getsizeof(relevant_bg_labels)
    total_memory += relevant_bg_labels.nelement() * relevant_bg_labels.element_size()
    total_memory += sys.getsizeof(relevant_bg_logits)
    total_memory += relevant_bg_logits.nelement() * relevant_bg_logits.element_size()
    total_memory += sys.getsizeof(relevant_bg_grad)
    total_memory += relevant_bg_grad.nelement() * relevant_bg_grad.element_size()
    total_memory += sys.getsizeof(sorting_error)
    total_memory += sorting_error.nelement() * sorting_error.element_size()
    total_memory += sys.getsizeof(ranking_error)
    total_memory += ranking_error.nelement() * ranking_error.element_size()
    total_memory += sys.getsizeof(fg_grad)
    total_memory += fg_grad.nelement() * fg_grad.element_size()
    total_memory += sys.getsizeof(fg_grad_temp)
    total_memory += fg_grad_temp.nelement() * fg_grad_temp.element_size()
    total_memory += sys.getsizeof(order)
    total_memory += order.nelement() * order.element_size()
    total_memory += sys.getsizeof(fg_relations)
    total_memory += fg_relations.nelement() * fg_relations.element_size()
    total_memory += sys.getsizeof(bg_relations)
    total_memory += bg_relations.nelement() * bg_relations.element_size()
    total_memory += sys.getsizeof(rank_pos)
    total_memory += sys.getsizeof(FP_num)
    total_memory += sys.getsizeof(rank)
    total_memory += sys.getsizeof(iou_relations)
    total_memory += iou_relations.nelement() * iou_relations.element_size()

    total_memory += sys.getsizeof(target_sorted_order)
    total_memory += target_sorted_order.nelement() * target_sorted_order.element_size()
    total_memory += sys.getsizeof(rank_pos_target)
    total_memory += sys.getsizeof(target_sorting_error)
    total_memory += sys.getsizeof(missorted_examples)
    total_memory += missorted_examples.nelement() * missorted_examples.element_size()
    total_memory += sys.getsizeof(sorting_pmf_denom)
    print("Total Memory: ", total_memory / 1024 ** 2)
    print("----------------------------------------")
    return classification_grads, ranking_error.mean(), sorting_error.mean()


def bucketedRankSort(logits, targets, delta_RS=0.0, eps=1e-10):
    try:
        st = time.time()
        total_memory = 0
        old_targets = targets

        p_indices = torch.nonzero(old_targets > 0).flatten()
        n_indices = torch.nonzero(old_targets == 0).flatten()

        p_and_n_indices = torch.cat((p_indices, n_indices))
        p_and_n_logits = logits[p_and_n_indices]
        p_and_n_targets = targets[p_and_n_indices]

        sorted_p_and_n_logits, sorted_p_and_n_indices = torch.sort(p_and_n_logits, descending=True)
        sorted_p_and_n_targets = p_and_n_targets[sorted_p_and_n_indices]
        et = time.time()
        print("Elapsed Time for Sorting:", (et - st))
        f_logits = sorted_p_and_n_logits[sorted_p_and_n_targets > 0]
        f_targets = sorted_p_and_n_targets[sorted_p_and_n_targets > 0]

        threshold_logit = torch.min(f_logits) - delta_RS
        irrelevant_b_index = torch.nonzero(sorted_p_and_n_logits >= threshold_logit)[-1] + 1
        relevant_logits = sorted_p_and_n_logits[0:irrelevant_b_index]
        relevant_targets = sorted_p_and_n_targets[0: irrelevant_b_index]

        relevant_f_indices = torch.nonzero(relevant_targets > 0)
        relevant_b_indices = torch.nonzero(relevant_targets == 0)

        grad = torch.zeros(targets.shape)  # .cuda()
        metric = torch.zeros(0)  # .cuda()

        if len(relevant_targets) == 0:
            return 1 - metric

        bucket_sizes_b = (torch.sub(relevant_f_indices[1:], relevant_f_indices[:len(relevant_f_indices) - 1]) - 1)
        bucket_sizes_b = torch.cat((torch.tensor([[relevant_f_indices[0]]], device='cpu'), bucket_sizes_b))
        bucket_sizes_b = torch.cat((bucket_sizes_b, torch.tensor([[relevant_b_indices[-1] -  relevant_f_indices[-1]]], device='cpu')))
        bucket_sizes_b = bucket_sizes_b[(bucket_sizes_b > 0)]
        bucket_sizes_b = bucket_sizes_b.reshape((len(bucket_sizes_b), 1))  # .cuda()
        l_bucket_sizes_b = [int(x) for x in torch.Tensor.tolist(bucket_sizes_b.flatten())]

        relevant_bg_buckets = torch.split(relevant_logits[relevant_b_indices], tuple(l_bucket_sizes_b))
        bg_bucket_mean = torch.Tensor([torch.mean(bucket) for bucket in list(relevant_bg_buckets)])

        bucket_sizes_f = torch.ones(len(f_logits))
        all_buckets = torch.cat((f_logits, bg_bucket_mean))

        all_buckets, indices_s = torch.sort(all_buckets, descending=True)

        new_targets = torch.zeros(len(all_buckets))
        new_targets[indices_s < len(f_logits)] = sorted_p_and_n_targets[sorted_p_and_n_targets != 0]

        # print("Calculation Started")
        if torch.max(new_targets) <= 0:
            return grad, metric


        labels_p = (new_targets > 0)
        allabels_p = torch.nonzero((sorted_p_and_n_targets > 0)).flatten()
        labels_n = torch.nonzero((sorted_p_and_n_targets[:irrelevant_b_index] == 0)).flatten()

        fg_logits = all_buckets[labels_p]

        threshold_logit = torch.min(fg_logits) - delta_RS

        # Ignore those negative j that satisfy (L_{ij}=0 for all positive i), to accelerate the AP-loss computation.
        valid_labels_n = ((new_targets == 0) & (all_buckets >= threshold_logit))

        valid_bg_logits = all_buckets[valid_labels_n]

        bg_relations = (valid_bg_logits[:, None] - fg_logits[None, :])  # .cuda()

        fg_relations = (fg_logits[:, None] - fg_logits[None, :])  # .cuda()
        if delta_RS > 0:
            fg_relations = torch.clamp(fg_relations / (2 * delta_RS) + 0.5, min=0, max=1)
            bg_relations = torch.clamp(bg_relations / (2 * delta_RS) + 0.5, min=0, max=1)
        else:
            fg_relations = (fg_relations >= 0).float()
            bg_relations = (bg_relations >= 0).float()

        fg_num = len(torch.nonzero((targets > 0)).flatten())

        # multiplication_bg = torch.mul(bg_relations, bucket_sizes_b.cuda())
        multiplication_bg = torch.mul(bg_relations, bucket_sizes_b)
        multiplication_fg = torch.mul(fg_relations, bucket_sizes_f)

        FP_num = torch.sum(multiplication_bg, axis=0)
        rank_pos = torch.sum(multiplication_fg, axis=0)

        #if delta_RS > 0:
        #    rank_pos += 0.5

        rank = rank_pos + FP_num
        ranking_error = (FP_num / rank.float())

        current_sorting_error = torch.sum(fg_relations.T * (1 - f_targets), axis=1) / rank_pos

        iou_relations = f_targets >= f_targets[:, None]

        target_sorted_order = fg_relations.T * iou_relations

        rank_pos_target = torch.sum(target_sorted_order, axis=1)

        target_sorting_error = torch.sum(target_sorted_order * (1 - f_targets), axis=1) / rank_pos_target
        sorting_error = current_sorting_error - target_sorting_error

        missorted_examples = fg_relations.t() * (~iou_relations)

        sorting_pmf_denom = torch.sum(missorted_examples, axis=1)
        FP_num[(FP_num == 0).nonzero()] = 1

        bucket_grads = (torch.sum(
            torch.mul(multiplication_bg, ranking_error * bucket_sizes_f.flatten()) / FP_num,
            axis=1) / bucket_sizes_b.flatten())

        duplication_bg = bucket_grads.repeat_interleave(bucket_sizes_b.flatten().type(torch.LongTensor))

        duplication_fg = ranking_error.repeat_interleave(
            bucket_sizes_f.flatten().type(torch.LongTensor))

        grad[p_and_n_indices[sorted_p_and_n_indices[:irrelevant_b_index][labels_n]]] = duplication_bg
        grad[p_and_n_indices[sorted_p_and_n_indices[allabels_p]]] = -duplication_fg

        # For ii the update is the sorting error
        grad[p_and_n_indices[sorted_p_and_n_indices[allabels_p]]] -= sorting_error * (sorting_pmf_denom != 0.)
        x = sorting_error / sorting_pmf_denom
        x[torch.isnan(x)] = 0
        x[torch.isinf(x)] = 0
        # For positives, distribute error via sorting pmf (i.e. missorted_examples/sorting_pmf_denom)
        grad[p_and_n_indices[sorted_p_and_n_indices[allabels_p]]] += torch.sum(missorted_examples.t() * x, axis=1)

        grad /= fg_num

        metric = torch.sum(1 - ranking_error, dim=0, keepdim=True) / fg_num
        et_2 = time.time()

        print("Elapsed Time:", (et_2 - st))
        sum_pos = torch.abs(torch.sum(grad[p_indices])).item()
        sum_neg = torch.abs(torch.sum(grad[n_indices])).item()

        total_memory += sys.getsizeof(old_targets)
        total_memory += old_targets.nelement() * old_targets.element_size()
        total_memory += sys.getsizeof(p_indices)
        total_memory += p_indices.nelement() * p_indices.element_size()
        total_memory += sys.getsizeof(n_indices)
        total_memory += n_indices.nelement() * n_indices.element_size()
        total_memory += sys.getsizeof(p_and_n_indices)
        total_memory += p_and_n_indices.nelement() * p_and_n_indices.element_size()
        total_memory += sys.getsizeof(p_and_n_logits)
        total_memory += p_and_n_logits.nelement() * p_and_n_logits.element_size()
        total_memory += sys.getsizeof(p_and_n_targets)
        total_memory += p_and_n_targets.nelement() * p_and_n_targets.element_size()
        total_memory += sys.getsizeof(sorted_p_and_n_logits)
        total_memory += sorted_p_and_n_logits.nelement() * sorted_p_and_n_logits.element_size()
        total_memory += sys.getsizeof(sorted_p_and_n_indices)
        total_memory += sorted_p_and_n_indices.nelement() * sorted_p_and_n_indices.element_size()
        total_memory += sys.getsizeof(sorted_p_and_n_targets)
        total_memory += sorted_p_and_n_targets.nelement() * sorted_p_and_n_targets.element_size()
        total_memory += sys.getsizeof(f_logits)
        total_memory += f_logits.nelement() * f_logits.element_size()
        total_memory += sys.getsizeof(f_targets)
        total_memory += f_targets.nelement() * f_targets.element_size()
        total_memory += sys.getsizeof(irrelevant_b_index)
        total_memory += sys.getsizeof(relevant_logits)
        total_memory += relevant_logits.nelement() * relevant_logits.element_size()
        total_memory += sys.getsizeof(relevant_targets)
        total_memory += relevant_targets.nelement() * relevant_targets.element_size()
        total_memory += sys.getsizeof(relevant_f_indices)
        total_memory += relevant_f_indices.nelement() * relevant_f_indices.element_size()
        total_memory += sys.getsizeof(relevant_b_indices)
        total_memory += relevant_b_indices.nelement() * relevant_b_indices.element_size()
        total_memory += sys.getsizeof(grad)
        total_memory += grad.nelement() * grad.element_size()
        total_memory += sys.getsizeof(metric)
        total_memory += metric.nelement() * metric.element_size()
        total_memory += sys.getsizeof(bucket_sizes_b)
        total_memory += bucket_sizes_b.nelement() * bucket_sizes_b.element_size()
        total_memory += sys.getsizeof(l_bucket_sizes_b)
        total_memory += sys.getsizeof(relevant_bg_buckets)
        total_memory += sys.getsizeof(bg_bucket_mean)
        total_memory += bg_bucket_mean.nelement() * bg_bucket_mean.element_size()
        total_memory += sys.getsizeof(bucket_sizes_f)
        total_memory += bucket_sizes_f.nelement() * bucket_sizes_f.element_size()
        total_memory += sys.getsizeof(all_buckets)
        total_memory += all_buckets.nelement() * all_buckets.element_size()
        total_memory += sys.getsizeof(indices_s)
        total_memory += indices_s.nelement() * indices_s.element_size()
        total_memory += sys.getsizeof(new_targets)
        total_memory += new_targets.nelement() * new_targets.element_size()
        total_memory += sys.getsizeof(labels_p)
        total_memory += labels_p.nelement() * labels_p.element_size()
        total_memory += sys.getsizeof(allabels_p)
        total_memory += allabels_p.nelement() * allabels_p.element_size()
        total_memory += sys.getsizeof(labels_n)
        total_memory += labels_n.nelement() * labels_n.element_size()
        total_memory += sys.getsizeof(fg_logits)
        total_memory += fg_logits.nelement() * fg_logits.element_size()
        total_memory += sys.getsizeof(threshold_logit)
        total_memory += sys.getsizeof(valid_labels_n)
        total_memory += valid_labels_n.nelement() * valid_labels_n.element_size()
        total_memory += sys.getsizeof(valid_bg_logits)
        total_memory += valid_bg_logits.nelement() * valid_bg_logits.element_size()
        total_memory += sys.getsizeof(bg_relations)
        total_memory += bg_relations.nelement() * bg_relations.element_size()
        total_memory += sys.getsizeof(fg_relations)
        total_memory += fg_relations.nelement() * fg_relations.element_size()
        total_memory += sys.getsizeof(fg_num)
        total_memory += sys.getsizeof(multiplication_bg)
        total_memory += multiplication_bg.nelement() * multiplication_bg.element_size()
        total_memory += sys.getsizeof(multiplication_fg)
        total_memory += multiplication_fg.nelement() * multiplication_fg.element_size()
        total_memory += sys.getsizeof(FP_num)
        total_memory += FP_num.nelement() * FP_num.element_size()
        total_memory += sys.getsizeof(rank_pos)
        total_memory += rank_pos.nelement() * rank_pos.element_size()
        total_memory += sys.getsizeof(rank)
        total_memory += rank.nelement() * rank.element_size()
        total_memory += sys.getsizeof(ranking_error)
        total_memory += ranking_error.nelement() * ranking_error.element_size()
        total_memory += sys.getsizeof(current_sorting_error)
        total_memory += current_sorting_error.nelement() * current_sorting_error.element_size()
        total_memory += sys.getsizeof(iou_relations)
        total_memory += iou_relations.nelement() * iou_relations.element_size()
        total_memory += sys.getsizeof(target_sorted_order)
        total_memory += target_sorted_order.nelement() * target_sorted_order.element_size()
        total_memory += sys.getsizeof(rank_pos_target)
        total_memory += rank_pos_target.nelement() * rank_pos_target.element_size()
        total_memory += sys.getsizeof(target_sorting_error)
        total_memory += target_sorting_error.nelement() * target_sorting_error.element_size()
        total_memory += sys.getsizeof(sorting_error)
        total_memory += sorting_error.nelement() * sorting_error.element_size()
        total_memory += sys.getsizeof(missorted_examples)
        total_memory += missorted_examples.nelement() * missorted_examples.element_size()
        total_memory += sys.getsizeof(sorting_pmf_denom)
        total_memory += sorting_pmf_denom.nelement() * sorting_pmf_denom.element_size()
        total_memory += sys.getsizeof(bucket_grads)
        total_memory += bucket_grads.nelement() * bucket_grads.element_size()
        total_memory += sys.getsizeof(duplication_bg)
        total_memory += duplication_bg.nelement() * duplication_bg.element_size()
        total_memory += sys.getsizeof(duplication_fg)
        total_memory += duplication_fg.nelement() * duplication_fg.element_size()
        print("Total Memory: ", total_memory / 1024 ** 2)
        print("----------------------------------------")
        return grad, 1 - metric, sorting_error.mean()
    except Exception as e:
        print(e)

def generate_logits(n, mean, std):
    return np.random.normal(mean, std, n)

import random

def generate_random_iou_values(num_values, min_iou=0.0, max_iou=1.0):
    """
    Generates a list of random IoU values.

    Parameters:
    num_values (int): Number of IoU values to generate.
    min_iou (float): Minimum possible value for IoU. Default is 0.0.
    max_iou (float): Maximum possible value for IoU. Default is 1.0.

    Returns:
    list: A list of random IoU values.
    """
    return [random.uniform(min_iou, max_iou) for _ in range(num_values)]

os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device("cpu")

total_num_of_examples = 100000
positive_ratio = 0.05

num_of_pos_logits = int(positive_ratio * total_num_of_examples)
num_of_neg_logits = total_num_of_examples - num_of_pos_logits

pos_logits = generate_logits(num_of_pos_logits, -1, 1)
pos_targets = generate_random_iou_values(num_of_pos_logits)
neg_logits = generate_logits(num_of_neg_logits, 1, 1)
neg_targets = torch.zeros(num_of_neg_logits, device=device)

#logits = torch.Tensor(np.concatenate((pos_logits, neg_logits)))
#targets = torch.cat((pos_targets, neg_targets))
#logits, indices = torch.sort(torch.Tensor(logits, device=device), descending=True)
#targets = torch.Tensor([1 if i < num_of_pos_logits else 0 for i in indices], device=device)

logits = torch.tensor([3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0], device=device)
targets = torch.tensor([0.0, 0., 0.4, 0.0, 0.0, 0.0, 0.0, 0.8], device=device)

g, r, s = RankSort(logits, targets, delta_RS=0.0)
g_2, r_2, s_2 = bucketedRankSort(logits, targets, delta_RS=0.0)
g_2, a = AP_loss(logits, targets)
g_2, a = bucketed_ap_loss(logits, targets)

