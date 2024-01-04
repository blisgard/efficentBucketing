import torch
import time
import numpy as np
import os
# import psutil
import sys
import torch.nn.functional as F
import math


def AP_loss(logits, targets):
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

    tmp1 = fg_logits - fg_logits[0]
    tmp2 = valid_bg_logits - fg_logits[0]

    a = torch.sum(tmp1)
    b = torch.sum(tmp2)

    return grad, 1 - metric


def Efficent_AP_loss(logits, targets):
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

    return grad, 1 - metric


def partition(arr, target_2,pivot_index,low, high):
    # Swap the pivot with the last element
    arr_2 = arr.detach().numpy().copy()
    target = target_2.detach().numpy().copy()


    # Now the pivot is at the end
    pivot = arr_2[pivot_index]
    i = low - 1  # Start `i` at `low - 1` so it can be incremented before the first swap

    for j in range(low, high):
        # If current element is smaller than or equal to pivot
        if arr_2[j] <= pivot:
            i += 1  # Increment `i` to find the next spot for a smaller element
            # Swap elements at i and j
            arr_2[i], arr_2[j] = arr_2[j], arr_2[i]
            target[i], target[j] = target[j], target[i]

    # Put the pivot back in its correct place
    arr_2[i + 1], arr_2[high] = arr_2[high], arr_2[i + 1]

    target[i + 1], target[high] = target[high], target[i + 1]

    # Return the new pivot index, which is now `i + 1`
    target_2 = target
    return arr_2,torch.tensor(target_2),i + 1


def quick_sort(arr, target,low, high):
    st = time.time()
    grad = torch.zeros(arr.shape)  # .cuda()
    metric = torch.zeros(1)
    if low < high:
        # Select pivot position and put all the elements greaterssw
        # than pivot on left and smaller than pivot on right
        positives = torch.tensor(arr[target == 1])
        sorted_positives, pos_indices = torch.sort(positives, dim=0, descending=False)
        pi = low
        for i in range(len(sorted_positives)):
            pivot = np.where(sorted_positives[i].item() == arr)[0].item()
            arr, target, pi = partition(arr, target,pivot, pi, high)
            arr = torch.tensor(arr)
    et = time.time()
    print("Elapsed Time in Quick:", (et - st))
    target = torch.tensor(target, dtype=torch.float64)
    arr = torch.tensor(arr, dtype=torch.float64)
    indices_pos = torch.where(target == 1)[0]
    bucket_sizes_b = (torch.sub(indices_pos[1:], indices_pos[:len(indices_pos) - 1]) - 1)
    bucket_sizes_b = torch.cat((torch.tensor([indices_pos[0]], device='cpu'), bucket_sizes_b))
    bucket_sizes_b = torch.cat(
        (bucket_sizes_b, torch.tensor([len(arr) - indices_pos[-1] -1], device='cpu')))
    bucket_sizes_b = bucket_sizes_b[torch.nonzero(bucket_sizes_b)]
    l_bucket_sizes_b = [int(x) for x in torch.Tensor.tolist(bucket_sizes_b.flatten())]

    negative_logits = arr[target == 0]
    relevant_bg_buckets = torch.split(negative_logits, tuple(l_bucket_sizes_b))
    bg_bucket_mean = torch.tensor([torch.mean(torch.tensor(bucket, dtype=torch.float64)) for bucket in list(relevant_bg_buckets)])

    f_logits = arr[target == 1]
    bucket_sizes_f = torch.ones(len(f_logits))
    all_buckets = torch.cat((f_logits, bg_bucket_mean))

    all_buckets, indices_s = torch.sort(all_buckets, descending=True)

    new_targets = torch.zeros(len(all_buckets), dtype=torch.float64)
    new_targets[indices_s < len(f_logits)] = target[target != 0]

    if torch.max(new_targets) <= 0:
        return grad, metric

    labels_p = (new_targets > 0)
    allabels_p = torch.nonzero((target > 0)).flatten()
    labels_n = torch.nonzero((target == 0)).flatten()

    fg_logits = all_buckets[labels_p]
    delta_RS = 0.0
    threshold_logit = torch.min(fg_logits) - delta_RS

    # Ignore those negative j that satisfy (L_{ij}=0 for all positive i), to accelerate the AP-loss computation.
    valid_labels_n = (new_targets == 0)

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
    #bucket_sizes_b = torch.flip(bucket_sizes_b, dims=(0,1))
    multiplication_bg = torch.mul(bg_relations, torch.flip(bucket_sizes_b, dims=(0,1)))
    multiplication_fg = torch.mul(fg_relations, bucket_sizes_f)

    FP_num = torch.sum(multiplication_bg, axis=0)
    rank_pos = torch.sum(multiplication_fg, axis=0)

    # if delta_RS > 0:
    #    rank_pos += 0.5

    rank = rank_pos + FP_num
    ranking_error = (FP_num / rank.float())

    FP_num[(FP_num == 0).nonzero()] = 1

    bucket_grads = (torch.sum(
        torch.mul(multiplication_bg, ranking_error * bucket_sizes_f.flatten()) / FP_num,
        axis=1) / torch.flip(bucket_sizes_b, dims=(0,1)).flatten())
    duplication_bg = bucket_grads.repeat_interleave(torch.flip(bucket_sizes_b, dims=(0,1)).flatten().type(torch.LongTensor))


    duplication_fg = ranking_error.repeat_interleave(
        bucket_sizes_f.flatten().type(torch.LongTensor))
    grad[allabels_p] = torch.flip(duplication_fg, dims=(0,))
    grad[labels_n] = torch.flip(duplication_bg, dims=(0,))
    #grad[p_and_n_indices[sorted_p_and_n_indices[:irrelevant_b_index][labels_n]]] = duplication_bg
    #grad[p_and_n_indices[sorted_p_and_n_indices[allabels_p]]] = -duplication_fg
    #indics = find_indices(old_arr, arr)
    #grad = grad[indics]
    grad /= fg_num

    metric = torch.sum(1 - ranking_error, dim=0, keepdim=True) / fg_num
    et = time.time()

    print("Elapsed Time:", (et - st))

    return grad, 1 - metric

def find_indices(original, permutation):
    # Create a dictionary to map elements to their indices in the original list
    old_arr = torch.tensor(original, dtype=torch.float64)
    indices = []
    for element in old_arr:
        indice = torch.where(permutation == element)
        indices.append(indice[0].item())
    return indices

def generate_logits(n, mean, std):
    return np.random.normal(mean, std, n)

def find_median(arr):
    median = np.quantile(arr, 0.5, method='lower')
    index = np.where(arr == median)[0].item()
    return index, median
os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device("cpu")

total_num_of_examples = 100000
positive_ratio = 0.01

num_of_pos_logits = int(positive_ratio * total_num_of_examples)
num_of_neg_logits = total_num_of_examples - num_of_pos_logits

logits = generate_logits(num_of_neg_logits+num_of_pos_logits, 2, 1)
targets = torch.zeros(len(logits))
random_indices = np.random.choice(len(logits), num_of_pos_logits, replace=False)
targets[random_indices] = 1
#logits, indices = torch.sort(torch.tensor(logits, device=device), descending=True)
#targets = torch.Tensor([1 if i < num_of_pos_logits else 0 for i in indices], device=device)
#logits = torch.tensor([10,5,2,9,11,7,6,1,4,8,3])
#targets = torch.tensor([0,0,0,1,0,0,0,0,0,0,1])
logits = torch.tensor(logits)
targets = torch.tensor(targets)
grad_q, metric_q = quick_sort(logits, targets, 0, len(logits) - 1)
grad, metric = AP_loss(logits, targets)






x = 1
# This code is contributed by Muskan Kalra.




