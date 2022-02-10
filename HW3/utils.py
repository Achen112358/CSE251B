import numpy as np
import torch

def iou(pred, target, n_classes = 10):
  ious = []
  pred = pred.view(-1)
  target = target.view(-1)

  # Ignore IoU for undefined class ("9")
  for cls in range(n_classes-1):  # last class is ignored

    pred_inds = pred == cls
    target_inds = target == cls
    intersection = torch.sum(pred_inds & target_inds)
    union = torch.sum(pred_inds | target_inds)
    if float(union) == 0:
      ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
    else:
      ious.append(float(intersection / union))

  return ious

def pixel_acc(pred, target):
    #TODO complete this function, make sure you don't calculate the accuracy for undefined class ("9")
    ret = pred == target
    ret_no_9 = ret[target != 9].to(torch.float)
    return float(torch.mean(ret_no_9))

