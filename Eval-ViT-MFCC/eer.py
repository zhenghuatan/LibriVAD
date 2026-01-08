import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve
from collections import Counter
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import sys
import code
score = sys.argv[1]
label = sys.argv[2]

def compute_mindcf(score, label, priors, costs):
    """Compute the minimum detection cost function (minDCF) for a given classifier."""
    fpr, tpr, thresholds = roc_curve(label, score, pos_label=1)  # Specify target class label
    # False Negative Rate (FNR) = 1 - True Positive Rate (TPR)
    fnr = 1 - tpr
    C_fn, C_fp = costs
    P_target, P_non_target = priors
    min_dcf = np.min(C_fn * P_target * fnr + C_fp * P_non_target * fpr)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return min_dcf, fpr, fnr, eer


# Define the prior probabilities for the target and non-target class
priors = [0.01, 1-0.01]  # Equal priors for simplicity
print('priors', priors)

# Define the cost of false negative and false positive (C_fn, C_fp)
costs = [10, 1]  # Equal cost for false negative and false positive
print('costs', costs)


score = np.loadtxt(str(score)).astype(float)
label = np.loadtxt(str(label)).astype(int)
print(score)
print(label)
if score.shape[0] != len(label):
   raise ValueError("mismatch number of score and labels!")
min_dcf_value_mfcc_vit, fpr, fnr, eer = compute_mindcf(score, label, priors, costs)

print('MinDCF', min_dcf_value_mfcc_vit, 'EER(%)', eer*100)

