import os
import os.path as osp
os.chdir(osp.abspath(osp.dirname(osp.dirname(__file__))))
import sys
sys.path.append(os.curdir)
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix
from scipy.ndimage import minimum_filter, maximum_filter


def computeF1(FP, TP, FN, TN):

    return 2*TP / np.maximum((2*TP + FN + FP), 1e-32)


def extractGTs(gt, erodeKernSize=15, dilateKernSize=11):
    gt1 = minimum_filter(gt, erodeKernSize)
    gt0 = np.logical_not(maximum_filter(gt, dilateKernSize))
    return gt0, gt1


def computeMetricsContinue(values, gt0, gt1):
    values = values.flatten().astype(np.float32)
    gt0 = gt0.flatten().astype(np.float32)
    gt1 = gt1.flatten().astype(np.float32)

    inds = np.argsort(values)
    inds = inds[(gt0[inds] + gt1[inds]) > 0]
    vet_th = values[inds]
    gt0 = gt0[inds]
    gt1 = gt1[inds]

    TN = np.cumsum(gt0)
    FN = np.cumsum(gt1)
    FP = np.sum(gt0) - TN
    TP = np.sum(gt1) - FN

    msk = np.pad(vet_th[1:] > vet_th[:-1], (0, 1), mode='constant', constant_values=True)
    FP = FP[msk]
    TP = TP[msk]
    FN = FN[msk]
    TN = TN[msk]
    vet_th = vet_th[msk]

    return FP, TP, FN, TN, vet_th


def computeMetrics_th(values, gt, gt0, gt1, th):
    values = values >= th
    values = values.flatten().astype(np.uint8)
    gt = gt.flatten().astype(np.uint8)
    gt0 = gt0.flatten().astype(np.uint8)
    gt1 = gt1.flatten().astype(np.uint8)

    gt = gt[(gt0 + gt1) > 0]
    values = values[(gt0 + gt1) > 0]
    cm = confusion_matrix(gt, values)
    try:
        TN = cm[0, 0]
    except:
        TN = 0
    try:
        FN = cm[1, 0]
    except:
        FN = 0
    try:
        FP = cm[0, 1]
    except:
        FP = 0
    try:
        TP = cm[1, 1]
    except:
        TP = 0

    return FP, TP, FN, TN

def computeLocalizationMetrics(map, gt):
    gt0, gt1 = extractGTs(gt)

    # best threshold
    try:
        FP, TP, FN, TN, _ = computeMetricsContinue(map, gt0, gt1)
        f1 = computeF1(FP, TP, FN, TN)
        f1i = computeF1(TN, FN, TP, FP)
        F1_best = max(np.max(f1), np.max(f1i))
    except:
        import traceback
        traceback.print_exc()
        F1_best = np.nan

    # fixed threshold
    try:
        FP, TP, FN, TN = computeMetrics_th(map, gt, gt0, gt1, 0.5)
        f1 = computeF1(FP, TP, FN, TN)
        f1i = computeF1(TN, FN, TP, FP)
        F1_th = max(f1, f1i)
    except:
        import traceback
        traceback.print_exc()
        F1_th = np.nan

    return F1_best, F1_th


def get_results():
    test_datasets = ['NIST', 'Columbia', 'Cover', 'CoCoGlide', 'Casiav1']
    datasets_dir = '/mnt/data1/yangzhou/datasets/'

    for test_dataset in test_datasets:
        total_f1_fixed = []
        total_f1_best = []
        for idx, img_name in enumerate(os.listdir(datasets_dir + test_dataset + '/' + 'seg')):
            img_path = datasets_dir + test_dataset + '/' + 'seg' + '/' + img_name
            mask_path = datasets_dir + test_dataset + '/' + 'mask' + '/' + img_name
         
            seg_logit = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            seg_logit_np = np.array(seg_logit).astype(np.float32) / 255

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_np = np.array(mask).astype(np.float32)

            f1_best, f1_fixed = computeLocalizationMetrics(seg_logit_np, mask_np)

            total_f1_fixed.append(f1_fixed)
            total_f1_best.append(f1_best)

            if idx % 50 == 0:
                print("total:%d, current:%d, dataset:%s" % (len(os.listdir(datasets_dir + test_dataset + '/seg')), idx, test_dataset))

        res_str = " dataset: " + test_dataset + " best f1: " + str(np.sum(total_f1_best) / len(total_f1_best)) + " fixed f1: " + str(np.sum(total_f1_fixed) / len(total_f1_fixed))

        print(res_str)



if __name__ == "__main__":
    get_results()





