from torchmetrics.classification import BinaryCalibrationError
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import matplotlib.pyplot as plt
import torch
import torchmetrics

def get_uncertainty(preds, targets, n_bins=5, title=None, **kwargs):

    plt.figure()
    # Get the calibration errors
    cal_ece = BinaryCalibrationError(n_bins=n_bins, norm='l1', ignore_index=-1)
    cal_ece_val = cal_ece(preds,targets)
    cal_mce = BinaryCalibrationError(n_bins=n_bins, norm='max', ignore_index=-1)
    cal_mce_val = cal_mce(preds,targets)
    cal_rmsce = BinaryCalibrationError(n_bins=n_bins, norm='l2', ignore_index=-1)
    cal_rmsce_val = cal_rmsce(preds,targets)
    print(f'ECE: {cal_ece_val:.4f}')
    print(f'MCE: {cal_mce_val:.4f}')
    print(f'RMSCE: {cal_rmsce_val:.4f}')

    # Get the statistics for the calibration curve
    a,b = calibration_curve(targets, preds, n_bins=n_bins)
    # Plot perfectly calibrated
    plt.plot([0, 1], [0, 1], linestyle='--', label='Ideally Calibrated')
    #Plot the calibration curve
    plt.plot(b,a,marker = '.', label = 'Classifier')
    plt.legend(loc='upper left')
    plt.xlabel("Predicted Probabilities")
    plt.ylabel("True Probabilities")
    if title is not None:
        plt.title(title)

    # Save the image
    if 'save_dir' in kwargs:
        plt.savefig(kwargs['save_dir'], bbox_inches='tight', pad_inches=0)
    plt.show()

    #Plot the histogram for the predicted probabilities
    plt.figure()
    plt.hist(preds, bins=n_bins)
    plt.ylim(0, 2000)
    plt.title("Histogram of Predicted Probabilities")
    plt.show()

    # metrics = {'ECE': f'{cal_ece_val:.4f}',
    #             'MCE': f'{cal_mce_val:.4f}',
    #             'RMSCE': f'{cal_rmsce_val:.4f}'}

    metrics = {'ECE': round(cal_ece_val.item(), 4),
                'MCE': round(cal_mce_val.item(), 4),
                'RMSCE': round(cal_rmsce_val.item(), 4)}

    return metrics

def get_uncertainty_many(preds, targets, n_bins=5, title=None, labels=None, **kwargs):
    '''
    preds: (n_samples, n_types) tensor of predictions
    '''
    plt.figure()
    metrics = {}

    for i in range(preds.shape[1]):

        # Get the calibration errors
        cal_ece = BinaryCalibrationError(n_bins=n_bins, norm='l1', ignore_index=-1)
        cal_ece_val = cal_ece(preds[:,i],targets)
        cal_mce = BinaryCalibrationError(n_bins=n_bins, norm='max', ignore_index=-1)
        cal_mce_val = cal_mce(preds[:,i],targets)
        cal_rmsce = BinaryCalibrationError(n_bins=n_bins, norm='l2', ignore_index=-1)
        cal_rmsce_val = cal_rmsce(preds[:,i],targets)
        if title is not None:
            print(title)

        print(f'Type: {labels[i]}')
        print(f'ECE: {cal_ece_val:.4f}')
        print(f'MCE: {cal_mce_val:.4f}')
        print(f'RMSCE: {cal_rmsce_val:.4f}')

        metrics[f'{labels[i]}'] = {'ECE': round(cal_ece_val.item(), 4),
                   'MCE': round(cal_mce_val.item(), 4),
                   'RMSCE': round(cal_rmsce_val.item(), 4)}

        # Get the statistics for the calibration curve
        a,b = calibration_curve(targets, preds[:,i], n_bins=n_bins)

        #Plot the calibration curve
        plt.plot(b,a, marker = '.', label = labels[i])

    if title is not None:
        plt.title(title)
    # Plot perfectly calibrated
    plt.plot([0, 1], [0, 1], linestyle='--', label='Ideally Calibrated')
    plt.legend(loc='upper left')
    plt.xlabel("Predicted Probabilities")
    plt.ylabel("True Probabilities")

    # Save the image
    if 'save_dir' in kwargs:
        plt.savefig(kwargs['save_dir'], bbox_inches='tight', pad_inches=0)

    plt.show()


    #Plot the histogram for the predicted probabilities
    for i in range(preds.shape[1]):
        plt.figure()
        plt.hist(preds[:,i], bins=n_bins, label=labels[i])
        plt.ylim(0, 2000)
        plt.title(f"Histogram of Predicted Probabilities ({labels[i]})")
        plt.show()

    return metrics


#Get different metrics
def get_classification_metrics(preds, targets, title=None):
    accuracy = torchmetrics.Accuracy(task='binary')
    precision = torchmetrics.Precision(task='binary')
    recall = torchmetrics.Recall(task='binary')
    auroc = torchmetrics.AUROC(task='binary')

    if title is not None:
        print(f'Classification Metrics for {title}:')

    #Print the metrics
    print(f'Accuracy: {accuracy(preds, targets):.3f}')
    print(f'Precision: {precision(preds, targets):.3f}')
    print(f'Recall: {recall(preds, targets):.3f}')
    print(f'AUROC: {auroc(preds, targets):.3f}')

    # metrics =  {'Accuracy': f'{accuracy(preds, targets):.3f}',
    #                             'Precision': f'{precision(preds, targets):.3f}',
    #                             'Recall': f'{recall(preds, targets):.3f}',
    #                             'AUROC': f'{auroc(preds, targets):.3f}'}

    metrics = {'Accuracy': round(accuracy(preds, targets).item(),3),
                                'Precision': round(precision(preds, targets).item(),3),
                                'Recall': round(recall(preds, targets).item(),3),
                                'AUROC': round(auroc(preds, targets).item(),3)}

    return metrics