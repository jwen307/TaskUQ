import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from scipy.stats import beta, betabinom



def conformal_calibration(recon_preds, gt_preds, calib_idxs, alpha=0.05, conformal_method='residual'):
    # conformal_method: 'residual' or 'cqr' or 'studentized_residual'

    n = len(calib_idxs)
    scores = []

    for i in calib_idxs:
        # Get the predictions for this image
        yhat = recon_preds[i]
        yhat_gt = gt_preds[i]

        if conformal_method=='studentized_residual':
            # Get the conformal scores
            s = np.abs(yhat_gt - yhat.mean(axis=0))/yhat.std(axis=0)
        elif conformal_method=='residual': #Used for point estimate methods
            s = np.abs(yhat_gt - yhat)
        elif conformal_method=='cqr':
            lower_quantile = np.quantile(yhat, alpha/2, axis=0)
            upper_quantile = np.quantile(yhat, 1-alpha/2, axis=0)
            s = np.max([lower_quantile - yhat_gt, yhat_gt - upper_quantile])

        else:
            raise ValueError('Invalid conformal method')

        # Get the conformal scores
        scores.append(s)

    # Get the quantile
    qhat = np.quantile(scores, np.ceil((n + 1) * (1 - alpha)) / n, axis=0)

    return qhat


def conformal_inference(recon_preds, qhat, alpha = 0.05, conformal_method='residual'):
    # conformal_method: 'residual' or 'cqr' or 'studentized_residual'
    # recon_preds: (n, p) array of predictions for all the slices

    if conformal_method=='studentized_residual':
        uncertainty = qhat * recon_preds.std(axis=1)
        lower_interval = recon_preds.mean(axis=1) - uncertainty
        upper_interval = recon_preds.mean(axis=1) + uncertainty

    elif conformal_method=='residual':
        uncertainty = qhat
        lower_interval = recon_preds - uncertainty
        upper_interval = recon_preds + uncertainty

    elif conformal_method=='cqr':
        lower_interval = np.quantile(recon_preds, alpha/2, axis=1) - qhat
        upper_interval = np.quantile(recon_preds, 1-alpha/2, axis=1) + qhat

    else:
        raise ValueError('Invalid conformal method')

    interval_size = upper_interval - lower_interval

    return lower_interval, upper_interval, interval_size


def conformal_test(recon_preds, gt_preds, ys, test_idxs, qhat, alpha=0.05, conformal_method='residual', show_plots=False):

    num_in_interval = 0
    num_out_interval = 0

    out_interval_margin = []
    interval_size = []

    # Keep track of the number of points in each interval separated based on class
    num_in_interval_class = [0, 0]
    num_out_interval_class = [0, 0]

    # Keep track of the number of points in each interval separated based on interval size
    num_in_interval_size = [0, 0, 0, 0, 0] # 0-0.05, 0.05-0.1, 0.1-0.15, 0.15-0.2, > 0.2
    num_out_interval_size = [0, 0, 0, 0, 0] # 0-0.05, 0.05-0.1, 0.1-0.15, 0.15-0.2, > 0.2
    num_in_bin = [0, 0, 0, 0, 0] # 0-0.05, 0.05-0.1, 0.1-0.15, 0.15-0.2, > 0.2

    for i in test_idxs:
        yhat = recon_preds[i]
        yhat_gt = gt_preds[i]

        y = ys[i]

        if conformal_method=='studentized_residual':
            uncertainty = qhat * yhat.std()
            interval_size.append(2 * uncertainty)
            lower_interval = yhat.mean(axis=0) - uncertainty
            upper_interval = yhat.mean(axis=0) + uncertainty

        elif conformal_method=='residual':
            uncertainty = qhat
            interval_size.append( 2 * uncertainty )
            lower_interval = yhat - uncertainty
            upper_interval = yhat + uncertainty

        elif conformal_method=='cqr':
            lower_interval = np.quantile(yhat, alpha/2, axis=0) - qhat
            upper_interval = np.quantile(yhat, 1-alpha/2, axis=0) + qhat
            interval_size.append(upper_interval - lower_interval)

        else:
            raise ValueError('Invalid conformal method')

        # Get the bin based on the interval size
        bin = int(interval_size[-1] // 0.05)
        if bin >= len(num_in_interval_size):
            bin = 4

        num_in_bin[bin] += 1

        # Check if the ground truth is in the interval
        if yhat_gt >= lower_interval and yhat_gt <= upper_interval:
            num_in_interval += 1

            # Keep track of the number of points in each interval separated based on class
            num_in_interval_class[y] += 1

            # Keep track of the number of points in each interval separated based on interval size
            num_in_interval_size[bin] += 1


        else:
            num_out_interval += 1
            out_interval_margin.append(np.min([np.abs(yhat_gt - lower_interval), np.abs(yhat_gt - upper_interval)]))

            # Keep track of the number of points in each interval separated based on class
            num_out_interval_class[y] += 1

            # Keep track of the number of points in each interval separated based on interval size
            num_out_interval_size[bin] += 1


    coverage = num_in_interval / (num_in_interval + num_out_interval)
    class_0_coverage = num_in_interval_class[0] / (num_in_interval_class[0] + num_out_interval_class[0])
    class_1_coverage = num_in_interval_class[1] / (num_in_interval_class[1] + num_out_interval_class[1])

    bin_coverage = np.array(num_in_interval_size) / (np.array(num_in_interval_size) + np.array(num_out_interval_size))
    bin_coverage = np.nan_to_num(bin_coverage)

    if show_plots:
        # Plot the interval size
        plt.figure()
        plt.hist(interval_size, bins=100)
        plt.title('Interval Size (Conformal Method: {0})'.format(conformal_method))
        plt.ylabel('Frequency')
        plt.xlabel('Interval Size')
        plt.show()

        # Print the percentage of points in the interval

        print('Percentage in interval: {0:.3f}'.format(coverage))
        print('Average margin outside interval: {0:.3f}'.format(np.mean(out_interval_margin)))
        print('Average Interval Size: {0:.3f}'.format(np.mean(interval_size)))
        print('StdDev of Interval Size: {0:.3f}'.format(np.std(interval_size)))

        # Get the feature stratifed coverage
        print('Percentage in interval for class 0: {0:.3f}'.format(class_0_coverage))
        print('Percentage in interval for class 1: {0:.3f}'.format(class_1_coverage))

        # Get the coverage stratified by interval size
        print('Percentage in interval for interval size 0-0.05: {0:.3f} ({1:.0f})'.format(bin_coverage[0], num_in_bin[0]))
        print('Percentage in interval for interval size 0.05-0.1: {0:.3f} ({1:.0f})'.format(bin_coverage[1], num_in_bin[1]))
        print('Percentage in interval for interval size 0.1-0.15: {0:.3f} ({1:.0f})'.format(bin_coverage[2], num_in_bin[2]))
        print('Percentage in interval for interval size 0.15-0.2: {0:.3f} ({1:.0f})'.format(bin_coverage[3], num_in_bin[3]))
        print('Percentage in interval for interval size > 0.2: {0:.3f} ({1:.0f})'.format(bin_coverage[4], num_in_bin[4]))

    return coverage, np.mean(interval_size), np.std(interval_size), class_0_coverage, class_1_coverage, bin_coverage, num_in_bin


def conformal_eval(recon_preds, gt_preds, ys, calib_idxs, test_idxs, alpha=0.05, conformal_method='residual', show_plots=False):

    # Calibrate first
    qhat = conformal_calibration(recon_preds, gt_preds, calib_idxs, alpha=alpha, conformal_method=conformal_method)

    # Test the conformal intervals
    coverage, avg_interval_size, std_interval_size, class_0_coverage, class_1_coverage, bin_coverage, num_in_bin = conformal_test(recon_preds, gt_preds, ys, test_idxs, qhat, alpha=alpha, conformal_method=conformal_method, show_plots=show_plots)


    return coverage, avg_interval_size, std_interval_size, class_0_coverage, class_1_coverage, bin_coverage, num_in_bin



def conformal_eval_empirical(recon_preds, gt_preds, ys, calib_split = 0.7, alpha=0.05, num_trials=100, conformal_method='residual', **kwargs):
    # Get the size of the validation set
    n_val = len(gt_preds)

    # Run the conformal evaluation multiple times
    coverages = []
    avg_interval_sizes = []
    std_interval_sizes = []
    class_0_coverages = []
    class_1_coverages = []
    bin_coverages = []
    num_in_bins = []

    for i in tqdm(range(num_trials)):
        # Get the calibration and test indices
        idxs = torch.randperm(n_val).numpy()
        calib_idxs = idxs[:int(n_val * calib_split)]  # Calibrate on the first 70% of the validation set
        test_idxs = idxs[int(n_val * calib_split):]

        # Evaluate the conformal interval
        coverage, avg_interval_size, std_interval_size, class_0_coverage, class_1_coverage, bin_coverage, num_in_bin = conformal_eval(recon_preds, gt_preds, ys, calib_idxs, test_idxs, alpha=alpha, conformal_method=conformal_method, show_plots=False)

        # Log the results
        coverages.append(coverage)
        avg_interval_sizes.append(avg_interval_size)
        std_interval_sizes.append(std_interval_size)
        class_0_coverages.append(class_0_coverage)
        class_1_coverages.append(class_1_coverage)
        bin_coverages.append(bin_coverage)
        num_in_bins.append(num_in_bin)

    bin_coverages = np.stack(bin_coverages, axis=0)
    num_in_bins = np.stack(num_in_bins, axis=0)

    # Get the average results
    print('Results for Method: {0}'.format(conformal_method))
    print('Average Coverage: {0:.5f} +/- {1:.5f}'.format(np.mean(coverages), np.std(coverages)/np.sqrt(len(coverages))))
    print('Average Interval Size: {0:.5f} +/- {1:.5f}'.format(np.mean(avg_interval_sizes), np.std(avg_interval_sizes)/np.sqrt(len(avg_interval_sizes))))
    print('Average StdDev of Interval Size: {0:.5f} +/- {1:.5f}'.format(np.mean(std_interval_sizes), np.std(std_interval_sizes)/np.sqrt(len(std_interval_sizes))))
    print('Average Class 0 Coverage: {0:.5f} +/- {1:.5f}'.format(np.mean(class_0_coverages), np.std(class_0_coverages)/np.sqrt(len(class_0_coverages))))
    print('Average Class 1 Coverage: {0:.5f} +/- {1:.5f}'.format(np.mean(class_1_coverages), np.std(class_1_coverages)/np.sqrt(len(class_1_coverages))))
    print('Average Bin Coverage for interval size 0-0.05: {0:.5f} +/- {1:.5f} ({2:.0f})'.format(np.mean(bin_coverages[:,0]), np.std(bin_coverages[:,0])/np.sqrt(len(bin_coverages[:,0])), np.mean(num_in_bins[:,0])))
    print('Average Bin Coverage for interval size 0.05-0.1: {0:.5f} +/- {1:.5f} ({2:.0f})'.format(np.mean(bin_coverages[:,1]), np.std(bin_coverages[:,1])/np.sqrt(len(bin_coverages[:,1])), np.mean(num_in_bins[:,1])))
    print('Average Bin Coverage for interval size 0.1-0.15: {0:.5f} +/- {1:.5f} ({2:.0f})'.format(np.mean(bin_coverages[:,2]), np.std(bin_coverages[:,2])/np.sqrt(len(bin_coverages[:,2])) , np.mean(num_in_bins[:,2])))
    print('Average Bin Coverage for interval size 0.15-0.2: {0:.5f} +/- {1:.5f} ({2:.0f})'.format(np.mean(bin_coverages[:,3]), np.std(bin_coverages[:,3])/np.sqrt(len(bin_coverages[:,3])) , np.mean(num_in_bins[:,3])))
    print('Average Bin Coverage for interval size > 0.2: {0:.5f} +/- {1:.5f} ({2:.0f})'.format(np.mean(bin_coverages[:,4]), np.std(bin_coverages[:,4])/np.sqrt(len(bin_coverages[:,4])) , np.mean(num_in_bins[:,4])))



    # Plot a histogrom of the coverages
    plt.figure()
    plt.hist(coverages, bins=30, alpha=0.5, label='Empirical')

    plt.ylabel('Frequency')
    plt.xlabel('Coverage')


    # Plot the theoretical histogram of coverages for give T, calibration samples, and test samples
    n_calib = int(n_val * calib_split)
    n_test = int(n_val * (1 - calib_split))
    l = np.floor((n_calib + 1) * alpha)
    a = n_calib + 1 - l
    b = l

    rv = betabinom(n_test, a, b)
    # x = np.array(range(int(n_test * (0.80)), n_test))
    # plt.plot(x / n_test, rv.pmf(x) * n_test , label='Theoretical')
    theoretical_samples = rv.rvs(size=num_trials) / n_test
    plt.hist(theoretical_samples, bins=30, alpha=0.5, label='Theoretical')
    plt.legend()
    plt.title('Coverage Histogram over {0} Trials, n={1}, ntest={2}'.format(num_trials, n_calib, n_test))
    plt.show()

    if 'save_output_dir' in kwargs:
        save_output_dir = kwargs['save_output_dir']
        with open(save_output_dir + '/conformal_coverage_{0}.txt'.format(conformal_method), 'w') as f:
            f.write('Results for Method: {0}\n'.format(conformal_method))
            f.write('Average Coverage: {0:.5f} +/- {1:.5f}\n'.format(np.mean(coverages), np.std(coverages)/np.sqrt(len(coverages))))
            f.write('Average Interval Size: {0:.5f} +/- {1:.5f}\n'.format(np.mean(avg_interval_sizes), np.std(avg_interval_sizes)/np.sqrt(len(avg_interval_sizes))))
            f.write('Average StdDev of Interval Size: {0:.5f} +/- {1:.5f}\n'.format(np.mean(std_interval_sizes), np.std(std_interval_sizes)/np.sqrt(len(std_interval_sizes))))
            f.write('Average Class 0 Coverage: {0:.5f} +/- {1:.5f}\n'.format(np.mean(class_0_coverages), np.std(class_0_coverages)/np.sqrt(len(class_0_coverages))))
            f.write('Average Class 1 Coverage: {0:.5f} +/- {1:.5f}\n'.format(np.mean(class_1_coverages), np.std(class_1_coverages)/np.sqrt(len(class_1_coverages))))
            f.write('Average Bin Coverage for interval size 0-0.05: {0:.5f} +/- {1:.5f} ({2:.0f})\n'.format(np.mean(bin_coverages[:,0]), np.std(bin_coverages[:,0])/np.sqrt(len(bin_coverages[:,0])), np.mean(num_in_bins[:,0])))
            f.write('Average Bin Coverage for interval size 0.05-0.1: {0:.5f} +/- {1:.5f} ({2:.0f})\n'.format(np.mean(bin_coverages[:,1]), np.std(bin_coverages[:,1])/np.sqrt(len(bin_coverages[:,1])), np.mean(num_in_bins[:,1])))
            f.write('Average Bin Coverage for interval size 0.1-0.15: {0:.5f} +/- {1:.5f} ({2:.0f})\n'.format(np.mean(bin_coverages[:,2]), np.std(bin_coverages[:,2])/np.sqrt(len(bin_coverages[:,2])) , np.mean(num_in_bins[:,2])))
            f.write('Average Bin Coverage for interval size 0.15-0.2: {0:.5f} +/- {1:.5f} ({2:.0f})\n'.format(np.mean(bin_coverages[:,3]), np.std(bin_coverages[:,3])/np.sqrt(len(bin_coverages[:,3])) , np.mean(num_in_bins[:,3])))
            f.write('Average Bin Coverage for interval size > 0.2: {0:.5f} +/- {1:.5f} ({2:.0f})\n'.format(np.mean(bin_coverages[:,4]), np.std(bin_coverages[:,4])/np.sqrt(len(bin_coverages[:,4])) , np.mean(num_in_bins[:,4])))


    return coverages, theoretical_samples





