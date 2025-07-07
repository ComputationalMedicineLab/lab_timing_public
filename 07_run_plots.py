"""Produce PDFs of model results"""
import pickle
from pathlib import Path

import bottleneck as bn
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import xgboost as xgb
from matplotlib.backends.backend_pdf import PdfPages

from cml.plot.err_interval import plot_errors_and_intervals


def load_models(modeldir):
    with open(f'./{modeldir}/full_feat.pkl', 'rb') as file:
        full_feat = pickle.load(file)

    with open(f'./{modeldir}/hg_only.pkl', 'rb') as file:
        hg_only = pickle.load(file)

    with open(f'./{modeldir}/mse_reg.pkl', 'rb') as file:
        mse_reg = pickle.load(file)

    with open(f'./{modeldir}/p025.pkl', 'rb') as file:
        p025 = pickle.load(file)

    with open(f'./{modeldir}/p500.pkl', 'rb') as file:
        p500 = pickle.load(file)

    with open(f'./{modeldir}/p975.pkl', 'rb') as file:
        p975 = pickle.load(file)

    return (full_feat, hg_only, mse_reg, p025, p500, p975)


def plot_losses(full_feat, hg_only):
    fig, axes = plt.subplots(figsize=(12, 8), nrows=2, ncols=3, layout='constrained', sharey=True)
    for n, ax in zip((1, 10, 100, 1000, 10_000, 100_000), axes.ravel()):
        ax.plot(hg_only[1][n]['Train']['quantile'], color='C0', label='HG Only Train', linestyle='--', alpha=0.75)
        ax.plot(hg_only[1][n]['Test']['quantile'], color='C0', label='HG Only Test', alpha=0.5)

        ax.plot(full_feat[1][n]['Train']['quantile'], color='C1', label='Full Feature Train', linestyle='--', alpha=0.75)
        ax.plot(full_feat[1][n]['Test']['quantile'], color='C1', label='Full Feature Test', alpha=0.5)

        ax.legend()
        ax.set_title(f'{n:,} Training Instances')
    return fig, axes


def plot_scores(y_test, y_train, train_scores, test_scores):
    fig, axes = plt.subplots(figsize=(12, 12), nrows=6, ncols=2, layout='constrained', sharey=True)
    N = 200
    for (ax, (n, scores)) in zip(axes[:, 0].ravel(), train_scores.items()):
        # n = min(n, N)
        yorder = np.argsort(y_train[:n])
        xs = np.arange(len(yorder))
        lo, med, hi = scores[yorder].T
        ax.scatter(xs, med, color='C0', marker='.', alpha=0.5)
        ax.fill_between(xs, lo, hi, color='C0', alpha=0.2)
        ax.scatter(xs, y_train[yorder], color='C1', marker='+', alpha=0.5)
        ax.set_ylabel(f'Training {n=:,}')

    for (ax, (n, scores)) in zip(axes[:, 1].ravel(), test_scores.items()):
        xs = np.arange(N)
        yorder = np.argsort(y_test[:N])
        lo, med, hi = scores[yorder].T
        ax.scatter(xs, med, color='C0', marker='.', alpha=0.5)
        ax.fill_between(xs, lo, hi, color='C0', alpha=0.2)
        ax.scatter(xs, y_test[yorder], color='C1', marker='+', alpha=0.5)

    axes[0, 0].set_title(f'Train Set Scores [full]')
    axes[0, 1].set_title(f'Test Set Scores [{N}]')

    for ax in axes.ravel():
        ax.set_xticks([])
    return fig, axes


def plot_against_each_other(y_test, full_feat_test, hg_test, mse_test, p50_test):
    fig, axes = plt.subplots(figsize=(12, 8), nrows=2, ncols=2, layout='constrained', sharey=True, sharex=True)

    xs = np.arange(200)
    yorder = np.argsort(y_test[:200])

    def plot_(ax, scores, n=100_000):
        lo, med, hi = scores[n][yorder].T
        ax.scatter(xs, med, color='C0', marker='.', alpha=0.5)
        ax.fill_between(xs, lo, hi, color='C0', alpha=0.2)
        ax.scatter(xs, y_test[yorder], color='C1', marker='+', alpha=0.5)

    plot_(axes[0, 0], full_feat_test)
    plot_(axes[0, 1], hg_test)
    plot_(axes[1, 0], mse_test)
    plot_(axes[1, 1], p50_test)

    axes[0, 0].set_title('Full Feature')
    axes[0, 1].set_title('HG Only')
    axes[1, 0].set_title('MSE (3 Model)')
    axes[1, 1].set_title('p50 (3 Model)')

    fig.suptitle('200 Test Cases; 100K Training Instances')
    return fig, axes


def plot_err_intervals(y_test, full_feat_test, hg_test, mse_test, p50_test):
    fig, axes = plt.subplots(figsize=(12, 8), nrows=2, ncols=2, layout='constrained', sharey=True, sharex=True)

    xs = np.arange(200)

    def plot_(ax, scores, n=100_000):
        lo, med, hi = scores[n].T
        intervals = (np.abs(hi - lo) / 2) + 1e-12
        errors = med - y_test
        order = np.argsort(intervals[:400])
        plot_errors_and_intervals(errors[order], intervals[order], ax=ax)

    plot_(axes[0, 0], full_feat_test)
    plot_(axes[0, 1], hg_test)
    plot_(axes[1, 0], mse_test)
    plot_(axes[1, 1], p50_test)

    axes[0, 0].set_title('Full Feature')
    axes[0, 1].set_title('HG Only')
    axes[1, 0].set_title('MSE (3 Model)')
    axes[1, 1].set_title('p50 (3 Model)')

    fig.suptitle('200 Test Cases; 100K Training Instances')
    return fig, axes


def hist_intervals(full_feat_test, hg_test, mse_test):
    fig, ax = plt.subplots(figsize=(9, 6), nrows=1, ncols=1, layout='constrained', sharey=True, sharex=True)

    def plot_(ax, scores, label=''):
        lo, med, hi = scores[100_000].T
        ax.hist(hi-lo, bins=100, histtype='step', label=f'{label} ({np.mean(hi-lo):.4f})')

    plot_(ax, full_feat_test, 'Full Feat')
    plot_(ax, hg_test, 'HG Only')
    plot_(ax, mse_test, 'MSE')
    ax.legend()

    fig.suptitle('Test Cases Invervals; 100K Training Instances')
    return fig, ax


def cum_hist_intervals(full_feat_test, hg_test, mse_test):
    fig, ax = plt.subplots(figsize=(9, 6), nrows=1, ncols=1, layout='constrained', sharey=True, sharex=True)

    def plot_(ax, scores, label=''):
        lo, med, hi = scores[100_000].T
        ax.hist(hi-lo, bins=100, histtype='step', label=f'{label} ({np.median(hi-lo):.4f})', cumulative=True, density=True)

    ax.axhline(0.5, linestyle='--', color='black', alpha=0.5)
    plot_(ax, full_feat_test, 'Full Feat')
    plot_(ax, hg_test, 'HG Only')
    plot_(ax, mse_test, 'MSE')
    ax.legend()

    fig.suptitle('Test Cases Invervals; 100K Training Instances')
    return fig, ax


def plot_all(outfile, results, y_train, y_test):
    """Run all the plots for a set of results and training targets"""
    full_feat, hg_only, mse_reg, p025, p500, p975 = results

    full_feat_train = {}
    full_feat_test = {}
    for n in sorted(full_feat[2].keys()):
        full_feat_train[n] = full_feat[2][n]['Train']
        full_feat_test[n] = full_feat[2][n]['Test']

    hg_train = {}
    hg_test = {}
    for n in sorted(hg_only[2].keys()):
        hg_train[n] = hg_only[2][n]['Train']
        hg_test[n] = hg_only[2][n]['Test']

    mse_train = {}
    mse_test = {}
    p50_train = {}
    p50_test = {}
    for n in mse_reg[2].keys():
        mse_train[n] = np.stack((p025[2][n]['Train'], mse_reg[2][n]['Train'], p975[2][n]['Train']), axis=1)
        mse_test[n] = np.stack((p025[2][n]['Test'], mse_reg[2][n]['Test'], p975[2][n]['Test']), axis=1)

        p50_train[n] = np.stack((p025[2][n]['Train'], p500[2][n]['Train'], p975[2][n]['Train']), axis=1)
        p50_test[n] = np.stack((p025[2][n]['Test'], p500[2][n]['Test'], p975[2][n]['Test']), axis=1)

    with PdfPages(f'./pdfs/{outfile}.pdf') as pdf:
        plot_losses(full_feat, hg_only)
        pdf.savefig()
        plt.close()

        fig, axes = plot_scores(y_test, y_train, full_feat_train, full_feat_test)
        fig.suptitle('Train/Test Quantiles over the Full Feature Data')
        pdf.savefig()
        plt.close()

        fig, axes = plot_scores(y_test, y_train, hg_train, hg_test)
        fig.suptitle('Train/Test Set Scores using only Hemoglobin Recency and Last Value')
        pdf.savefig()
        plt.close()

        fig, axes = plot_scores(y_test, y_train, mse_train, mse_test)
        fig.suptitle('Train/Test Set Scores using MSE')
        pdf.savefig()
        plt.close()

        fig, axes = plot_scores(y_test, y_train, p50_train, p50_test)
        fig.suptitle('Train/Test Set Scores using Separate Models')
        pdf.savefig()
        plt.close()

        plot_against_each_other(y_test, full_feat_test, hg_test, mse_test, p50_test)
        pdf.savefig()
        plt.close()

        plot_err_intervals(y_test, full_feat_test, hg_test, mse_test, p50_test)
        pdf.savefig()
        plt.close()

        hist_intervals(full_feat_test, hg_test, mse_test)
        pdf.savefig()
        plt.close()

        cum_hist_intervals(full_feat_test, hg_test, mse_test)
        pdf.savefig()
        plt.close()

        lo, mid, hi = full_feat_test[100_000].T
        error = (mid - y_test)
        uncertainty = (hi - lo)
        plt.figure(figsize=(9, 9))
        plt.scatter(error, uncertainty, alpha=0.05)
        plt.xlabel('Error')
        plt.ylabel('Uncertainty')
        plt.title(f'Full Feature (Corr. {np.corrcoef(error, uncertainty)[0, 1]:.4f})')
        pdf.savefig()
        plt.close()

        lo, mid, hi = hg_test[100_000].T
        error = (mid - y_test)
        uncertainty = (hi - lo)
        plt.figure(figsize=(9, 9))
        plt.scatter(error, uncertainty, alpha=0.05)
        plt.xlabel('Error')
        plt.ylabel('Uncertainty')
        plt.title(f'HG Only (Corr. {np.corrcoef(error, uncertainty)[0, 1]:.4f})')
        pdf.savefig()
        plt.close()


if __name__ == '__main__':
    Path('./pdfs').mkdir(exist_ok=True)

    # Test set is same between base and random offset models
    y_test = np.load('./data/small/test_y.npy')

    # Plot the random offset models first
    y_train = np.load('./data/small_interp/train_y_clean.npy')
    results = load_models('./models/qr_interp')
    plot_all('interp_model', results, y_train, y_test)

    # Plot the base models next
    y_train = np.load('./data/small/train_y.npy')
    results = load_models('./models/qr_small')
    plot_all('base_model', results, y_train, y_test)
