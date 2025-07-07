import pathlib
import pickle

import bottleneck as bn
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import xgboost as xgb


modeldir = pathlib.Path('./models/qr_interp/')
#modeldir = pathlib.Path('./models/qr_small/')
modeldir.mkdir(exist_ok=True, parents=True)

X_train = np.copy(np.load('./data/small_interp/train_X_clean.npy').T)
y_train = np.load('./data/small_interp/train_y_clean.npy')
#X_train = np.copy(np.load('./data/small/train_X.npy').T)
#y_train = np.load('./data/small/train_y.npy')
X_test = np.copy(np.load('./data/small/test_X.npy').T)
y_test = np.load('./data/small/test_y.npy')

channels = np.load('./data/channels.npz')
hg_concept = 3000963
hg_indices = np.argwhere(channels['sources'] == hg_concept).squeeze()
hg_train = np.copy(X_train[:, hg_indices])
hg_test = np.copy(X_test[:, hg_indices])

n_train = (1, 10, 100, 1000, 10_000, 100_000)
full_qdmats = {}
hg_qdmats = {}
for n in n_train:
    Xy = xgb.QuantileDMatrix(X_train[:n], y_train[:n])
    Xy_test = xgb.QuantileDMatrix(X_test, y_test, ref=Xy)
    full_qdmats[n] = (Xy, Xy_test, X_train[:n], X_test)

    Xy = xgb.QuantileDMatrix(hg_train[:n], y_train[:n])
    Xy_test = xgb.QuantileDMatrix(hg_test, y_test, ref=Xy)
    hg_qdmats[n] = (Xy, Xy_test, hg_train[:n], hg_test)


def run_models(xy_sets, params):
    models = {}
    results = {}
    scores = {}

    for n, (Xy, Xy_test, X_train, X_test) in xy_sets.items():
        evals_result = {}
        models[n] = xgb.train(
            params,
            Xy,
            num_boost_round=256,
            early_stopping_rounds=8,
            # The evaluation result is a weighted average across multiple quantiles.
            evals=[(Xy, 'Train'), (Xy_test, 'Test')],
            evals_result=evals_result,
            verbose_eval=False,
        )
        results[n] = evals_result
        scores[n] = {
            'Train': models[n].inplace_predict(X_train),
            'Test': models[n].inplace_predict(X_test)
        }
    return models, results, scores


def save_results(name, xy_sets, params):
    models, results, scores = run_models(xy_sets, params)
    with open(modeldir/f'{name}.pkl', 'wb') as file:
        pickle.dump((models, results, scores), file, protocol=pickle.HIGHEST_PROTOCOL)


base_params = {
    'booster': 'gbtree',
    'tree_method': 'hist',
    'max_depth': 8,
    'learning_rate': 0.04,
    'subsample': 0.5,
    'nthread': 0,
}


if __name__ == '__main__':
    save_results('hg_only', hg_qdmats, {
        'objective': 'reg:quantileerror',
        'quantile_alpha': np.array([0.025, 0.500, 0.975]),
        **base_params
    })

    save_results('full_feat', full_qdmats, {
        'objective': 'reg:quantileerror',
        'quantile_alpha': np.array([0.025, 0.500, 0.975]),
        **base_params
    })

    save_results('p025', full_qdmats, {
        'objective': 'reg:quantileerror',
        'quantile_alpha': np.array([0.025]),
        **base_params
    })

    save_results('p500', full_qdmats, {
        'objective': 'reg:quantileerror',
        'quantile_alpha': np.array([0.500]),
        **base_params
    })

    save_results('p975', full_qdmats, {
        'objective': 'reg:quantileerror',
        'quantile_alpha': np.array([0.975]),
        **base_params
    })

    save_results('mse_reg', full_qdmats, {
        'objective': 'reg:squarederror',
        **base_params
    })
