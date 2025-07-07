import pathlib
import pickle
import numpy as np
from cml.ehr.curves import (build_age_curve,
                            build_demographic_curves,
                            calculate_age_stats)
from cml.ehr.dtypes import Cohort, ConceptMeta, EHR, Visits
from cml.ehr.samplespace import SampleSpace, SampleIndex, sample_uniform
from cml.label_expand import expand
from cml.recency_curves import eval_recency_curves

pathlib.Path('./data/small/').mkdir(exist_ok=True, parents=True)

meta = ConceptMeta.from_pickle_seq('./data/meta.pkl')
modes = ConceptMeta.make_mode_mapping(meta)

S0_concepts = np.unique(np.array([m.concept_id for m in meta]))
S1_concepts = np.unique(np.array([k for k, v in modes.items() if v == 'Measurement']))

demo_meta = ConceptMeta.from_pickle_seq('./data/demo_meta.pkl')
demo_concepts = np.unique([m.concept_id for m in demo_meta])

sources = np.concatenate([[3022304], demo_concepts, S0_concepts, S1_concepts])
codes = np.empty(len(sources), dtype=bytes)
codes[0] = b'a'
codes[1:1+len(demo_concepts)] = b'd'
codes[1+len(demo_concepts):1+len(demo_concepts)+len(S0_concepts)] = b'r'
codes[1+len(demo_concepts)+len(S0_concepts):] = b'v'
np.savez('./data/channels.npz', sources=sources, codes=codes)

# Make smaller subsets for easier development: 100k / 20k
train = EHR.from_npz('./data/splits/train_targets.npz')[:100_000]
test = EHR.from_npz('./data/splits/test_targets.npz')[:20_000]

np.save('./data/small/train_y.npy', train.value)
np.save('./data/small/test_y.npy', test.value)

# The curves are evaluated ten minutes before the target time
offset = np.timedelta64(600, 's')

train_index = SampleIndex(train.person_id, train.datetime - offset)
train_index.to_npz('./data/small/train_index.npz', compress=True)

test_index = SampleIndex(test.person_id, test.datetime - offset)
test_index.to_npz('./data/small/test_index.npz', compress=True)

ehr = EHR.from_npz('./data/ehr.npz')
space = SampleSpace.from_npz('./data/space.npz')
cohort = Cohort.from_pickle('./data/cohort.pkl')


# Make points and rectangular sparse matrix
def make_samples(index):
    points = eval_recency_curves(index, space, ehr, modes)
    S0, S1 = zip(*points)
    X = np.full((len(sources), len(points)), np.nan)

    X[0] = build_age_curve(index, cohort)
    X[1:len(demo_meta)+1] = build_demographic_curves(index, cohort, demo_concepts)

    X0 = X[1+len(demo_meta):1+len(demo_meta)+len(S0_concepts)]
    X1 = X[1+len(demo_meta)+len(S0_concepts):]

    check_index, check_columns, _ = expand(S0, S0_concepts, out=X0)
    assert np.all(check_index[0].astype(np.int64) == index.person_id)
    assert np.all(check_index[1].astype('M8[s]') == index.datetime)
    assert np.all(check_columns == S0_concepts)

    check_index, check_columns, _ = expand(S1, S1_concepts, out=X1)
    assert np.all(check_index[0].astype(np.int64) == index.person_id)
    assert np.all(check_index[1].astype('M8[s]') == index.datetime)
    assert np.all(check_columns == S1_concepts)
    return points, X


test_points, X_test = make_samples(test_index)
with open('./data/small/test_points.pkl', 'wb') as file:
    pickle.dump(test_points, file, protocol=pickle.HIGHEST_PROTOCOL)
np.save('./data/small/test_X.npy', X_test)

train_points, X_train = make_samples(train_index)
with open('./data/small/train_points.pkl', 'wb') as file:
    pickle.dump(train_points, file, protocol=pickle.HIGHEST_PROTOCOL)
np.save('./data/small/train_X.npy', X_train)
