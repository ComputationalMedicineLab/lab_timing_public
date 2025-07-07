import pathlib

import bottleneck as bn
import numpy as np

from cml.ehr.dtypes import EHR, Visits
from cml.ehr.samplespace import SampleSpace

hg_ehr = EHR.from_npz('./data/inpatient_hg.npz')
assert np.all(hg_ehr.concept_id == 3000963)
assert not bn.anynan(hg_ehr.value)

def split(x, perc=0.2):
    at = int(perc * len(x))
    np.random.shuffle(x)
    return x[:at], x[at:]

persons = np.unique(hg_ehr.person_id)
n20 = int(len(persons) * 0.2)
np.random.shuffle(persons)
val, test, train = persons[:n20], persons[n20:2*n20], persons[2*n20:]

val_ehr = hg_ehr[np.isin(hg_ehr.person_id, val)]
test_ehr = hg_ehr[np.isin(hg_ehr.person_id, test)]
train_ehr = hg_ehr[np.isin(hg_ehr.person_id, train)]

pathlib.Path('./data/splits/').mkdir(exist_ok=True, parents=True)
val_ehr.to_npz('./data/splits/eval_targets.npz', compress=True)
test_ehr.to_npz('./data/splits/test_targets.npz', compress=True)
train_ehr.to_npz('./data/splits/train_targets.npz', compress=True)
