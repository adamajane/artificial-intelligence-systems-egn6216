"""Every Milestone 4 entry point sees only gap-corrected val rows (docs/3-milestone4-plan.md section 2.1).

Needs the local data/ mirror (or bucket access); a test whose input is not built yet is skipped.
"""

import pandas as pd
import pytest

from solarsentinel import storage
from solarsentinel.config import load_experiments
from solarsentinel.models import dataset

FIRST_VAL = pd.Timestamp("2015-02-05 06:00", tz="UTC")  # train's last slot 2015-01-09 06:00 + 27 days


def _val_first(meta: pd.DataFrame) -> pd.Timestamp:
    return meta.slot_time[meta.split == "val"].min()


def test_observations_loader():
    obs = dataset.load_observations("v2")
    assert _val_first(obs) == FIRST_VAL
    assert set(obs.split) == {"train", "val"}


def test_replay_labels_refused():
    with pytest.raises(ValueError):
        dataset.load_observations("v2", splits=("replay",))
    with pytest.raises(ValueError):
        dataset.load_observations("v2", splits=("replay",), columns=["HARPNUM", "label"])


@pytest.mark.parametrize("rel", ["processed/v2/cutouts", f"{load_experiments()['m4']['prefix']}/cutouts_stretch"])
def test_cutout_loaders(rel):
    if not storage.local_path(f"{rel}/index.parquet").exists():
        pytest.skip(f"{rel} not built yet")
    _, meta = dataset.load_cutouts(rel, version="v2")
    assert _val_first(meta) >= FIRST_VAL


def test_persistence_cases():
    from solarsentinel.models import baseline

    obs = dataset.load_observations("v2")
    pred = baseline.persistence(obs)
    assert len(pred) == len(obs)
    assert _val_first(obs[obs.split == "val"]) >= FIRST_VAL
