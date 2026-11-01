import numpy as np
import pytest

from telepathy.negotiate.translators import fit_translator


def _synthetic(n=200, N=4, d=8, seed=0, orthogonal=False):
    rng = np.random.default_rng(seed)
    own = rng.standard_normal((n, N, d)).astype(np.float32)
    maps, biases = [], []
    for _ in range(N):
        A = np.linalg.qr(rng.standard_normal((d, d)))[0] if orthogonal else rng.standard_normal((d, d))
        maps.append(A.astype(np.float32))
        biases.append(rng.standard_normal(d).astype(np.float32) * (0 if orthogonal else 1))
    # peer = own @ inv(A) - so that peer @ A (+ bias) recovers own
    peer = np.stack([(own[:, i] - biases[i]) @ np.linalg.inv(maps[i]) for i in range(N)], 1).astype(np.float32)
    return peer, own


@pytest.mark.parametrize("family", ["linear", "mlp"])
def test_recovers_known_linear_map(family):
    peer, own = _synthetic()
    tr = fit_translator(family, peer[:150], own[:150])
    out = np.stack([tr.apply_message(m) for m in peer[150:]])
    rel = np.abs(out - own[150:]).max() / np.abs(own[150:]).max()
    assert rel < (0.02 if family == "linear" else 0.1)


def test_procrustes_recovers_rotation():
    peer, own = _synthetic(orthogonal=True)
    tr = fit_translator("procrustes", peer[:150], own[:150])
    out = np.stack([tr.apply_message(m) for m in peer[150:]])
    assert np.abs(out - own[150:]).max() < 1e-3


def test_full_message_linear():
    peer, own = _synthetic()
    tr = fit_translator("full_linear", peer[:180], own[:180])
    out = np.stack([tr.apply_message(m) for m in peer[180:]])
    assert np.abs(out - own[180:]).max() < 0.2
