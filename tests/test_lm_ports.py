"""Telepathy-1 ports and translators: shapes, masks, quantization. No language model is loaded."""
import numpy as np
import torch

from telepathy.lm.ports import ReceivePort, SendPort
from telepathy.lm.translators import RelativeDecoder, Relative, fit, rel_coords


def test_send_port_shapes_and_quantization():
    torch.manual_seed(0)
    sp = SendPort(d_sender=48, N=8, d=64, width=32, heads=4)
    h = torch.randn(3, 20, 48)
    m = torch.ones(3, 20, dtype=torch.bool)
    m[1, 10:] = False
    for bits in (16, 8, 4):
        z = sp(h, m, bits)
        assert z.shape == (3, 8, 64)
    zb = sp(h, m, torch.tensor([16, 8, 4]))
    # 4-bit rows take at most 15 distinct values per slot (+- qmax and 0) times the scale
    assert len(torch.unique(zb[2, 0].detach())) <= 15


def test_send_port_ignores_masked_positions():
    torch.manual_seed(0)
    sp = SendPort(d_sender=16, N=4, d=8, width=32, heads=4)
    h = torch.randn(1, 10, 16)
    m = torch.ones(1, 10, dtype=torch.bool)
    m[0, 6:] = False
    h2 = h.clone()
    h2[0, 6:] = 100.0
    assert torch.allclose(sp(h, m), sp(h2, m), atol=1e-5)


def test_receive_port_starts_closed_and_masks_slots():
    torch.manual_seed(0)
    rp = ReceivePort(d_recv=32, N=4, d=8, heads=4)
    x = torch.randn(2, 5, 32)
    slots = torch.randn(2, 4, 8)
    mask = torch.ones(2, 4, dtype=torch.bool)
    mem, vis = rp.memory(slots, mask)
    assert mem.shape == (2, 5, 32) and vis.shape == (2, 5) and vis[:, 0].all()
    # zero output projections: the port is the identity at initialisation
    assert torch.allclose(rp(x, mem, vis, torch.ones(2)), x)
    with torch.no_grad():
        torch.nn.init.normal_(rp.xa.o.weight)
    mask2 = mask.clone()
    mask2[:, 2:] = False
    s2 = slots.clone()
    s2[:, 2:] = 50.0                     # missing slots carry garbage
    m1, v1 = rp.memory(slots, mask2)
    m2, v2 = rp.memory(s2, mask2)
    assert torch.allclose(rp(x, m1, v1, torch.ones(2)), rp(x, m2, v2, torch.ones(2)), atol=1e-5)
    # no slots at all is well defined (only the null slot is visible)
    m0, v0 = rp.memory(slots, torch.zeros(2, 4, dtype=torch.bool))
    assert torch.isfinite(rp(x, m0, v0, torch.zeros(2))).all()


def test_relative_translation_recovers_a_rotation():
    rng = np.random.default_rng(0)
    n, N, d, K = 2000, 4, 16, 64
    own = rng.standard_normal((n, N, d)).astype(np.float32)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    peer = (own @ Q.T * 3.0).astype(np.float32)   # a rotated and scaled dialect
    anchors = slice(0, K)
    dec = RelativeDecoder(own[anchors], own[K:1800])
    tr = Relative(peer[anchors], dec)
    test = slice(1800, 2000)
    out = tr.apply_batch(peer[test])
    cos = (out * own[test]).sum(-1) / (np.linalg.norm(out, axis=-1) * np.linalg.norm(own[test], axis=-1))
    assert cos.mean() > 0.8
    # relative coordinates are invariant to the rotation
    assert np.allclose(rel_coords(peer[test], peer[anchors]), rel_coords(own[test], own[anchors]), atol=1e-4)
    pr = fit("procrustes", peer[anchors], own[anchors])
    assert np.allclose(pr.apply_batch(peer[test]), own[test], atol=1e-3)
