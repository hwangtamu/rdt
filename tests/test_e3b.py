"""Protocol sanity tests for the corrected E3-B token channel."""

import math

import torch

from w2c.train_e3b import TokenVIB, make_batch


def _model(d_in=6, d=3, pooling="last", **kwargs):
    return TokenVIB(d_in, d, seed=0, pooling=pooling, **kwargs)


def test_analytic_kl_and_padding_zero():
    torch.manual_seed(0)
    h = torch.randn(2, 5, 6)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]], dtype=torch.float32)
    model = _model()
    mu = model.mu(h)
    logvar = torch.clamp(model.logvar(h), -8.0, 8.0)
    kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(-1) / math.log(2)
    _, rate = model(h, mask, mc=1, sample=True)
    assert torch.allclose(rate, (kl * mask).sum(1), atol=1e-5)
    # Padding contributes exactly zero.
    _, rate_pad = model(h[:, :2], mask[:, :2], mc=1, sample=True)
    assert abs(rate_pad[1].item() - rate[1].item()) < 1e-5


def test_total_rate_is_sum_not_mean():
    torch.manual_seed(0)
    h = torch.randn(1, 2, 6)
    model = _model()
    _, r2 = model(h, torch.ones(1, 2), mc=1, sample=True)
    _, r4 = model(h.repeat(1, 2, 1), torch.ones(1, 4), mc=1, sample=True)
    assert abs(r4.item() - 2.0 * r2.item()) < 1e-5


def test_identity_bottleneck_matches_raw_last_token():
    torch.manual_seed(0)
    h = torch.randn(3, 4, 6)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 0]],
                        dtype=torch.float32)
    model = _model(d_in=6, d=6, pooling="last", identity_init=True)
    with torch.no_grad():
        model.head.weight.normal_()
        model.head.bias.zero_()
        logits, _ = model(h, mask, mc=1, sample=False)
        idx = mask.sum(1).long() - 1
        expected = model.head(h[torch.arange(3), idx])
    assert torch.allclose(logits[0], expected, atol=1e-5)


def test_fixed_support_prior_has_zero_rate_and_no_mask_dependence():
    torch.manual_seed(0)
    tokens = [torch.randn(5, 6), torch.randn(3, 6), torch.randn(1, 6)]
    model = _model(pooling="attention_fixed_support")
    h, mask = make_batch(tokens, [0, 1, 2], torch.device("cpu"), support_length=7)
    _, rate = model(h, mask, mc=2, sample=True)
    assert rate.shape == (3,)
    assert torch.all(rate > 0)
    _, prior_rate = model(h, mask, mc=2, sample=True, prior=True)
    assert torch.equal(prior_rate, torch.zeros_like(prior_rate))
    # Same fixed support for every example, including all-padding positions.
    assert h.shape[1] == 7


def test_forced_prior_is_input_independent_given_same_shape():
    torch.manual_seed(0)
    h = torch.randn(2, 5, 6)
    mask = torch.ones(2, 5)
    model = _model(pooling="last")
    torch.manual_seed(123)
    logits_a, _ = model(h, mask, mc=4, sample=True, prior=True)
    torch.manual_seed(123)
    logits_b, _ = model(h.flip(0), mask, mc=4, sample=True, prior=True)
    assert torch.allclose(logits_a, logits_b, atol=1e-6)
