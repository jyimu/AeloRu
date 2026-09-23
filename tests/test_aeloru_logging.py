import torch

from AeloRu.V3.aeloru_v3 import AeloruConfig, AeloruLayer


def test_config_exposes_logging_flags():
    cfg = AeloruConfig()
    assert cfg.logging_enabled is False
    assert cfg.diagnostic_enabled is False


def test_layer_respects_logging_flags():
    cfg = AeloruConfig(logging_enabled=False, diagnostic_enabled=False, r=2, in_features=4, out_features=4)
    layer = AeloruLayer(cfg.in_features, cfg.out_features, cfg)
    assert layer.should_log() is False
    assert layer.should_diagnose() is False

    layer.set_logging(enabled=True)
    layer.set_diagnostics(enabled=True)
    assert layer.should_log() is True
    assert layer.should_diagnose() is True

    # debug override should still be able to silence all output quickly
    layer.set_logging(enabled=False)
    layer.set_diagnostics(enabled=False)
    assert layer.should_log() is False
    assert layer.should_diagnose() is False
