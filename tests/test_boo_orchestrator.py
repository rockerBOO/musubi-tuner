"""Smoke tests for BooTrainerOrchestrator.

Does not run training — just verifies the orchestrator wires up correctly
and the dispatch table contains the expected extensions.
"""

from musubi_tuner.flux_2_train_network_boo import BooTrainerOrchestrator
from musubi_tuner.flux_2_train_network_self_flow import Flux2SelfFlowNetworkTrainer
from musubi_tuner.flux_2_train_network_wavelet_loss import Flux2WaveletLossNetworkTrainer


def test_boo_orchestrator_registers_both_extensions():
    trainer = BooTrainerOrchestrator()
    types = [type(e) for e in trainer._extensions]
    assert Flux2SelfFlowNetworkTrainer in types
    assert Flux2WaveletLossNetworkTrainer in types


def test_boo_orchestrator_self_flow_in_process_batch_table():
    trainer = BooTrainerOrchestrator()
    table_types = [type(e) for e in trainer._dispatch_table.get("process_batch", [])]
    assert Flux2SelfFlowNetworkTrainer in table_types


def test_boo_orchestrator_wavelet_in_compute_loss_table():
    trainer = BooTrainerOrchestrator()
    table_types = [type(e) for e in trainer._dispatch_table.get("compute_loss", [])]
    assert Flux2WaveletLossNetworkTrainer in table_types


def test_boo_orchestrator_extra_metadata_merges_both():
    import argparse
    trainer = BooTrainerOrchestrator()
    args = argparse.Namespace(
        self_flow=True,
        self_flow_gamma=0.8,
        self_flow_gamma_warmup_steps=0,
        mask_ratio=0.25,
        ema_decay=0.999,
        student_feature_layer=None,
        teacher_feature_layer=None,
        self_flow_teacher_coupling_prob=0.0,
        self_flow_teacher_coupling_decay="constant",
        self_flow_teacher_mismatch_ratio=1.0,
        wavelet_loss=True,
        wavelet_loss_alpha=0.1,
        wavelet_loss_primary=False,
        wavelet_loss_type=None,
        wavelet_loss_transform="swt",
        wavelet_loss_wavelet="sym7",
        wavelet_loss_level=1,
        wavelet_loss_band_weights=None,
        wavelet_loss_band_level_weights=None,
        wavelet_loss_quaternion_component_weights=None,
        wavelet_loss_ll_level_threshold=None,
    )
    meta = trainer.extra_metadata(args)
    assert "ss_self_flow" in meta
    assert "ss_wavelet_loss" in meta
