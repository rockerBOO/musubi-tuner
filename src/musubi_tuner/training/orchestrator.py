"""TrainerOrchestrator — compose multiple NetworkTrainer extensions without
multiple inheritance.

Internal extension point — no API stability guarantees.
"""

from __future__ import annotations

import argparse
from typing import Any

from musubi_tuner.training.trainer_base import NetworkTrainer

VOID_HOOKS: frozenset[str] = frozenset({
    "handle_model_specific_args",
    "on_transformer_loaded",
    "on_train_start",
    "on_post_optimizer_step",
    "on_post_save",
    "on_before_sample_images",
    "on_after_sample_images",
})

DICT_MERGE_HOOKS: frozenset[str] = frozenset({
    "extra_metadata",
    "extra_step_logs",
})

CHAIN_HOOKS: frozenset[str] = frozenset({
    "extra_trainable_params",
})

CONTENDED_HOOKS: frozenset[str] = frozenset({
    "process_batch",
    "compute_loss",
})

ALL_HOOKS: frozenset[str] = VOID_HOOKS | DICT_MERGE_HOOKS | CHAIN_HOOKS | CONTENDED_HOOKS


class TrainerOrchestrator(NetworkTrainer):
    """Orchestrates multiple NetworkTrainer extensions without multiple inheritance.

    Register extensions via ``add_extension(ext)``. The orchestrator introspects
    each extension at registration time and builds a dispatch table of which hooks
    it overrides. Only overriding extensions are called for each hook.

    Composition rules
    -----------------
    void           call all extensions in registration order, ignore return value
    dict_merge     call all, merge returned dicts (later extensions win on key conflict)
    chain          thread the accumulator through each extension in order
    contended      raise NotImplementedError — subclass must override and handle manually

    Contended hooks: ``process_batch``, ``compute_loss``
    These are raised (not silently ignored) so the bug of forgetting to wire them
    is loud rather than silent.
    """

    def __init__(self) -> None:
        super().__init__()
        self._extensions: list[NetworkTrainer] = []
        self._dispatch_table: dict[str, list[NetworkTrainer]] = {}

    def add_extension(self, ext: NetworkTrainer) -> None:
        """Register an extension and update the dispatch table.

        Introspects ``type(ext)`` at call time; adding the same extension twice
        will register it twice (don't do that).
        """
        self._extensions.append(ext)
        for method_name in ALL_HOOKS:
            ext_method = getattr(type(ext), method_name, None)
            base_method = getattr(NetworkTrainer, method_name, None)
            if ext_method is not None and ext_method is not base_method:
                self._dispatch_table.setdefault(method_name, []).append(ext)

    # region void hooks

    def handle_model_specific_args(self, args: argparse.Namespace) -> None:
        for ext in self._dispatch_table.get("handle_model_specific_args", []):
            ext.handle_model_specific_args(args)

    def on_transformer_loaded(self, **kwargs: Any) -> None:  # type: ignore[override]
        for ext in self._dispatch_table.get("on_transformer_loaded", []):
            ext.on_transformer_loaded(**kwargs)

    def on_train_start(self, **kwargs: Any) -> None:  # type: ignore[override]
        for ext in self._dispatch_table.get("on_train_start", []):
            ext.on_train_start(**kwargs)

    def on_post_optimizer_step(self, **kwargs: Any) -> None:  # type: ignore[override]
        for ext in self._dispatch_table.get("on_post_optimizer_step", []):
            ext.on_post_optimizer_step(**kwargs)

    def on_post_save(self, **kwargs: Any) -> None:  # type: ignore[override]
        for ext in self._dispatch_table.get("on_post_save", []):
            ext.on_post_save(**kwargs)

    def on_before_sample_images(self, **kwargs: Any) -> None:  # type: ignore[override]
        for ext in self._dispatch_table.get("on_before_sample_images", []):
            ext.on_before_sample_images(**kwargs)

    def on_after_sample_images(self, **kwargs: Any) -> None:  # type: ignore[override]
        for ext in self._dispatch_table.get("on_after_sample_images", []):
            ext.on_after_sample_images(**kwargs)

    # endregion

    # region dict_merge hooks

    def extra_metadata(self, args: argparse.Namespace) -> dict:
        result: dict = {}
        for ext in self._dispatch_table.get("extra_metadata", []):
            result |= ext.extra_metadata(args)
        return result

    def extra_step_logs(self, args: argparse.Namespace, logs: dict) -> dict:
        result: dict = {}
        for ext in self._dispatch_table.get("extra_step_logs", []):
            result |= ext.extra_step_logs(args, logs)
        return result

    # endregion

    # region chain hooks

    def extra_trainable_params(self, args, accelerator, network, transformer, trainable_params: list) -> list:
        params = trainable_params
        for ext in self._dispatch_table.get("extra_trainable_params", []):
            params = ext.extra_trainable_params(args, accelerator, network, transformer, params)
        return params

    # endregion

    # region contended hooks — subclass must override if any extension registers

    def process_batch(self, **kwargs: Any):  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        if self._dispatch_table.get("process_batch"):
            names = [type(e).__name__ for e in self._dispatch_table["process_batch"]]
            raise NotImplementedError(
                f"{type(self).__name__} must override process_batch — "
                f"extensions {names} register for it and composition is not automatic."
            )
        return super().process_batch(**kwargs)

    def compute_loss(self, **kwargs: Any):  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        if self._dispatch_table.get("compute_loss"):
            names = [type(e).__name__ for e in self._dispatch_table["compute_loss"]]
            raise NotImplementedError(
                f"{type(self).__name__} must override compute_loss — "
                f"extensions {names} register for it and composition is not automatic."
            )
        return super().compute_loss(**kwargs)

    # endregion
