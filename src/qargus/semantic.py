from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from blockflow.programs.program import Program
from blockflow.semantic.executor import SemanticExecutor
from blockflow.semantic.state import StateVector
from blockflow.semantic.tracking import RunReport

from .activations import activation_apply
from .blocks import ResidualBlock


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return vec / norm


@dataclass
class SemanticResult:
    state: np.ndarray
    report: RunReport


@dataclass
class ResidualBlockStep:
    block: ResidualBlock

    def run_semantic(self, state: StateVector, report: RunReport) -> None:
        input_vec = np.asarray(state.data)
        conv_block = self.block.conv_block
        conv_vec = conv_block.semantic_apply(input_vec)
        report.include_use(
            success_prob=conv_block.success.success_prob,
            anc_clean=conv_block.resources.ancilla_qubits_clean,
            anc_dirty=conv_block.resources.ancilla_qubits_dirty,
        )
        conv_vec = _l2_normalize(conv_vec)
        act_vec = activation_apply(
            conv_vec,
            kind=self.block.activation,
            approx_coeffs=self.block.approx_coeffs,
            scale=self.block.activation_scale,
        )
        act_vec = _l2_normalize(act_vec)
        if act_vec.shape == input_vec.shape:
            tau = self.block.skip_tau
            act_vec = tau * input_vec + (1.0 - tau) * act_vec
        state.data = _l2_normalize(act_vec)


@dataclass
class ClassifierOutputStep:
    classifier: object
    output_dim: Optional[int] = None

    def run_semantic(self, state: StateVector, report: RunReport) -> None:
        block = self.classifier
        out = block.semantic_apply(state.data)
        if self.output_dim is not None:
            out = out[: self.output_dim]
        state.data = _l2_normalize(out)
        report.include_use(
            success_prob=block.success.success_prob,
            anc_clean=block.resources.ancilla_qubits_clean,
            anc_dirty=block.resources.ancilla_qubits_dirty,
        )


@dataclass
class Regime3OutputStep:
    output_padded_dim: int
    output_pool_size: int

    def run_semantic(self, state: StateVector, report: RunReport) -> None:
        vec = _l2_normalize(state.data)
        if self.output_padded_dim < vec.shape[0]:
            raise ValueError("output_padded_dim must be >= current dimension")
        if self.output_padded_dim > vec.shape[0]:
            padded = np.zeros(self.output_padded_dim, dtype=vec.dtype)
            padded[: vec.shape[0]] = vec
            vec = padded
        pool = self.output_pool_size
        if pool <= 0 or vec.shape[0] % pool != 0:
            raise ValueError("invalid pool size for output dimension")
        grouped = vec.reshape(-1, pool)
        pooled = np.sqrt(np.sum(np.abs(grouped) ** 2, axis=1))
        pooled = _l2_normalize(pooled)
        squared = np.abs(pooled) ** 2
        total = float(np.sum(squared))
        if total > 0.0:
            squared = squared / total
        state.data = squared


def build_semantic_program(model: object, *, include_output: bool = True) -> Program:
    program = Program()
    for block in getattr(model, "blocks", []):
        if not isinstance(block, ResidualBlock):
            raise ValueError("semantic program expects ResidualBlock instances")
        program.append(ResidualBlockStep(block=block))

    if include_output:
        output_mode = getattr(model, "output_mode", "regime3")
        if output_mode == "classifier":
            classifier = getattr(model, "classifier", None)
            if classifier is None:
                raise ValueError("classifier output_mode requires model.classifier")
            output_dim = getattr(model, "classifier_out_dim", None)
            program.append(ClassifierOutputStep(classifier=classifier, output_dim=output_dim))
        else:
            output_padded = int(getattr(model, "output_padded_dim"))
            output_pool = int(getattr(model, "output_pool_size"))
            program.append(Regime3OutputStep(output_padded_dim=output_padded, output_pool_size=output_pool))
    return program


def run_semantic_model(
    model: object,
    x: np.ndarray,
    *,
    include_output: bool = True,
    renormalize_each_step: bool = False,
) -> SemanticResult:
    expected_shape = getattr(model, "config").input_shape
    arr = np.asarray(x)
    if arr.shape != expected_shape:
        raise ValueError("input shape must match model config.input_shape")
    state = StateVector(_l2_normalize(arr.reshape(-1)))
    program = build_semantic_program(model, include_output=include_output)
    executor = SemanticExecutor()
    final_state, report = executor.run(
        program,
        state,
        renormalize_each_step=renormalize_each_step,
    )
    return SemanticResult(state=final_state.data, report=report)
