# QArgus

Lean, semantic (statevector-level) prototype of the Regime-3 constructions
from `LottoTicketAttempt2.pdf`, built on top of the `qublock` package.
This repository focuses on a toy but complete pipeline that is small enough
to run on a laptop while exposing the key primitives: vector-encoding ops,
QRAM-free convolution block encodings, non-linear activations, residual blocks,
and a semantic quantum-counting estimator.

## Scope
- Regime 3 (no QRAM) only
- Semantic execution only (no circuit synthesis beyond QuBlock primitives)
- Toy-scale models and datasets
- Output block matches Figure 1(c): flatten + normalize + L2 pool + square (set `output_mode="classifier"` for the legacy dense head)

## Install
```bash
pip install -e .[dev,demo]
```

## Quickstart
```python
import numpy as np
from qargus import QuantumResNet, QuantumResNetConfig

config = QuantumResNetConfig(input_shape=(4, 8, 8), num_classes=10, num_blocks=1)
model = QuantumResNet(config)
x = np.random.rand(*config.input_shape)
result = model.forward(x)
print(result.probabilities)
```

## Error Tracking
Use the `QuantumResNetConfig` error fields to accumulate approximation error in `ForwardResult.trace.epsilon`.

## Weights
You can import weights from sklearn-style linear models or Keras models:
`load_sklearn_classifier`, `load_keras_model`, `load_conv_filters`, and `load_classifier_weights`.
Keras kernels are converted automatically; bias terms are ignored.

## QASM Export
Use `model_to_qasm(model, input_state=...)` to export an OpenQASM 2.0 circuit
with decomposed gates via Qiskit/Aer. The export prepares the semantic state
per layer using fully unitary state-transfer circuits for a specific input.

## Unitary Subroutines (Reference)
The `qargus.unitary` and `qargus.qram` modules provide unitary reference
implementations for block-encoding, LCU skip connections, Grover-style
amplification, polynomial activation circuits (erf, square), and QRAM-style
loaders. These are correctness-first and not optimized for scale.

## Regime-3 Unitary Circuit
Use `build_regime3_unitary_circuit(model)` to generate a reference Regime-3
unitary circuit that composes block-encoding, polynomial activation, and LCU
skip connections. This is a reference implementation and not optimized for scale.

## Notebook
See `notebooks/mnist10_quantum_counting.ipynb` for a MNIST-10 demo that runs
the semantic model on a small subset and applies quantum counting.

## Tests
```bash
pytest
```
