# QArgus

QArgus is a lean, semantic (statevector-level) prototype of the Regime-3 constructions in
A. G. Rattew, P.-W. Huang, N. Guo, L. A. Pira, and P. Rebentrost,
"Accelerating Inference for Multilayer Neural Networks with Quantum Computers"
(arXiv:2510.07195, 2025). It builds on the `qublock` package and focuses on a complete
pipeline while exposing the key primitives:
vector-encoding ops, QRAM-free convolution block encodings, non-linear activations,
residual blocks, and a semantic quantum-counting estimator.


## Theory Summary
This repository uses amplitude encoding and block encodings to apply linear layers without QRAM.
This repository simulates those constructions at the statevector level and tracks success
probabilities and approximation error without building full circuits by default.

Key ideas implemented here:
- Vector encoding: inputs are L2-normalized and represented as amplitudes.
- Convolution and dense layers: represented as block-encoded linear operators.
- Non-linear activations: applied via polynomial (Chebyshev) approximations in the semantic model.
- Residual connections: implemented as a linear combination of unitaries (LCU).
- Output: flatten, normalize, L2-pool, then square to yield a probability vector.
- Quantum counting: a semantic phase-estimation style estimator for marked items.

## Repository Layout
- `src/qargus`: core library implementation.
- `notebooks`: demos and experiments.
- `tests`: unit tests for semantic ops, weights, unitary circuits, and QASM export.

## Install
Requires Python 3.9+.

Base install:
```bash
pip install -e .
```

Dev/demo extras (tests, sklearn, matplotlib):
```bash
pip install -e .[dev,demo]
```

Optional QASM/unitary dependencies:
```bash
pip install qiskit qiskit-aer
```

Conda environment:
```bash
conda env create -f environment.yml
conda activate qargus
```

## Quickstart
Inputs are channel-first arrays with shape `(channels, height, width)`. `forward` performs
its own L2-normalization, so inputs must be nonzero but do not need to be normalized.

```python
import numpy as np
from qargus import QuantumResNet, QuantumResNetConfig

config = QuantumResNetConfig(input_shape=(4, 8, 8), num_classes=10, num_blocks=1)
model = QuantumResNet(config)
x = np.random.rand(*config.input_shape)
result = model.forward(x)
print(result.probabilities)
print(result.trace.epsilon)
```

`result.probabilities` is a length `num_classes` vector. In `output_mode="regime3"` it is the
normalized squared L2-pooled output. In `output_mode="classifier"` it is a softmax over
dense logits.

## Configuration
`QuantumResNetConfig` fields:
- `input_shape`: `(channels, height, width)` channel-first input shape.
- `num_classes`: number of output classes.
- `num_blocks`: number of residual blocks (conv + activation + skip + normalize).
- `channels`: must match `input_shape[0]`.
- `kernel_size`: convolution kernel size; padding is `kernel_size // 2`.
- `activation`: `"erf"`, `"sigmoid"`, or `"identity"`.
- `activation_scale`: scales inputs before activation.
- `activation_error`, `normalize_error`, `pool_error`, `square_error`: additive error terms
  accumulated in `Trace.epsilon`.
- `pool_size`: optional output pool size in Regime-3 mode. If `None`, it is derived from
  `num_classes`.
- `output_mode`: `"regime3"` (default) or `"classifier"`.
- `seed`: RNG seed for random initialization.

## Outputs and Error Tracking
`QuantumResNet.forward` returns `ForwardResult`:
- `logits`: log-probabilities (Regime-3) or dense logits (classifier).
- `probabilities`: normalized probability vector.
- `encoded`: `EncodedTensor` holding the final `VectorEncoding`.
- `trace`: `Trace` with aggregated resource usage, success probability, and `epsilon`.

## Weights and Interoperability
By default, the model initializes random weights. You can load weights into an existing model:

```python
import numpy as np
from qargus import (
    QuantumResNet,
    QuantumResNetConfig,
    load_classifier_weights,
    load_conv_filters,
)

config = QuantumResNetConfig(
    input_shape=(1, 8, 8),
    num_classes=10,
    num_blocks=2,
    channels=1,
    kernel_size=3,
    output_mode="classifier",
)
model = QuantumResNet(config)

filters = [
    np.random.randn(config.channels, config.channels, config.kernel_size, config.kernel_size)
    for _ in range(config.num_blocks)
]
load_conv_filters(model, filters)

weights = np.random.randn(config.num_classes, model.flatten_dim)
load_classifier_weights(model, weights)
```

Other loaders:
- `load_sklearn_classifier(model, estimator)` expects a linear estimator with `coef_`.
- `load_keras_model(model, keras_model)` extracts Conv2D and Dense kernels. Bias terms are ignored.
- `conv_filters_from_keras` and `dense_weights_from_keras` convert raw Keras kernels.

Keras/TensorFlow is not a dependency of this project; install it separately if you use
the Keras helpers.

## Semantic Execution (Blockflow)
Use the Blockflow semantic executor to run a model step-by-step and collect a `RunReport`:

```python
from qargus import run_semantic_model

semantic = run_semantic_model(model, x)
print(semantic.state.shape)
print(semantic.report)
```

## Quantum Counting (Semantic)
`estimate_count(marks, precision_bits=6)` performs a semantic, phase-estimation style count
of marked items and returns a `QuantumCountingResult` with estimated count and query cost.

```python
from qargus import estimate_count

marks = [True, False, True, False, False, True]
result = estimate_count(marks, precision_bits=6)
print(result.estimated_count, result.oracle_queries)
```

## QASM Export
QASM export requires `qiskit` and `qiskit-aer`. It generates a circuit for a specific input.

```python
from qargus import model_to_qasm

qasm = model_to_qasm(model, input_state=x, include_measurements=False)
print(qasm.splitlines()[0])
```

Options:
- `strategy="unitary"` (default) builds a Regime-3 unitary circuit.
- `strategy="state"` exports per-layer state-preparation diagnostics.
- `per_layer=False` with `strategy="state"` only prepares the initial state.

## Unitary Reference Circuits
The `qargus.unitary` and `qargus.qram` modules provide correctness-first reference circuits
for block-encoding, polynomial activations (erf, square), LCU skip connections, and QRAM-style
loaders. These are not optimized for scale.

`build_regime3_unitary_circuit(model)` builds a reference Regime-3 unitary using block-encoding,
polynomial activations, and LCU skip connections. `regime3_data_qubits(model)` reports the number
of data qubits required for the model.

## Notebook
See `notebooks/mnist10_quantum_counting.ipynb` for a MNIST-10 demo that runs the semantic model
on a small subset and applies quantum counting.

## Tests
```bash
pytest
```

## License
MIT. See `LICENSE`.
