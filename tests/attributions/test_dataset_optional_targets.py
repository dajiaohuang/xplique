"""Dataset inputs contain their targets for both public explanation entry points."""

import numpy as np
import pytest
import tensorflow as tf

from xplique.attributions import Occlusion, Saliency


@pytest.mark.parametrize("method_class", [Occlusion, Saliency])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("entry_point", ["explain", "__call__"])
def test_dataset_without_separate_targets(method_class, batched, entry_point):
    inputs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    targets = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    model_inputs = tf.keras.Input((2,))
    outputs = tf.keras.layers.Dense(2, use_bias=False, kernel_initializer="ones")(model_inputs)
    model = tf.keras.Model(model_inputs, outputs)
    kwargs = {"patch_size": 1, "patch_stride": 1} if method_class is Occlusion else {}
    method = method_class(model, **kwargs)
    expected = method(inputs, targets)
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
    if batched:
        dataset = dataset.batch(1)

    actual = getattr(method, entry_point)(dataset)

    np.testing.assert_allclose(actual, expected)
