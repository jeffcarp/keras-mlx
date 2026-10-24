from keras.src.backend.common.name_scope import name_scope
from keras_mlx.src import ops
from keras_mlx.src import random
from keras_mlx.src import rnn
from keras_mlx.src.ops.core import compute_output_spec
from keras_mlx.src.ops.core import device_scope
from keras_mlx.src.variable import Variable

SUPPORTS_SPARSE_TENSORS = False
SUPPORTS_RAGGED_TENSORS = False
SUPPORTS_COMPLEX_DTYPES = True
SUPPORTS_GRADIENT = True
# TODO: follow updates and adjust to thread safe when possible
IS_THREAD_SAFE = False  # False as of mlx 0.24.0

distribution_lib = None


def _register_tf_converter():
    try:
        import sys
        if "tensorflow" in sys.modules:
            import mlx.core as mx
            import numpy as np
            import tensorflow as tf

            tf.register_tensor_conversion_function(
                mx.array,
                lambda val, dtype=None, name=None, as_ref=False: tf.convert_to_tensor(
                    np.array(val), dtype=dtype, name=name
                ),
            )
    except (ImportError, AttributeError):
        pass


import sys  # noqa: E402

if "tensorflow" in sys.modules:
    _register_tf_converter()
else:
    class _TFImportHook:
        @classmethod
        def find_spec(cls, fullname, path=None, target=None):
            if fullname == "tensorflow":
                if cls in sys.meta_path:
                    sys.meta_path.remove(cls)
                import importlib
                importlib.import_module("tensorflow")
                _register_tf_converter()
            return None

    sys.meta_path.insert(0, _TFImportHook)
