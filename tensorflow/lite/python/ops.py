import enum
from tensorflow.python.util.tf_export import tf_export as _tf_export


@_tf_export("lite.Ops")
class Ops(enum.IntEnum):
    # TODO Should be the same as in tensorflow/lite/schema/schema_generated.h
    # Check how to do it properly.
    Add = 0
    Mul = 18
