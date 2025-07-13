"""
ONNX Test Model Generator with Past Key-Value Caching

This script creates a test ONNX model that simulates a 2-layer transformer model
with past key-value caching functionality. The model passes through input tensors
and manages key-value cache states for efficient inference.

Inputs:
    - INPUT__0 (TensorProto.FLOAT): Main input tensor of shape [1, 1]
    - past_key_values.0.key (TensorProto.INT64): Past key cache for layer 0, shape [1, 2]
    - past_key_values.0.value (TensorProto.INT64): Past value cache for layer 0, shape [1, 2]
    - past_key_values.1.key (TensorProto.INT64): Past key cache for layer 1, shape [1, 2]
    - past_key_values.1.value (TensorProto.INT64): Past value cache for layer 1, shape [1, 2]

Outputs:
    - OUTPUT__0 (TensorProto.FLOAT): Main output tensor of shape [1, 1]
    - present_key_values.0.key (TensorProto.INT64): Updated key cache for layer 0, shape [1, 2]
    - present_key_values.0.value (TensorProto.INT64): Updated value cache for layer 0, shape [1, 2]
    - present_key_values.1.key (TensorProto.INT64): Updated key cache for layer 1, shape [1, 2]
    - present_key_values.1.value (TensorProto.INT64): Updated value cache for layer 1, shape [1, 2]

The model uses Identity nodes to pass through all tensors unchanged, making it suitable
for testing ONNX runtime integration with key-value caching patterns commonly used
in transformer models.

Generated file: with_past_kv.onnx
"""

from onnx import helper, TensorProto, __version__ as onnx_version
import onnx



# Suppose a 2-layer model where each KV tensor is shape [1,2]
kv_shape = [1, 2]
inputs = []
outputs = []
nodes = []

# original I/O
inputs.append(helper.make_tensor_value_info(
    "INPUT__0", TensorProto.FLOAT, [1, 1]))
outputs.append(helper.make_tensor_value_info(
    "OUTPUT__0", TensorProto.FLOAT, [1, 1]))
nodes.append(helper.make_node("Identity", ["INPUT__0"], ["OUTPUT__0"]))

# add past-KV for each layer
for layer in range(2):
    in_key = f"past_key_values.{layer}.key"
    in_value = f"past_key_values.{layer}.value"
    out_key = f"present_key_values.{layer}.key"
    out_value = f"present_key_values.{layer}.value"

    # define their ValueInfoProtos
    inputs.append(helper.make_tensor_value_info(
        in_key,   TensorProto.INT64, kv_shape))
    inputs.append(helper.make_tensor_value_info(
        in_value, TensorProto.INT64, kv_shape))
    outputs.append(helper.make_tensor_value_info(
        out_key,   TensorProto.INT64, kv_shape))
    outputs.append(helper.make_tensor_value_info(
        out_value, TensorProto.INT64, kv_shape))

    # identity nodes to wire them through
    nodes.append(helper.make_node("Identity", [in_key],   [out_key]))
    nodes.append(helper.make_node("Identity", [in_value], [out_value]))

# build & save
graph = helper.make_graph(nodes, "with_past_kv", inputs, outputs)
opset_id = helper.make_operatorsetid("", 21)
model = helper.make_model(graph, ir_version=10, opset_imports=[opset_id]) # version & opset must be compatible with runtime

onnx.save(model, "with_past_kv.onnx")
