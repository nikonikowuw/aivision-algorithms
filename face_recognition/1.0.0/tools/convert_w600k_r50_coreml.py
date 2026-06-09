#!/usr/bin/env python3
"""Convert w600k_r50.onnx to be fully CoreML-compatible:
1. Fuse terminal BatchNormalization into Gemm weights
2. Replace Flatten with Reshape
3. Fix batch=1
"""

import numpy as np
import onnx
from onnx import numpy_helper, helper, shape_inference

src = "models/insightface/w600k_r50.onnx"
dst = "models/insightface/w600k_r50_coreml.onnx"

model = onnx.load(src)
node_map = {n.name: n for n in model.graph.node}
init_map = {i.name: i for i in model.graph.initializer}

gemm_node = node_map["Gemm_128"]
bn_node = node_map["BatchNormalization_129"]
flatten_node = node_map["Flatten_127"]

# --- Fuse BN into Gemm ---
eps = 1e-5
gamma = numpy_helper.to_array(init_map[bn_node.input[1]])
beta = numpy_helper.to_array(init_map[bn_node.input[2]])
mean = numpy_helper.to_array(init_map[bn_node.input[3]])
var = numpy_helper.to_array(init_map[bn_node.input[4]])
scale = gamma / np.sqrt(var + eps)
bias_new = beta - mean * scale

W_old = numpy_helper.to_array(init_map[gemm_node.input[1]])
b_old = numpy_helper.to_array(init_map[gemm_node.input[2]])

# Gemm with transB=1: W is [512, 25088], x @ W^T => scale along output dim
transB = 0
for attr in gemm_node.attribute:
    if attr.name == "transB":
        transB = attr.i

if transB:
    W_new = W_old * scale[:, np.newaxis]
else:
    W_new = W_old * scale[np.newaxis, :]
b_new = b_old * scale + bias_new

W_new_init = numpy_helper.from_array(W_new.astype(np.float32), gemm_node.input[1])
b_new_init = numpy_helper.from_array(b_new.astype(np.float32), gemm_node.input[2])

bn_param_names = set(bn_node.input[1:])
new_initializers = []
for i in model.graph.initializer:
    if i.name == gemm_node.input[1]:
        new_initializers.append(W_new_init)
    elif i.name == gemm_node.input[2]:
        new_initializers.append(b_new_init)
    elif i.name in bn_param_names:
        continue
    else:
        new_initializers.append(i)

# --- Flatten -> Reshape ---
reshape_shape_init = numpy_helper.from_array(
    np.array([1, 25088], dtype=np.int64), "reshape_25088"
)
new_initializers.append(reshape_shape_init)

reshape_node = helper.make_node(
    "Reshape",
    inputs=[flatten_node.input[0], "reshape_25088"],
    outputs=flatten_node.output,
    name="Reshape_127",
)

# New Gemm output directly to original BN output name
transB = 0
for attr in gemm_node.attribute:
    if attr.name == "transB":
        transB = attr.i

gemm_new = helper.make_node(
    "Gemm",
    inputs=[gemm_node.input[0], gemm_node.input[1], gemm_node.input[2]],
    outputs=bn_node.output,
    name=gemm_node.name,
    alpha=1.0,
    beta=1.0,
    transB=transB,
)

new_nodes = []
for n in model.graph.node:
    if n.name == flatten_node.name:
        new_nodes.append(reshape_node)
    elif n.name == gemm_node.name:
        new_nodes.append(gemm_new)
    elif n.name == bn_node.name:
        continue
    else:
        new_nodes.append(n)

# Rebuild model
model = helper.make_model(
    helper.make_graph(
        new_nodes, model.graph.name,
        model.graph.input, model.graph.output, new_initializers,
    ),
    opset_imports=model.opset_import,
)

# Fix batch=1
for vi in list(model.graph.input) + list(model.graph.output):
    shape = vi.type.tensor_type.shape
    if shape.dim:
        first = shape.dim[0]
        if first.dim_param or first.dim_value <= 0:
            first.ClearField("dim_param")
            first.dim_value = 1

model = shape_inference.infer_shapes(model)
onnx.save(model, dst)
print(f"OK {dst}")

# Verify
m2 = onnx.load(dst)
for i in m2.graph.input:
    dims = [d.dim_value for d in i.type.tensor_type.shape.dim]
    print(f"  input {i.name} {dims}")
for o in m2.graph.output:
    dims = [d.dim_value for d in o.type.tensor_type.shape.dim]
    print(f"  output {o.name} {dims}")
ops = {}
for n in m2.graph.node:
    ops[n.op_type] = ops.get(n.op_type, 0) + 1
print(f"  ops {sorted(ops.items())}")
