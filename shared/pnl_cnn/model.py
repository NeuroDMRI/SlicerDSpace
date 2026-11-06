# Modified version of PNLBrainMask-Torch @ 8d201a5 (Brigham and Women's Hospital licence, see shared/pnl_cnn/LICENSE.txt); modified by SlicerDSpace: architecture() reads data/CompNetBasicModel.json by a path next to this file instead of importlib.resources; import lines changed to match.
"""Exact inference graph of the pinned Keras CompNet (NHWC -> NCHW).

Modified implementation of PNL's released model; see data/PNL-LICENSE.txt.
All three branches and their original layer names are retained for auditing.
"""

import json
from pathlib import Path

import torch
from torch import nn


def architecture():
    return json.loads((Path(__file__).with_name("data") / "CompNetBasicModel.json").read_text(encoding="utf-8"))


class CompNet(nn.Module):
    def __init__(self):
        super().__init__()
        config = architecture()["config"]
        self.spec = config["layers"]
        self.output_names = [item[0] for item in config["output_layers"]]
        self.ops = nn.ModuleDict()
        self.parents = {}
        self.activations = {}
        channels = {}
        for layer in self.spec:
            name, kind, c = layer["name"], layer["class_name"], layer["config"]
            parents = [item[0] for item in layer["inbound_nodes"][0]] if layer["inbound_nodes"] else []
            self.parents[name] = parents
            if kind == "InputLayer":
                channels[name] = 1
                continue
            incoming = channels[parents[0]]
            channels[name] = incoming
            if kind in ("Conv2D", "Conv2DTranspose"):
                kernel, stride = tuple(c["kernel_size"]), tuple(c["strides"])
                if kind == "Conv2D":
                    if c["strides"] != [1, 1] or c["dilation_rate"] != [1, 1]:
                        raise ValueError("Unreviewed convolution configuration")
                    padding = tuple(k // 2 for k in kernel) if c["padding"] == "same" else 0
                    op = nn.Conv2d(incoming, c["filters"], kernel, stride, padding, bias=c["use_bias"])
                else:
                    # Released transpose kernels equal stride: Keras SAME needs no crop.
                    if kernel != (2, 2) or stride != (2, 2) or c["padding"] != "same":
                        raise ValueError("Unreviewed transposed convolution")
                    op = nn.ConvTranspose2d(incoming, c["filters"], kernel, stride, bias=c["use_bias"])
                self.ops[name] = op
                channels[name] = c["filters"]
                self.activations[name] = c["activation"]
                if c["activation"] not in ("linear", "relu", "sigmoid"):
                    raise ValueError("Unreviewed activation")
            elif kind == "BatchNormalization":
                if c["axis"] != -1 or not c["center"] or not c["scale"]:
                    raise ValueError("Unreviewed batch normalization")
                self.ops[name] = nn.BatchNorm2d(incoming, eps=c["epsilon"], momentum=1 - c["momentum"])
            elif kind == "MaxPooling2D":
                if c["padding"] != "valid":
                    raise ValueError("Unreviewed pooling")
                self.ops[name] = nn.MaxPool2d(tuple(c["pool_size"]), tuple(c["strides"]))
            elif kind == "Concatenate":
                if c["axis"] not in (-1, 3):
                    raise ValueError("Unreviewed concatenation")
                channels[name] = sum(channels[parent] for parent in parents)
            elif kind not in ("Add", "Dropout"):
                raise ValueError("Unsupported layer: " + kind)
        self.eval()

    def forward(self, x, *, capture=(), segmentation_only=False):
        if self.training:
            raise RuntimeError("This converted model supports inference only")
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, 256, 256):
            raise ValueError("CompNet input must be NCHW (N,1,256,256)")
        targets = {"final_op"} if segmentation_only else set(self.output_names)
        targets.update(capture)
        needed = set()

        def require(name):
            if name not in self.parents:
                raise ValueError("Unknown capture layer: " + name)
            if name not in needed:
                needed.add(name)
                for parent in self.parents[name]:
                    require(parent)

        for name in targets:
            require(name)
        uses = {name: 0 for name in needed}
        for name in needed:
            for parent in self.parents[name]:
                uses[parent] += 1
        values, captured = {}, {}
        for layer in self.spec:
            name, kind = layer["name"], layer["class_name"]
            if name not in needed:
                continue
            parents = self.parents[name]
            inputs = [values[parent] for parent in parents]
            if kind == "InputLayer":
                value = x
            elif kind == "Concatenate":
                value = torch.cat(inputs, dim=1)
            elif kind == "Add":
                value = inputs[0] + inputs[1]
            elif kind == "Dropout":
                value = inputs[0]
            else:
                value = self.ops[name](inputs[0])
                activation = self.activations.get(name)
                if activation == "relu":
                    value = torch.relu(value)
                elif activation == "sigmoid":
                    value = torch.sigmoid(value)
            values[name] = value
            if name in capture:
                captured[name] = value
            for parent in parents:
                uses[parent] -= 1
                if uses[parent] == 0 and parent not in targets:
                    del values[parent]
        result = values["final_op"] if segmentation_only else tuple(values[name] for name in self.output_names)
        return (result, captured) if capture else result
