import json

from coremltools.converters.mil.mil.ops.defs._op_reqs import register_op
from coremltools.converters.mil.mil import (
    Operation,
    types
)
from coremltools.converters.mil.mil.input_type import (
    InputSpec,
    TensorInputType,
)


def _process_class_name(classname):
    return 'dneprDroid_' + classname


@register_op(is_custom_op=True)
class addmm_op(Operation):

    input_spec = InputSpec(
        p1=TensorInputType(type_domain="T"),
        p2=TensorInputType(type_domain="T"),
    )

    type_domains = {
        "T": (types.fp32,),
    }

    bindings = {
        "class_name": _process_class_name('addmm'),
        "input_order": ["p1", "p2"],
        "parameters": [],
        "description": "addmm  developed by dneprDroid",
    }

    def __init__(self, **kwargs):
        super(addmm_op, self).__init__(**kwargs)

    def type_inference(self):
        assert self.p1.shape[-1] == self.p2.shape[-2]

        out_shape = list(self.p2.shape)
        out_shape[-2] = 1
        return types.tensor(self.p1.dtype, out_shape)


@register_op(is_custom_op=True)
class deform_conv2d_op(Operation):

    input_spec = InputSpec(
        outShape=TensorInputType(const=True, type_domain=types.str),
        offsetShape=TensorInputType(const=True, type_domain=types.str),
        maskShape=TensorInputType(const=True, type_domain=types.str),

        input=TensorInputType(type_domain="T"),

        # weights
        dataOffset=TensorInputType(type_domain="T"),
        dataMask=TensorInputType(type_domain="T"),

        inChannels=TensorInputType(const=True, type_domain=types.int32),
        height=TensorInputType(const=True, type_domain=types.int32),
        width=TensorInputType(const=True, type_domain=types.int32),
        weightH=TensorInputType(const=True, type_domain=types.int32),
        weightW=TensorInputType(const=True, type_domain=types.int32),
        padH=TensorInputType(const=True, type_domain=types.int32),
        padW=TensorInputType(const=True, type_domain=types.int32),
        strideH=TensorInputType(const=True, type_domain=types.int32),
        strideW=TensorInputType(const=True, type_domain=types.int32),

        dilationH=TensorInputType(const=True, type_domain=types.int32),
        dilationW=TensorInputType(const=True, type_domain=types.int32),
        outH=TensorInputType(const=True, type_domain=types.int32),
        outW=TensorInputType(const=True, type_domain=types.int32),
        parallelImgs=TensorInputType(const=True, type_domain=types.int32),
        deformableGroup=TensorInputType(const=True, type_domain=types.int32),

        useMask=TensorInputType(const=True, type_domain=types.bool),
    )

    type_domains = {
        "T": (types.fp32,),
    }

    bindings = {
        "class_name": _process_class_name('deform_conv2d'),
        "input_order": ["input", "dataOffset", "dataMask"],
        "parameters": [
            "outShape",
            "offsetShape",
            "maskShape",

            "inChannels",
            "height",
            "width",
            "weightH",
            "weightW",
            "padH",
            "padW",
            "strideH",
            "strideW",

            "dilationH",
            "dilationW",
            "outH",
            "outW",
            "parallelImgs",
            "deformableGroup",

            "useMask",
        ],
        "description": "deform_conv2d developed by dneprDroid",
    }

    def __init__(self, **kwargs):
        super(deform_conv2d_op, self).__init__(**kwargs)

    def type_inference(self):
        input = self.input
        input_type = input.dtype

        ret_shape_str = self.outShape.val
        ret_shape = json.loads(ret_shape_str)
        return types.tensor(input_type, ret_shape)
