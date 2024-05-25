import json

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import (
    Builder as mb,
)
from coremltools.converters.mil.frontend.torch.ops import (
    _get_inputs as mil_get_inputs
)
from coremltools.converters.mil import (
    register_torch_op
)


def register_op():
    register_torch_op(
        _func=torchvision_deform_conv2d,
        torch_alias=["torchvision::deform_conv2d"],
        override=True
    )

# @register_torch_op(torch_alias=["torchvision::deform_conv2d"], override=True)
def torchvision_deform_conv2d(context, node):
    inputs = mil_get_inputs(context, node, expected=14)

    input = inputs[0]
    weight = inputs[1]
    offset = inputs[2]
    mask = inputs[3]
    bias = inputs[4]

    stride_h = inputs[5].val
    stride_w = inputs[6].val
    pad_h = inputs[7].val
    pad_w = inputs[8].val
    dil_h = inputs[9].val
    dil_w = inputs[10].val
    n_weight_grps = inputs[11].val
    n_offset_grps = inputs[12].val
    use_mask = inputs[13].val

    batch_sz = input.shape[0]
    n_in_channels = input.shape[1]
    in_h = input.shape[2]
    in_w = input.shape[3]

    n_parallel_imgs = batch_sz

    out_channels = weight.shape[0]
    weight_h = weight.shape[2]
    weight_w = weight.shape[3]

    ker_h = dil_h * (weight_h - 1) + 1
    ker_w = dil_w * (weight_w - 1) + 1
    out_h = ((in_h + 2 * pad_h - ker_h) / stride_h) + 1
    out_w = ((in_w + 2 * pad_w - ker_w) / stride_w) + 1

    out_shape_final = [
        batch_sz, out_channels, out_h, out_w
    ]
    out_shape_final = [int(_o) for _o in out_shape_final]

    class _State:
        _view_index = 0
        _flatten_index = 0

    _state = _State()

    def _view(x=None, shape=None, name=None):
        op_name = name if name != None else node.name + \
            ('___view_%i' % _state._view_index)
        # print("view: ", op_name, shape)
        res = mb.reshape(x=x, shape=[int(o) for o in shape], name=op_name)
        _state._view_index += 1
        context.add(res)
        return res

    input = _view(
        x=input,
        # shape=[batch_sz / n_parallel_imgs, n_parallel_imgs, n_in_channels, in_h, in_w]
        shape=[n_parallel_imgs, n_in_channels, in_h, in_w]
    )
    offset = _view(
        x=offset,
        shape=[
            batch_sz / n_parallel_imgs,
            n_parallel_imgs,
            n_offset_grps * 2 * weight_h * weight_w,
            out_h,
            out_w
        ]
    )
    mask = _view(
        x=mask,
        shape=[
            batch_sz / n_parallel_imgs,
            n_parallel_imgs,
            n_offset_grps * weight_h * weight_w,
            out_h,
            out_w
        ]
    )

    def _zeros(shape=None, name=None):
        assert name != None

        return mb.fill(
            value=0.0,
            shape=[int(o) for o in shape],
            name=name
        )

    out_buf = _zeros(
        shape=[
            batch_sz / n_parallel_imgs,
            out_channels,
            n_parallel_imgs * out_h,
            out_w
        ],
        name=node.name+'_out_buf_first'
    )

    out_buf = _view(
        x=out_buf,
        shape=[
            out_buf.shape[0],
            n_weight_grps,
            out_buf.shape[1] / n_weight_grps,
            out_buf.shape[2],
            out_buf.shape[3]
        ]
    )

    weight = _view(
        x=weight,
        shape=[
            n_weight_grps,
            weight.shape[0] / n_weight_grps,
            weight.shape[1],
            weight.shape[2],
            weight.shape[3]
        ])

    # columns_shape = [
    #       n_in_channels * weight_h * weight_w,
    #       n_parallel_imgs * out_h * out_w
    # ]
    columns_shape = [
        n_in_channels * weight_h * weight_w,
        #   n_parallel_imgs * out_h * out_w
        out_w,
        out_h,
    ]
    columns_shape = [int(_o) for _o in columns_shape]

    def _shapeToStr(shape=None):
        shape_int = [int(_o) for _o in shape]
        return json.dumps(list(shape_int))

    columns_shape_str = _shapeToStr(shape=[1, 1] + list(columns_shape))

    # for b in range(batch_sz / n_parallel_imgs):

    __weight_g = _view(x=weight, shape=weight.shape[1:])
    __weight_g = _view(x=weight, shape=[1, 1, 1, __weight_g.val.size])

    weight_g = __weight_g

    columns = mb.deform_conv2d_op(
        outShape=columns_shape_str,
        offsetShape=_shapeToStr(shape=offset.shape),
        maskShape=_shapeToStr(shape=mask.shape),

        input=input,

        # Weights
        dataOffset=offset,
        dataMask=mask,

        inChannels=n_in_channels,
        height=in_h,
        width=in_w,
        weightH=weight_h,
        weightW=weight_w,
        padH=pad_h,
        padW=pad_w,
        strideH=stride_h,
        strideW=stride_w,
        dilationH=dil_h,
        dilationW=dil_w,
        outH=int(out_h),
        outW=int(out_w),
        parallelImgs=n_parallel_imgs,
        deformableGroup=n_offset_grps,
        useMask=use_mask,

        name=node.name+'_columns'
    )
    columns = _view(
        x=columns,
        shape=[
            n_weight_grps,
            n_in_channels * weight_h * weight_w / n_weight_grps,
            n_parallel_imgs * out_h * out_w,
        ]
    )

    def _flatten(x, startDim):
        lastDim = 1
        for d in x.shape[startDim:]:
            lastDim *= d
        new_shape = x.shape[:startDim]
        new_shape += (lastDim,)
        res = mb.reshape(
            x=x,
            shape=new_shape,
            name=node.name + ("__flatten__%i" % _state._flatten_index)
        )
        context.add(res)

        _state._flatten_index += 1
        return res

    columns_g = _view(x=columns, shape=columns.shape[1:])

    def _as_img_tensor2(x):
        assert len(x.shape) == 2
        return _view(x=x, shape=([1, 1] + list(x.shape)))

    out_buf_addmm = mb.addmm_op(
        p1=_as_img_tensor2(_flatten(weight_g, 1)),
        p2=_as_img_tensor2(columns_g)
    )
    out_buf = _view(x=out_buf_addmm, shape=out_buf.shape)

    columns = _view(
        x=columns,
        shape=[columns.shape[0] * columns.shape[1], columns.shape[2]]
    )

    # assert batch_sz / n_parallel_imgs == 1

    out_buf = _view(
        x=out_buf,
        shape=[
            # batch_sz / n_parallel_imgs,
            out_channels,
            n_parallel_imgs,
            out_h,
            out_w
        ]
    )

    result_shape = [batch_sz, out_channels, out_h, out_w]
    result_shape = [int(_o) for _o in result_shape]

    out_buf = _view(
        x=out_buf,
        shape=result_shape,
    )
    bias = _view(x=bias, shape=[1, out_channels, 1, 1])
    out_buf = mb.add(
        x=out_buf,
        y=bias,
        name=node.name
    )
    context.add(out_buf)
