"""
Run in the root dir:

python3 -m converter.demo

"""
import os
import shutil
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops.deform_conv import deform_conv2d

import coremltools

from .. import register_op

class TestModel(nn.Module):

    def __init__(self, sizewh = 16, kwh=3):
        super(TestModel, self).__init__()

        self.weight = torch.rand(1, 3, kwh, kwh)
        print("self.weight: ", self.weight.shape)

    def forward(self, x, offset, mask):
        input = F.interpolate(
            x,
            size=(18, 18),
            mode='bilinear'
        )
        result = deform_conv2d(input, offset, self.weight, mask=mask)
        return result

def rm(path):
    if not os.path.exists(path):
        return
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)


def save_as_json(tensor, filename, output_dir):
    values = tensor.numpy().tolist()
    values_str = json.dumps(values)

    path = os.path.join(output_dir, filename)
    with open(path, 'w') as file:
        file.write(values_str)


def convert(output_dir, filename='test-model'):
    output_path = os.path.join(output_dir, filename)

    sizewh, kwh = 16, 3
    torch_model = TestModel(sizewh, kwh).eval()

    print("generating random input tensor...")
    example_input = torch.rand(1, 3, 600, 600).type(torch.float32)
    offset = torch.rand(1, 2 * kwh * kwh, sizewh, sizewh)
    mask = torch.rand(1, kwh * kwh, sizewh, sizewh)
    example_output = torch_model(example_input, offset, mask)

    print("example output (flatten): ", example_output.flatten())

    save_as_json(example_input, 'example_input.json', output_dir)
    save_as_json(offset, 'example_offset.json', output_dir)
    save_as_json(mask, 'example_mask.json', output_dir)
    save_as_json(example_output, 'example_output.json', output_dir)

    traced_model = torch.jit.trace(torch_model, [example_input, offset, mask])

    input_name = "input"
    offset_name = "dataOffset"
    mask_name = "dataMask"
    output_name = "output"

    mlmodel = coremltools.convert(
        traced_model,
        inputs=[
            coremltools.TensorType(
                name=input_name,
                shape=(example_input.shape)
            ),
            coremltools.TensorType(
                name=offset_name,
                shape=(offset.shape)
            ),
            coremltools.TensorType(
                name=mask_name,
                shape=(mask.shape)
            ),
        ],
        outputs=[
            coremltools.TensorType(
                name=output_name
            ),
        ],
        convert_to="neuralnetwork"
    )
    mlmodel_path = output_path + ".mlmodel"

    out_pb_path = mlmodel_path + ".pb"

    rm(mlmodel_path)
    rm(output_path)
    rm(out_pb_path)

    mlmodel.save(mlmodel_path)

    shutil.copyfile(mlmodel_path, out_pb_path)

    print(f"Saved to {output_dir}")

def main():
    register_op()
    
    current_dir = os.path.dirname(os.path.realpath(__file__))
    out_dir_path = os.path.join(current_dir, "../../DemoApp/generated")
    out_dir_path = os.path.abspath(out_dir_path)
    convert(out_dir_path)    