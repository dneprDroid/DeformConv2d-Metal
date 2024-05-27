# DeformConv2d-Metal

CoreML custom layer (GPU-accelerated) and converter for [`torchvision.ops.deform_conv2d`](https://pytorch.org/vision/main/generated/torchvision.ops.deform_conv2d.html).


## Demo 

Convert [the demo ml-model](converter/demo/convert.py) with `torchvision.ops.deform_conv2d` operation to CoreML format:

``` bash 
# Run in the root dir:

python3 -m converter.demo
```
It'll save the ml-model and example input/output tensors to the `DemoApp/generated` directory 
so the demo app can validate the CoreML output results and compare them with the PyTorch output.

The converted ML-model contains custom layers:


![Screenshot 2024-05-27 at 12 55 43](https://github.com/dneprDroid/DeformConv2d-Metal/assets/13742733/d9044a31-e598-4072-bab9-53c2e7da20a4)



### iOS and macOS demo apps
Open `DemoApp/DemoApp.xcodeproj` in Xcode and run the demo app.

The `Test-iOS` target contains the demo for iOS.

The `Test-macOS` target contains the demo for macOS.

In `MLModelTestWorker` it loads the generated CoreML model and the example input tensor from the `DemoApp/generated` directory 
and compares the calculated CoreML output tensor with the PyTorch example output tensor from the `DemoApp/generated` directory. 

### Custom layers
Custom layers are located in [`DeformConv2dMetal/Sources/CustomOps`](DeformConv2dMetal/Sources/CustomOps).


## Use in your project


Install this pip package and import it in your convertor script:

```bash 

pip install git+https://github.com/dneprDroid/DeformConv2d-Metal.git

```

```python
import DeformConv2dConvert

...

# register op so CoreML Tools can find the convetor function  
DeformConv2dConvert.register_op()

# and convert your model...
...

```

**NOTE**: In the `coremltools.convert` function you need to set `convert_to="neuralnetwork"`:

```python
mlmodel = coremltools.convert(
    traced_model,
    inputs=...,
    outputs=...,
    convert_to="neuralnetwork"
)
```
#### iOS/macOS app

In your iOS/macOS app add the SwiftPM package from this repository:

```
https://github.com/dneprDroid/DeformConv2d-Metal.git
```
CoreML should find and load the custom layers from the `DeformConv2dMetal` module automatically, so you don't need to do anything.  





