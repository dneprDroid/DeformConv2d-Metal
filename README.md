# DeformConv2d-Metal

CoreML custom layer (GPU-accelerated) and converter for [`torchvision.ops.deform_conv2d`](https://pytorch.org/vision/main/generated/torchvision.ops.deform_conv2d.html).


## Demo 

Convert a demo ml-model to CoreML format:


``` bash 
# Run in the root dir:

python3 -m converter.demo
```
It'll save the ml-model and example input/output tensors to the `DemoApp/generated` directory 
so the demo app can validate the CoreML output results and compare them with the PyTorch output.

### iOS and macOS demo app
Open `DemoApp/DemoApp.xcodeproj` in Xcode and run the demo app.

The `Test-iOS` target contains the demo for iOS.

The `Test-macOS` target contains the demo for macOS.

In `MLModelTestWorker` it loads the generated CoreML model and the example input tensor from the `DemoApp/generated` directory 
and compares the calculated CoreML output tensor with the PyTorch example output tensor from the `DemoApp/generated` directory. 


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





