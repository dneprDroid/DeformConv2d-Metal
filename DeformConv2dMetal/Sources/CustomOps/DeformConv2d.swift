import Foundation
import CoreML

// it'll be loaded by CoreML engine, don't change the objc class name
@objc(dneprDroid_deform_conv2d)
final class DeformConv2d: NSObject, MLCustomLayer {
    
    let device: MTLDevice
    
    let outShape: [NSNumber]
    let params: DeformConv2dParams

    var gpuParams: DeformConv2dParams.GPUParams
    
    let pipelineState: MTLComputePipelineState

    required init(parameters: [String : Any]) throws {
        guard
            let params = try? DeformConv2dParams.decode(from: parameters)
        else {
            throw ErrorCommon.invalidLayerParams
        }
        self.params = params
        self.gpuParams = params.gpuParams()
        
        self.outShape = params.outShape.shape.map { NSNumber(value: $0) }
        
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw ErrorCommon.metalNotSupported
        }
        self.device = device
        
        let library = try device.moduleLibrary()
        guard
            let function = library.makeFunction(name: "dneprDroid::deform_conv2d")
        else { throw ErrorCommon.shaderNotFound }
        
        pipelineState = try device.makeComputePipelineState(function: function)
        super.init()
    }
    
    func setWeightData(_ weights: [Data]) throws {}
    
    func outputShapes(forInputShapes inputShapes: [[NSNumber]]) throws -> [[NSNumber]] {
        return [outShape]
    }
    
    func encode(commandBuffer: MTLCommandBuffer, inputs: [MTLTexture], outputs: [MTLTexture]) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw ErrorCommon.encoderInvalid
        }
        let input = inputs[0]
        let offset = inputs[1]
        let mask = inputs[2]
        let output = outputs[0]

        if output.pixelFormat != .rgba16Float {
            throw ErrorCommon.pixelFormatNotSupported(output.pixelFormat)
        }        
        
        encoder.setTexture(input, index: 0)
        encoder.setTexture(offset, index: 1)
        encoder.setTexture(mask, index: 2)
        encoder.setBytes(
            &gpuParams,
            length: MemoryLayout<DeformConv2dParams.GPUParams>.stride,
            index: 3
        )
        encoder.setTexture(output, index: 4)
        
        let w = pipelineState.threadExecutionWidth
        let h = pipelineState.maxTotalThreadsPerThreadgroup / w
        let threadGroupSize = MTLSize(width: w, height: h, depth: 1)

        let threadGroups = MTLSize(
            width:  (output.width       + threadGroupSize.width  - 1) / threadGroupSize.width,
            height: (output.height      + threadGroupSize.height - 1) / threadGroupSize.height,
            depth:  (output.arrayLength + threadGroupSize.depth  - 1) / threadGroupSize.depth
        )
        encoder.setComputePipelineState(pipelineState)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()
    }
    
    func evaluate(inputs: [MLMultiArray], outputs: [MLMultiArray]) throws {
        throw ErrorCommon.cpuNotImplemented
    }
}

extension MTLDevice {
    func moduleLibrary() throws -> MTLLibrary {
        return try self.makeDefaultLibrary(bundle: .module)
    }
}
