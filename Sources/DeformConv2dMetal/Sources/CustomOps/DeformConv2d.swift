import Foundation
import CoreML

enum ErrorCommon: Error {
    case metalNotSupported
    case shaderLibNotFound
    case shaderNotFound
}

enum DeformConv2dError: Error {
    case cpuNotImplemented
    case encoderInvalid
    
    case invalidLayerParams
    case paramsDecode
}

@objc(dneprDroid_deform_conv2d)
final class DeformConv2d: NSObject, MLCustomLayer {
    
    let device: MTLDevice
    
    let outShape: [NSNumber]
    let params: DeformConv2dParams

    var gpuParams: DeformConv2dParams.GPUParams
    
    let pipelineState: MTLComputePipelineState
    
    var offsetWeights: MTLTexture?
    var maskWeights: MTLTexture?

    required init(parameters: [String : Any]) throws {
        guard
            let params = try? DeformConv2dParams.decode(from: parameters)
        else {
            throw DeformConv2dError.invalidLayerParams
        }
        self.params = params
        self.gpuParams = params.gpuParams()
        
        self.outShape = params.outShape.shape.map { NSNumber(value: $0) }
        
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw ErrorCommon.metalNotSupported
        }
        self.device = device
        let function = try device.makeFunction(name: "dneprDroid::deform_conv2d")
        pipelineState = try device.makeComputePipelineState(function: function)
        super.init()
    }
    
    func setWeightData(_ weights: [Data]) throws {
        self.offsetWeights = TextureFactory.createTexture2DArray(
            device: device,
            from: weights[0],
            shape: params.offsetShape.shape
        )
        self.maskWeights = TextureFactory.createTexture2DArray(
            device: device,
            from: weights[1],
            shape: params.maskShape.shape
        )
    }
    
    func outputShapes(forInputShapes inputShapes: [[NSNumber]]) throws -> [[NSNumber]] {
        return [outShape]
    }
    
    func evaluate(inputs: [MLMultiArray], outputs: [MLMultiArray]) throws {
        fatalError()
    }
    
    func encode(commandBuffer: MTLCommandBuffer, inputs: [MTLTexture], outputs: [MTLTexture]) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            fatalError()
        }
        let output = outputs[0]

        assert(output.pixelFormat == .rgba16Float)
        assert(output.pixelFormat == inputs[0].pixelFormat)
        assert(outputs.count == 1)
        
        guard let offsetWeights, let maskWeights else {
            fatalError()
        }
        encoder.setTexture(inputs[0], index: 0)
        encoder.setTexture(offsetWeights, index: 1)
        encoder.setTexture(maskWeights, index: 2)
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
}
