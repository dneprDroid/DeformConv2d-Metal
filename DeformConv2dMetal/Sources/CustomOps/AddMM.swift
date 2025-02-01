import Foundation
import CoreML

// it'll be loaded by CoreML engine, don't change the objc class name
@objc(dneprDroid_addmm)
final class AddMM: NSObject, MLCustomLayer {
    let pipelineState: MTLComputePipelineState

    init(parameters: [String : Any]) throws {  
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw ErrorCommon.metalNotSupported
        }
        let function = try device.makeFunction(name: "dneprDroid::addmm")
        pipelineState = try device.makeComputePipelineState(function: function)
        
        super.init()
    }
    
    func setWeightData(_ weights: [Data]) throws { }
    
    func outputShapes(forInputShapes inputShapes: [[NSNumber]]) throws -> [[NSNumber]] {        
        var outShape = inputShapes[1].map { $0.intValue }
        outShape[outShape.count - 2] = 1
        return [outShape.map { NSNumber(value: $0) }]
    }
    
    func encode(commandBuffer: any MTLCommandBuffer, inputs: [any MTLTexture], outputs: [any MTLTexture]) throws {

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw ErrorCommon.encoderInvalid
        }
        let p0 = inputs[0]
        let p1 = inputs[1]
        
        let output = outputs[0]

        encoder.setTexture(p0, index: 0)
        encoder.setTexture(p1, index: 1)
        encoder.setTexture(output, index: 2)
        
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
