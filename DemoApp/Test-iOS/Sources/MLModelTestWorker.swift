import Foundation
import SwiftUI
import CoreML
import DeformConv2dMetal

enum State {
    case initial
    case loadingModel
    case loadingExampleTensors
    case runningModel
    case validation
    case completed(ok: Bool)
    case error(Error)
}

final class MLModelTestWorker {
    
    var onUpdateState: (State) async -> Void = { _ in }

    func test() async throws {
        await onUpdateState(.loadingModel)
        
        guard let modelUrl = Bundle.main.url(forResource: "test-model.mlmodel.pb", withExtension: nil) else {
            fatalError("Can't find ML model")
        }
        let compiledUrl = try MLModel.compileModel(at: modelUrl)
        defer {
            try? FileManager.default.removeItem(at: compiledUrl)
        }
        
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndGPU
        
        configuration.allowLowPrecisionAccumulationOnGPU = false

        let model = try MLModel(contentsOf: compiledUrl, configuration: configuration)
        
        print("loading example inputs/outputs from JSON files...")
        
        await onUpdateState(.loadingExampleTensors)

        let (exampleInput, _) = try NdArrayUtil.readTensor(resource: "example_input.json", type: NdArray4d.self)
        let (exampleOffset, _) = try NdArrayUtil.readTensor(resource: "example_offset.json", type: NdArray4d.self)
        let (exampleMask, _) = try NdArrayUtil.readTensor(resource: "example_mask.json", type: NdArray4d.self)
        let (_, exampleOutputArray) = try NdArrayUtil.readTensor(resource: "example_output.json", type: NdArray4d.self)
        
        let combinedInputs: [String: Any] = [
            "input": exampleInput,
            "dataOffset": exampleOffset,
            "dataMask": exampleMask
        ]
        let input = try MLDictionaryFeatureProvider(dictionary: combinedInputs)
        
        print("loaded")
        
        await onUpdateState(.runningModel)
        
        let output = try model.prediction(from: input)
            .featureValue(for: "output")?
            .multiArrayValue
        
        await onUpdateState(.validation)

        guard let output else { fatalError("output is empty") }

        assert(output.dataType == .float32)
        
        let outputArray = output.toNdArray4d()
        let flattenArray = output.toFlattenArray(for: Float32.self)
        print("calculated output (flatten tensor): ", flattenArray)
        
        let isOk = NdArrayUtil.validate(
            actual: outputArray,
            expected: exampleOutputArray
        )
        await onUpdateState(.completed(ok: isOk))
    }
}
