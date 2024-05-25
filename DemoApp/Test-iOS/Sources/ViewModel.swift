import Foundation
import SwiftUI
import CoreML
import DeformConv2dMetal

final class ViewModel: ObservableObject {
    
    @Published var state: State = .initial

    func test() async throws {
        await updateState(.loadingModel)
        
        guard let modelUrl = Bundle.main.url(forResource: "test-model.mlmodel.pb", withExtension: nil) else {
            fatalError()
        }
        let compiledUrl = try MLModel.compileModel(at: modelUrl)
        defer {
            try? FileManager.default.removeItem(at: compiledUrl)
        }
        
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndGPU
        
        configuration.allowLowPrecisionAccumulationOnGPU = true

        let model = try MLModel(contentsOf: compiledUrl, configuration: configuration)
        
        print("loading example inputs/outputs from JSON files...")
        
        await updateState(.loadingExampleTensors)

        let (exampleInputTensor, _) = try NdArrayUtil.readTensor(resource: "example_input.json", type: NdArray4d.self)
        let (_, exampleOutputArray) = try NdArrayUtil.readTensor(resource: "example_output.json", type: NdArray4d.self)

        let input = Input(input: exampleInputTensor)
        
        print("loaded")
        
        await updateState(.runningModel)
        
        let output = try model.prediction(from: input)
            .featureValue(for: "output")?
            .multiArrayValue
        
        await updateState(.validation)

        guard let output else { fatalError() }

        assert(output.dataType == .float32)
        
        let outputArray = output.toNdArray4d()
        let flattenArray = output.toFlattenArray(for: Float32.self)
        print("Received output (flatten tensor): ", flattenArray)
        
        let isOk = NdArrayUtil.validate(
            actual: outputArray,
            expected: exampleOutputArray
        )
        await updateState(.completed(ok: isOk))
    }
    
    @MainActor private func updateState(_ newState: State) {
        self.state = newState
    }
}

extension ViewModel {
    
    final class Input: MLFeatureProvider {
        
        var featureNames: Set<String> = ["input"]
        
        private let input: MLMultiArray

        init(input: MLMultiArray) {
            self.input = input
        }
        
        func featureValue(for featureName: String) -> MLFeatureValue? {
            if featureName == "input" {
                return MLFeatureValue(multiArray: input)
            }
            return .none
        }
    }
    
    enum State {
        case initial
        case loadingModel
        case loadingExampleTensors
        case runningModel
        case validation
        case completed(ok: Bool)
    }
}
