import Foundation
import AppKit
import CoreML
import DeformConv2dMetal

@main
class App {
    
    static func main() async {
        do {
            try await test()
        } catch {
            print("error: ", error)
            show(message: "Error", description: error.localizedDescription)
            return
        }
        
        let description = "Output tensors from PyTorch and CoreML custom layers are equal. " +
                          "For more info please check the logs."
        show(
            message: "Success",
            description: description
        )
    }
    
    private static func test() async throws {
        guard let modelUrl = Bundle.main.url(forResource: "test-model.mlmodel.pb", withExtension: nil) else {
            fatalError("Can't find ML model")
        }
        let compiledUrl = try await MLModel.compileModel(at: modelUrl)
        defer {
            try? FileManager.default.removeItem(at: compiledUrl)
        }
        
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndGPU
        
        configuration.allowLowPrecisionAccumulationOnGPU = false

        let model = try MLModel(contentsOf: compiledUrl, configuration: configuration)
        
        print("loading example inputs...")
        
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
        
        let output = try await model.prediction(from: input)
            .featureValue(for: "output")?
            .multiArrayValue
        
        guard let output else { fatalError("output is empty") }

        assert(output.dataType == .float32)
        
        let outputArray = output.toNdArray4d()
        
        print("calculated output: ", outputArray)
        let isValid = NdArrayUtil.validate(
            actual: outputArray,
            expected: exampleOutputArray
        )
        assert(isValid)
    }
    
    static private func show(message: String, description: String) {
        let alert = NSAlert()
        alert.messageText = message
        alert.informativeText = description
        alert.runModal()
    }
}
