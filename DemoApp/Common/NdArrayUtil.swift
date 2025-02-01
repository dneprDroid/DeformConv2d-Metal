import Foundation
import CoreML

enum NdArrayUtil {

    static func readTensor<NdArrayType: NdArray>(
        resource: String,
        type: NdArrayType.Type = NdArrayType.self
    ) throws -> (MLMultiArray, NdArrayType) {
        guard let url = Bundle.main.url(forResource: resource, withExtension: "json") else {
            throw NdArrayError.resNotFound
        }
        return try readTensor(url: url)
    }
    
    static func readTensor<NdArrayType: NdArray>(
        url: URL,
        type: NdArrayType.Type = NdArrayType.self
    ) throws -> (MLMultiArray, NdArrayType) {
        let fileData = try Data(contentsOf: url)
        let ndarray = try JSONDecoder().decode(NdArrayType.self, from: fileData)
        
        let shape = ndarray.shape
        
        let nsshape = shape.map { NSNumber(value: $0) }
        
        let mlarray = try MLMultiArray(shape: nsshape, dataType: .float32)
        
        ndarray.forEach { ndIndex, value in
            let index = ndIndex.map { NSNumber(value: $0) }
            mlarray[index] = NSNumber(value: value)
        }
        return (mlarray, ndarray)
    }
    
    static func validate<NdArrayType: NdArray>(
        actual: NdArrayType,
        expected: NdArrayType,
        precision: Float32 = 0.009
    ) -> Bool {
        print("Checking....")
        
        var maxDiff: Float32 = 0
        
        actual.forEach { ndIndex, actualValue in
            let expectedValue = expected[ndIndex]
            
            let diff = abs(actualValue - expectedValue)
//            assert(diff < precision)
            maxDiff = max(maxDiff, diff)
        }
        let isOk = maxDiff < precision
        if isOk {
            print("Validation: successful")
        } else {
            print("maxDiff=\(maxDiff) > precision")
            print("Validation: failed (CoreML and PyTorch output tesnors aren't equal)")
        }
        return isOk
    }
}
