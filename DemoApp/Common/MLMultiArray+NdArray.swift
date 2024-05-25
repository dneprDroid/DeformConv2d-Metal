import Foundation
import CoreML

extension MLMultiArray {
    
    func toNdArray4d() -> NdArray4d {
        let shape = self.shape.map { $0.intValue }
        assert(shape.count == 4)

        var out = NdArray4d(
            repeating: NdArray3d(
                repeating: NdArray2d(
                    repeating: NdArray1d(repeating: 0, count: shape[3]),
                    count: shape[2]
                ),
                count: shape[1]
            ),
            count: shape[0]
        )
        out.forEach { ndIndex, _ in
            let index = ndIndex.map { NSNumber(value: $0) }
            out[ndIndex] = self[index].floatValue
        }
        return out
    }
}
