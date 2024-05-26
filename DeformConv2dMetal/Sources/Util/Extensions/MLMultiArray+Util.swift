import Foundation
import CoreML

extension MLMultiArray {
    
    public func toFlattenArray<T: Numeric>(for type: T.Type  = T.self) -> [T] {
        let ptr = self.dataPointer.assumingMemoryBound(to: T.self)
        let array = [T](UnsafeBufferPointer(start: ptr, count: self.count))
        return array
    }
}
