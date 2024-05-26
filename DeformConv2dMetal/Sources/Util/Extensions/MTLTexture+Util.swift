import Foundation
import CoreML
import Accelerate
import Metal

extension MTLTexture {
    
    public var region: MTLRegion {
        .init(
            origin: .init(x: 0, y: 0, z: 0),
            size: self.size
        )
    }
    
    public var size: MTLSize {
        .init(
            width: self.width,
            height: self.height,
            depth: self.depth
        )
    }
    
    public func toFlattenArray<T: Numeric>(
        bytesPerComponent: Int,
        for type: T.Type  = T.self
    ) -> [T] {
        let bytesPerRow = width * bytesPerComponent
        
        var values = [T](repeating: 0, count: bytesPerRow * height * depth)
        
        values.withUnsafeMutableBytes { ptr in
            guard let rawPtr = ptr.baseAddress else { return }
            
            self.getBytes(
                rawPtr,
                bytesPerRow: bytesPerRow,
                from: self.region,
                mipmapLevel: 0
            )
        }
        return values
    }
}
