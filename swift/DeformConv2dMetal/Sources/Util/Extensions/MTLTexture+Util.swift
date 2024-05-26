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

extension MLMultiArray {
    
    public func toFlattenArray<T: Numeric>(for type: T.Type  = T.self) -> [T] {
        let ptr = self.dataPointer.assumingMemoryBound(to: T.self)
        let array = [T](UnsafeBufferPointer(start: ptr, count: self.count))
        return array
    }
}

extension Array where Element == Int {
    func tensorSize() -> Int {
        return self.reduce(1, { $0 * $1 })
    }
}

extension MTLDevice {
    func makeFunction(name: String) throws -> MTLFunction {
        guard
            let rootUrl = Bundle.module.resourceURL
        else { throw ErrorCommon.missingBundle }
        
        #if os(iOS)
            let osName = "iOS"
        #elseif os(macOS)
            let osName = "macOS"
        #else
            #error("OS isn't supported")
        #endif
        let url = rootUrl.appendingPathComponent("Metal")
            .appendingPathComponent("DeformConv2d-Metal-\(osName).metallib")
        
        let library = try self.makeLibrary(URL: url)
        
        guard let function = library.makeFunction(name: name) else {
            throw ErrorCommon.shaderNotFound
        }
        return function
    }
}
