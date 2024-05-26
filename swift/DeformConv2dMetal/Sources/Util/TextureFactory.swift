import Foundation
import Accelerate
import Metal

#if arch(arm64)
// for iOS/macOS arm64
typealias _Float16 = Float16
#else
// for macOS x86_64
typealias _Float16 = UInt16
#endif

enum TextureFactoryError: Error {
    case failedToAllocate
    case dataTypeNotSupported(bytesPerComponent: Int)
}

enum TextureFactory {
        
    static func createTexture2DArray(
        device: MTLDevice,
        from data: Data,
        shape: [Int]
    ) throws -> MTLTexture {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r16Float,
            width: shape[shape.count - 2],
            height: shape[shape.count - 1],
            mipmapped: false
        )
        descriptor.usage = .shaderRead
        descriptor.textureType = .type2DArray
        descriptor.depth = 1
        descriptor.arrayLength = shape[shape.count - 3]
        
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw ErrorCommon.textureAllocateFailed
        }
        try fill(texture: texture, from: data, shape: shape)
        return texture
    }
    
    static func fill(texture: MTLTexture, from data: Data, shape: [Int]) throws {
        let tensorSize = shape.tensorSize()
        
        let dataBytesPerComponent = data.count / tensorSize
        
        switch (dataBytesPerComponent, texture.pixelFormat) {
        case (MemoryLayout<Float32>.stride, .rgba16Float): // float32 -> float16
            try fill(
                texture: texture,
                from: data,
                shape: shape,
                channels: 4, // rgba16Float
                convert: Self.float32to16
            )
        case (MemoryLayout<Float32>.stride, .r16Float): // float32 -> float16
            try fill(
                texture: texture,
                from: data,
                shape: shape,
                channels: 1,  // r16Float
                convert: Self.float32to16
            )
        default:
            throw ErrorCommon.textureDataConversionNotSupported(
                bytesPerComponent: dataBytesPerComponent,
                pixelFormat: texture.pixelFormat
            )
        }
    }
    
    private static func fill<ScalarSrc: Numeric, ScalarDst: Numeric>(
        texture: MTLTexture,
        from data: Data,
        shape: [Int],
        channels: Int,
        convert: (_ ptr: UnsafeMutablePointer<ScalarSrc>, _ count: Int) throws -> [ScalarDst]
    ) throws {
        let tensorSize = shape.tensorSize()
        
        var src = data.withUnsafeBytes { [ScalarSrc](UnsafeBufferPointer(start: $0, count: tensorSize)) }
        let dst: [ScalarDst] = try convert(&src, src.count)
                
        let bytesPerComponent = channels * MemoryLayout<ScalarDst>.stride
        let bytesPerRow = texture.width * bytesPerComponent
        let bytesPerImage = bytesPerRow * texture.height
                    
        dst.withUnsafeBytes { ptr in
            guard let rawPtr = ptr.baseAddress else { return }
            
            for sliceIndex in 0..<texture.arrayLength {
                let offset = sliceIndex * channels * texture.width * texture.height * texture.depth
                
                let slicePtr = rawPtr.advanced(by: offset * MemoryLayout<ScalarDst>.stride)
                
                texture.replace(
                    region: texture.region,
                    mipmapLevel: 0,
                    slice: sliceIndex,
                    withBytes: slicePtr,
                    bytesPerRow: bytesPerRow,
                    bytesPerImage: bytesPerImage
                )
            }
        }
    }
    
    private static func float32to16(
        input: UnsafeMutablePointer<Float32>,
        count: Int
    ) throws -> [_Float16] {
        
        var output = [_Float16](repeating: 0, count: count)
        
        try output.withUnsafeMutableBytes { ptr in
            guard let rawPtr = ptr.baseAddress else { return }
            
            var bufferFloat32 = vImage_Buffer(data: input,   height: 1, width: UInt(count), rowBytes: count * 4)
            var bufferFloat16 = vImage_Buffer(data: rawPtr, height: 1, width: UInt(count), rowBytes: count * 2)
            
            let status = vImageConvert_PlanarFtoPlanar16F(&bufferFloat32, &bufferFloat16, 0)
            if status != kvImageNoError {
                throw ErrorCommon.dataConversionUnsupported(
                    srcBytesPerComponent: 4,
                    dstBytesPerComponent: 2
                )
            }
        }
        return output
    }
}
