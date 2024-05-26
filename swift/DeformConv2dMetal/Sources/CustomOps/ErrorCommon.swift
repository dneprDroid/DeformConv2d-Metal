import Foundation
import Metal

public enum ErrorCommon: Error {
    case metalNotSupported
    case shaderLibNotFound
    case shaderNotFound
    case cpuNotImplemented
    case encoderInvalid
    case invalidLayerParams
    case missingWeights
    case missingBundle
    case textureAllocateFailed
    case textureDataConversionNotSupported(
        bytesPerComponent: Int,
        pixelFormat: MTLPixelFormat
    )
    case dataConversionUnsupported(
        srcBytesPerComponent: Int,
        dstBytesPerComponent: Int
    )
    case pixelFormatNotSupported(MTLPixelFormat)
}
