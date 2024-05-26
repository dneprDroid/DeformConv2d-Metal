import Foundation
import Metal

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
        let url = rootUrl.appendingPathComponent("SupportFiles")
            .appendingPathComponent("DeformConv2d-Metal-\(osName).metallib")
        
        let library = try self.makeLibrary(URL: url)
        
        guard let function = library.makeFunction(name: name) else {
            throw ErrorCommon.shaderNotFound
        }
        return function
    }
}
