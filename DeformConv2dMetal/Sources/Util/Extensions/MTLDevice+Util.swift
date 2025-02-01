import Foundation
import Metal

extension MTLDevice {
    func makeFunction(name: String) throws -> MTLFunction {
        let library = try self.makeDefaultLibrary(bundle: .module)
        guard let function = library.makeFunction(name: name) else {
            throw ErrorCommon.shaderNotFound
        }
        return function
    }
}
