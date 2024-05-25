import Foundation
import CoreML

typealias NdArray1d = [Float32]
typealias NdArray2d = [[Float32]]
typealias NdArray3d = [[[Float32]]]
typealias NdArray4d = [[[[Float32]]]]

enum NdArrayError: Error {
    case resNotFound
}

protocol NdArray: Decodable {
    var shape: [Int] { get }
    
    subscript(ndIndex: [Int]) -> Float32 { get set }
    
    func forEach(_ body: (_ ndIndex: [Int], _ value: Float32) -> Void)
}

extension NdArray4d: NdArray {
    
    var shape: [Int] {
        [
            self.count,
            self.first?.count ?? 0,
            self.first?.first?.count ?? 0,
            self.first?.first?.first?.count ?? 0,
        ]
    }
    
    subscript(ndIndex: [Int]) -> Float32 {
        get {
            validateIndex(ndIndex)
            
            return self[ndIndex[0]][ndIndex[1]][ndIndex[2]][ndIndex[3]]
        }
        set {
            validateIndex(ndIndex)

            self[ndIndex[0]][ndIndex[1]][ndIndex[2]][ndIndex[3]] = newValue
        }
    }
    
    func forEach(_ body: (_ ndIndex: [Int], _ value: Float32) -> Void) {
        let shape = self.shape
        
        for n in 0..<shape[0] {
            for c in 0..<shape[1] {
                for x in 0..<shape[2] {
                    for y in 0..<shape[3] {
                        let ndIndex = [n, c, x, y]
                        let value = self[n][c][x][y]
                        
                        body(ndIndex, value)
                    }
                }
            }
        }
    }
    
    private func validateIndex(_ ndIndex: [Int]) {
        guard ndIndex.count == 4 else { fatalError("Invalid index: \(ndIndex)") }
    }
}
