import Foundation

extension Array where Element == Int {
    func tensorSize() -> Int {
        return self.reduce(1, { $0 * $1 })
    }
}
