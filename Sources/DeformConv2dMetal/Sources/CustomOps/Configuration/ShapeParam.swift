import Foundation

struct ShapeParam: Decodable {
    let shape: [Int]
    
    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()

        let shapeStr = try container.decode(String.self)
        guard let data = shapeStr.data(using: .utf8) else {
            shape = []
            return
        }
        shape = try JSONDecoder().decode([Int].self, from: data)
    }
}
