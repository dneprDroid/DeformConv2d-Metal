import Foundation

struct DeformConv2dParams: Decodable {
    
    let outShape: ShapeParam
    let offsetShape: ShapeParam
    let maskShape: ShapeParam

    let inChannels: Int32
    let height: Int32
    let width: Int32
    let weightH: Int32
    let weightW: Int32
    let padH: Int32
    let padW: Int32
    let strideH: Int32
    let strideW: Int32
    let dilationH: Int32
    let dilationW: Int32
    let outH: Int32
    let outW: Int32
    let parallelImgs: Int32
    let deformableGroup: Int32
    let useMask: Bool
    
    func gpuParams() -> GPUParams {
        return GPUParams(
            n_in_channels: inChannels,
            height: height,
            width: width,
            weight_h: weightH,
            weight_w: weightW,
            pad_h: padH,
            pad_w: padW,
            stride_h: strideH,
            stride_w: strideW,
            dilation_h: dilationH,
            dilation_w: dilationW,
            out_h: outH,
            out_w: outW,
            parallel_imgs: parallelImgs,
            deformable_group: deformableGroup,
            use_mask: useMask
        )
    }
}

extension DeformConv2dParams {
    
    // will be copied to shader
    struct GPUParams {
        let n_in_channels: Int32
        let height: Int32
        let width: Int32
        let weight_h: Int32
        let weight_w: Int32
        let pad_h: Int32
        let pad_w: Int32
        let stride_h: Int32
        let stride_w: Int32
        let dilation_h: Int32
        let dilation_w: Int32
        let out_h: Int32
        let out_w: Int32
        let parallel_imgs: Int32
        let deformable_group: Int32
        
        let use_mask: Bool
    }
    
    static func decode(from parameters: [String: Any]) throws -> DeformConv2dParams {
        let parametersJson = try JSONSerialization.data(withJSONObject: parameters)
        return try JSONDecoder().decode(DeformConv2dParams.self, from: parametersJson)
    }
}
