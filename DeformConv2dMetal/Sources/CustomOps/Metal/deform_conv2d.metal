#include <metal_stdlib>
using namespace metal;

#define CHANNELS (4)

namespace dneprDroid {
    
    typedef struct {
        int32_t n_in_channels;
        int32_t height;
        int32_t width;
        int32_t weight_h;
        int32_t weight_w;
        int32_t pad_h;
        int32_t pad_w;
        int32_t stride_h;
        int32_t stride_w;
        int32_t dilation_h;
        int32_t dilation_w;
        int32_t out_h;
        int32_t out_w;
        int32_t parallel_imgs;
        int32_t deformable_group;
        bool use_mask;
    } deform_conv2d_params;
    
    static inline uint4 find_texture_coord_with_channels(const int index, const int width, const int height, const int array_size, const int channels_per_sample) {
        uint4 computed = 0;
        
        int out_x = index % width;
        int out_y = (index / width) % height;
        int out_b = (index / (width * height * channels_per_sample)) % array_size;
        int out_c = (index / (width * height)) % channels_per_sample;
            
        computed[0] = out_x;
        computed[1] = out_y;
        computed[2] = out_b;
        computed[3] = out_c; // Channel (R=0, G=1, B=2, A=3)
        
        return computed;
    }

    static inline float bilinear_interpolate_with_channels(texture2d_array<float, access::read> in, float w, float h, int width, int height, int in_c, int channels_per_sample) {
        if (h <= -1 || height <= h || w <= -1 || width <= w) {
            return 0;
        }
                
        int h_low = floor(h);
        int w_low = floor(w);
        int h_high = h_low + 1;
        int w_high = w_low + 1;

        float lh = h - h_low;
        float lw = w - w_low;
        float hh = 1 - lh, hw = 1 - lw;

        int channel = in_c % channels_per_sample;
        int array_index = in_c / channels_per_sample;

        float v1 = 0;
        if (h_low >= 0 && w_low >= 0)
            v1 = in.read(uint2(w_low, h_low), array_index)[channel];
        
        float v2 = 0;
        if (h_low >= 0 && w_high <= width - 1)
            v2 = in.read(uint2(w_high, h_low), array_index)[channel];
        
        float v3 = 0;
        if (h_high <= height - 1 && w_low >= 0)
            v3 = in.read(uint2(w_low, h_high), array_index)[channel];
        
        float v4 = 0;
        if (h_high <= height - 1 && w_high <= width - 1)
            v4 = in.read(uint2(w_high, h_high), array_index)[channel];
        
        float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

        float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
        return val;
    }

    kernel void deform_conv2d(texture2d_array<float, access::read> in [[texture(0)]],
                              texture2d_array<float, access::read> offset [[texture(1)]],
                              texture2d_array<float, access::read> mask [[texture(2)]],
                              const device deform_conv2d_params &params [[buffer(3)]],
                              texture2d_array<float, access::write>  out [[texture(4)]],
                              uint3 gid [[thread_position_in_grid]]) {

        float4 out_pixel = 0;
        const int out_tex_width = out.get_width();
        const int out_tex_height = out.get_height();
        const int mask_w = mask.get_width();
        const int mask_h = mask.get_height();
        const int mask_z = mask.get_array_size();
        const int offset_width = offset.get_width();
        const int offset_height = offset.get_height();
        const int offset_depth = offset.get_array_size();
        const int channels_per_sample = 4;

        for (int channel = 0; channel < CHANNELS; channel++) {
            const int col_index = (gid.z * CHANNELS * out_tex_width * out_tex_height) +
                                  ((channel) * out_tex_width * out_tex_height) +
                                  (gid.y * out_tex_width) + gid.x;
            const int out_b = 0; // TODO: multi-batch
            const int batch_sz = params.parallel_imgs;
            const int preInputIndex = col_index % ((params.weight_h * params.weight_w) * (batch_sz * params.out_h * params.out_w));
            const int inputIndex = preInputIndex / (batch_sz * params.out_h * params.out_w);
            const int i_index = inputIndex / params.weight_w;
            const int j_index = inputIndex % params.weight_h;
            const int mask_idx = i_index * params.weight_w + j_index;
            const int n_offset_grps = params.deformable_group;
            int c_per_offset_grp = params.n_in_channels / n_offset_grps;
            const int out_c = (col_index - inputIndex * (batch_sz * params.out_h * params.out_w))/(batch_sz * params.out_h * params.out_w);
            const int in_c = out_c / (params.weight_h * params.weight_w);
            const int grp_idx = in_c / c_per_offset_grp;
            const int offset_idx = 2 * mask_idx;
            const int base_offset_index = ((out_b * n_offset_grps + grp_idx) * 2 * params.weight_h * params.weight_w * params.out_h * params.out_w);
            const int offset_w_index = base_offset_index + (offset_idx + 1) * (params.out_h * params.out_w) + gid.y * params.out_w + gid.x;

            uint4 offset_coord_w = find_texture_coord_with_channels(
                  offset_w_index,
                  offset_width,
                  offset_height,
                  offset_depth,
                  channels_per_sample
            );
            const float offset_w = offset.read(offset_coord_w.xy, offset_coord_w.z)[offset_coord_w.w];
            
            uint4 offset_coord_h = find_texture_coord_with_channels(
                  base_offset_index + offset_idx * (params.out_h * params.out_w) + gid.y * params.out_w + gid.x,
                  offset_width,
                  offset_height,
                  offset_depth,
                  channels_per_sample
            );
            const float offset_h = offset.read(offset_coord_h.xy, offset_coord_h.z)[offset_coord_h.w];
            const float x = (gid.x * params.stride_w - params.pad_w) + j_index * params.dilation_w + offset_w;
            const float y = (gid.y * params.stride_h - params.pad_h) + i_index * params.dilation_h + offset_h;

            float mask_value = 1;
            if (params.use_mask) {
                const int base_mask_index = (out_b * n_offset_grps + grp_idx) * params.weight_h * params.weight_w * params.out_h * params.out_w;
                uint4 mask_coord = find_texture_coord_with_channels(
                       base_mask_index + mask_idx * (params.out_h * params.out_w) + gid.y * params.out_w + gid.x,
                       mask_w,
                       mask_h,
                       mask_z,
                       channels_per_sample
                );
                mask_value = mask.read(mask_coord.xy, mask_coord.z)[mask_coord.w];
            }

            float computed_value = bilinear_interpolate_with_channels(in, x, y, params.width, params.height, in_c, channels_per_sample);
            out_pixel[channel] = mask_value * computed_value;
        }
        out.write(out_pixel, gid.xy, gid.z);
    }
}
