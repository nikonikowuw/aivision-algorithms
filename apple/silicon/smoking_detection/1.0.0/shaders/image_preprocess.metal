#include <metal_stdlib>
using namespace metal;

struct PreprocessParams {
    float4 source_rect;
    float4 content;  // pad_x, pad_y, scale, unused
    uint full_range;
};

constant constexpr sampler linear_sampler(coord::pixel, address::clamp_to_edge,
                                            filter::linear);

inline bool map_source(uint2 gid, constant PreprocessParams& params,
                       thread float2& source) {
    const float2 pixel = float2(gid) + 0.5f;
    const float2 content_size = params.source_rect.zw * params.content.z;
    if (pixel.x < params.content.x || pixel.y < params.content.y ||
        pixel.x >= params.content.x + content_size.x ||
        pixel.y >= params.content.y + content_size.y) {
        return false;
    }
    source = params.source_rect.xy +
             (pixel - params.content.xy) / params.content.z;
    return true;
}

kernel void preprocess_bgra(texture2d<float, access::sample> source_texture [[texture(0)]],
                            texture2d<float, access::write> output_texture [[texture(2)]],
                            constant PreprocessParams& params [[buffer(0)]],
                            uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= output_texture.get_width() || gid.y >= output_texture.get_height()) return;
    float2 source;
    if (!map_source(gid, params, source)) {
        output_texture.write(float4(114.0f / 255.0f, 114.0f / 255.0f,
                                    114.0f / 255.0f, 1.0f), gid);
        return;
    }
    output_texture.write(source_texture.sample(linear_sampler, source), gid);
}

kernel void preprocess_nv12(texture2d<float, access::sample> y_texture [[texture(0)]],
                            texture2d<float, access::sample> uv_texture [[texture(1)]],
                            texture2d<float, access::write> output_texture [[texture(2)]],
                            constant PreprocessParams& params [[buffer(0)]],
                            uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= output_texture.get_width() || gid.y >= output_texture.get_height()) return;
    float2 source;
    if (!map_source(gid, params, source)) {
        output_texture.write(float4(114.0f / 255.0f, 114.0f / 255.0f,
                                    114.0f / 255.0f, 1.0f), gid);
        return;
    }

    float y = y_texture.sample(linear_sampler, source).r;
    float2 uv = uv_texture.sample(linear_sampler, source * 0.5f).rg;
    float cb = uv.x - 0.5f;
    float cr = uv.y - 0.5f;
    if (params.full_range == 0) {
        y = max(0.0f, (y - 16.0f / 255.0f) * (255.0f / 219.0f));
        cb *= 255.0f / 224.0f;
        cr *= 255.0f / 224.0f;
    }

    // BT.709 is the explicit fallback for HD surveillance input.
    float3 rgb;
    rgb.r = y + 1.5748f * cr;
    rgb.g = y - 0.1873f * cb - 0.4681f * cr;
    rgb.b = y + 1.8556f * cb;
    output_texture.write(float4(clamp(rgb, 0.0f, 1.0f), 1.0f), gid);
}
