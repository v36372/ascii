// World sprites are independent quads, not a screen-aligned character grid.
struct Scene {
    camera_view: vec4<f32>, // camera.xy, visible width/height
    ramp: array<vec4<u32>, 4>,
};
@group(0) @binding(0) var<uniform> scene: Scene;
@group(0) @binding(1) var atlas: texture_2d<f32>;
@group(0) @binding(2) var atlas_sampler: sampler;
@group(0) @binding(3) var knight: texture_2d<f32>;

struct Input {
    @location(0) position: vec2<f32>,
    @location(1) size: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) rotation: f32,
    @location(4) depth: f32,
    @location(5) glyph: u32,
    @location(6) kind: u32,
};
struct Varying {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) @interpolate(flat) glyph: u32,
    @location(3) @interpolate(flat) kind: u32,
};
@vertex fn vs(input: Input, @builtin(vertex_index) vertex: u32) -> Varying {
    let corners = array<vec2<f32>, 6>(
        vec2(-0.5,-0.5), vec2(0.5,-0.5), vec2(0.5,0.5),
        vec2(-0.5,-0.5), vec2(0.5,0.5), vec2(-0.5,0.5));
    let corner = corners[vertex];
    let local = corner * input.size;
    let c = cos(input.rotation);
    let s = sin(input.rotation);
    let world = input.position + vec2(c*local.x-s*local.y,s*local.x+c*local.y);
    var out: Varying;
    out.position = vec4((world-scene.camera_view.xy)*2.0/scene.camera_view.zw,input.depth,1.0);
    out.uv = vec2(corner.x+0.5,0.5-corner.y);
    out.color = input.color;
    out.glyph = input.glyph;
    out.kind = input.kind;
    return out;
}
fn ink(code: u32, uv: vec2<f32>) -> f32 {
    let atlas_uv = (vec2<f32>(f32(code%16u),f32(code/16u))+uv)/vec2(16.0,8.0);
    return smoothstep(0.04,0.32,textureSampleLevel(atlas,atlas_sampler,atlas_uv,0.0).r);
}
@fragment fn fs(input: Varying) -> @location(0) vec4<f32> {
    if input.kind == 1u {
        if input.color.a <= 0.001 { discard; }
        return input.color;
    }
    if input.kind == 2u {
        // Cell row zero is the bottom of the model. WebGPU textures start at top.
        let source = vec2<i32>(i32(input.glyph%29u)*2,(27-i32(input.glyph/29u))*2);
        let sample = (textureLoad(knight,source,0)
            + textureLoad(knight,source+vec2(1,0),0)
            + textureLoad(knight,source+vec2(0,1),0)
            + textureLoad(knight,source+vec2(1,1),0))*0.25;
        let luminance = dot(sample.rgb,vec3(0.3,0.59,0.11));
        if sample.a < 0.31 || luminance < 0.012 { discard; }
        let brightness = clamp(pow(luminance,0.85)*1.35,0.0,0.999);
        let rank = u32(floor(brightness*16.0));
        let code = scene.ramp[rank/4u][rank%4u];
        let hue = sample.rgb/max(max(sample.r,max(sample.g,sample.b)),0.05);
        let color = mix(vec3(1.0),hue,vec3(0.4))*(pow(brightness,1.3)*1.05+0.12);
        let glyph_alpha = ink(code,input.uv);
        return vec4(mix(vec3(0.012,0.014,0.022),color,vec3(glyph_alpha)),1.0);
    }
    let alpha = ink(min(input.glyph,127u),input.uv)*input.color.a;
    if alpha < 0.5 { discard; }
    return vec4(input.color.rgb,alpha);
}
