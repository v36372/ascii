// Lit real 3D triangles, rendered to a depth-tested 58x56 target every frame.
struct Camera { view_projection: mat4x4<f32> };
@group(0) @binding(0) var<uniform> camera: Camera;
struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};
struct Varying {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
};
@vertex fn vs(vertex: Vertex) -> Varying {
    var out: Varying;
    out.position = camera.view_projection * vec4(vertex.position,1.0);
    out.normal = vertex.normal;
    out.color = vertex.color;
    return out;
}
@fragment fn fs(input: Varying) -> @location(0) vec4<f32> {
    let normal = normalize(input.normal);
    let moon = max(dot(normal,normalize(vec3(-2.0,3.0,2.2))),0.0);
    let rim = max(dot(normal,normalize(vec3(2.5,1.0,-2.0))),0.0);
    let lighting = vec3(0.10,0.12,0.18) + vec3(1.55,1.65,1.85)*moon
        + vec3(0.22,0.32,0.58)*rim;
    return vec4(input.color*lighting,1.0);
}
