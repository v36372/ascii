// SPDX-License-Identifier: Apache-2.0
//! Lazily allocated optional live-character pass. No scene/demo dependency.
use super::{
    buffer_capacity, depth_state, knight, texture, uniform_entry, vertex_buffer, KnightPose,
    DEPTH_FORMAT, KNIGHT_FORMAT,
};
use wgpu::util::DeviceExt;

pub(super) struct KnightPass {
    knight_pipeline: wgpu::RenderPipeline,
    knight_bind: wgpu::BindGroup,
    knight_vertices: wgpu::Buffer,
    knight_capacity: u64,
    knight_target: wgpu::Texture,
    knight_depth: wgpu::Texture,
}

impl KnightPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let knight_target = texture(
            device,
            "live knight color",
            knight::WIDTH,
            knight::HEIGHT,
            KNIGHT_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let knight_depth = texture(
            device,
            "live knight depth",
            knight::WIDTH,
            knight::HEIGHT,
            DEPTH_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let knight_camera = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("knight orthographic camera"),
            contents: bytemuck::cast_slice(&knight::camera().to_cols_array()),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let knight_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("knight camera layout"),
            entries: &[uniform_entry(0, wgpu::ShaderStages::VERTEX)],
        });
        let knight_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("knight camera"),
            layout: &knight_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: knight_camera.as_entire_binding(),
            }],
        });
        let knight_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("knight diffuse lighting"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/knight.wgsl").into()),
        });
        let knight_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("knight pipeline layout"),
                bind_group_layouts: &[&knight_layout],
                push_constant_ranges: &[],
            });
        const KNIGHT_ATTRIBUTES: [wgpu::VertexAttribute; 3] =
            wgpu::vertex_attr_array![0=>Float32x3,1=>Float32x3,2=>Float32x3];
        let knight_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("live 3D knight"),
            layout: Some(&knight_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &knight_shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<knight::Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &KNIGHT_ATTRIBUTES,
                }],
            },
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(depth_state()),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &knight_shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: KNIGHT_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });
        let knight_capacity = 256 * 1024;
        let knight_vertices = vertex_buffer(device, "posed knight triangles", knight_capacity);

        Self {
            knight_pipeline,
            knight_bind,
            knight_vertices,
            knight_capacity,
            knight_target,
            knight_depth,
        }
    }

    pub fn texture_view(&self) -> wgpu::TextureView {
        self.knight_target.create_view(&Default::default())
    }

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        pose: KnightPose,
    ) -> Result<(), String> {
        let vertices = knight::mesh(pose);
        let vertex_bytes = bytemuck::cast_slice::<_, u8>(&vertices);
        if vertex_bytes.len() as u64 > self.knight_capacity {
            self.knight_capacity =
                buffer_capacity(vertex_bytes.len() as u64, device.limits().max_buffer_size)?;
            self.knight_vertices =
                vertex_buffer(device, "posed knight triangles", self.knight_capacity);
        }
        queue.write_buffer(&self.knight_vertices, 0, vertex_bytes);

        let knight_view = self.texture_view();
        let knight_depth_view = self.knight_depth.create_view(&Default::default());
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("pose and light 3D knight"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &knight_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &knight_depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        pass.set_pipeline(&self.knight_pipeline);
        pass.set_bind_group(0, &self.knight_bind, &[]);
        pass.set_vertex_buffer(0, self.knight_vertices.slice(..vertex_bytes.len() as u64));
        pass.draw(0..vertices.len() as u32, 0..1);

        Ok(())
    }
}
