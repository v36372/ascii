// SPDX-License-Identifier: Apache-2.0
//! Native instanced glyph renderer and a live 3D-to-glyph knight pipeline.
//!
//! The knight is posed on the CPU as triangles, lit/depth-tested on the GPU at
//! 58x56, then sampled as 29x28 independently placed glyph cells. There is no
//! baked character image and no character-image readback during normal drawing.
//! GPU validation/device-loss errors are retained and returned to the caller;
//! capture waits for completion and never reports success after a device error.

#[path = "atlas.rs"]
mod atlas;
#[path = "knight.rs"]
mod knight;
#[path = "knight_pass.rs"]
mod knight_pass;

use super::{GlyphInstance, KnightPose};
use crate::spatial::SceneFrame;
use std::{
    fs::File,
    io::BufWriter,
    path::Path,
    sync::{mpsc, Arc, Mutex},
    time::Duration,
};
use wgpu::util::DeviceExt;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const KNIGHT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const GPU_TIMEOUT: Duration = Duration::from_secs(30);
type Errors = Arc<Mutex<Option<String>>>;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SceneUniform {
    camera_view: [f32; 4],
    ramp: [[u32; 4]; 4],
}

/// Borrowed render input. Custom frame types can implement [FrameSource] without
/// cloning their instance buffer. All fields are validated at the GPU boundary.
pub struct FrameView<'a> {
    pub camera: [f32; 2],
    pub view_height: f32,
    pub time: f32,
    pub background: [f64; 4],
    pub instances: &'a [GlyphInstance],
    pub knight: Option<KnightPose>,
}

/// Device-independent adapter for native scene frames.
pub trait FrameSource {
    fn frame_view(&self) -> FrameView<'_>;
}

impl FrameSource for SceneFrame {
    fn frame_view(&self) -> FrameView<'_> {
        FrameView::from(self)
    }
}

impl<'a> From<&'a SceneFrame> for FrameView<'a> {
    fn from(frame: &'a SceneFrame) -> Self {
        Self {
            camera: frame.camera.center,
            view_height: frame.camera.view_height,
            time: 0.0,
            background: frame.background.channels().map(f64::from),
            instances: frame.instances(),
            knight: frame.knight,
        }
    }
}

/// Owns GPU resources, but not the window/surface or its configuration.
pub struct GpuRenderer {
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub format: wgpu::TextureFormat,
    errors: Errors,
    glyph_pipeline: wgpu::RenderPipeline,
    glyph_bind: wgpu::BindGroup,
    scene_uniform: wgpu::Buffer,
    ramp: [[u32; 4]; 4],
    instance_buffer: wgpu::Buffer,
    instance_capacity: u64,
    instance_scratch: Vec<GlyphInstance>,
    depth_target: Option<wgpu::Texture>,
    glyph_layout: wgpu::BindGroupLayout,
    atlas_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    knight: Option<knight_pass::KnightPass>,
}

fn record_error(errors: &Errors, message: String) {
    let mut error = errors.lock().unwrap_or_else(|poison| poison.into_inner());
    if error.is_none() {
        *error = Some(message);
    }
}

fn check_errors(errors: &Errors) -> Result<(), String> {
    match errors
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .as_ref()
    {
        Some(error) => Err(error.clone()),
        None => Ok(()),
    }
}

fn texture(
    device: &wgpu::Device,
    label: &str,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

fn vertex_buffer(device: &wgpu::Device, label: &str, capacity: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: capacity,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn depth_state() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: true,
        depth_compare: wgpu::CompareFunction::LessEqual,
        stencil: Default::default(),
        bias: Default::default(),
    }
}

fn uniform_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture_entry(binding: u32, filterable: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn glyph_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    scene_uniform: &wgpu::Buffer,
    atlas: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    character: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("glyph resources"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: scene_uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(atlas),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(character),
            },
        ],
    })
}

impl GpuRenderer {
    pub async fn new(
        instance: &wgpu::Instance,
        surface: Option<&wgpu::Surface<'_>>,
    ) -> Result<Self, String> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: surface,
            })
            .await
            .map_err(|e| format!("no compatible native GPU adapter: {e}"))?;
        let format = match surface {
            Some(surface) => surface
                .get_capabilities(&adapter)
                .formats
                .into_iter()
                .find(|format| {
                    matches!(
                        format,
                        wgpu::TextureFormat::Rgba8UnormSrgb | wgpu::TextureFormat::Bgra8UnormSrgb
                    )
                })
                .ok_or("surface does not support an 8-bit sRGB format")?,
            None => wgpu::TextureFormat::Rgba8UnormSrgb,
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("spatial GPU"),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("request GPU device: {e}"))?;
        let errors: Errors = Arc::new(Mutex::new(None));
        let uncaptured = errors.clone();
        device.on_uncaptured_error(Arc::new(move |error| {
            record_error(&uncaptured, format!("GPU error: {error}"))
        }));
        let lost = errors.clone();
        device.set_device_lost_callback(move |reason, message| {
            record_error(&lost, format!("GPU device lost ({reason:?}): {message}"))
        });
        let atlas = atlas::build()?;
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let atlas_texture = texture(
            &device,
            "ASCII atlas",
            atlas::WIDTH,
            atlas::HEIGHT,
            wgpu::TextureFormat::R8Unorm,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        queue.write_texture(
            atlas_texture.as_image_copy(),
            &atlas.pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(atlas::WIDTH),
                rows_per_image: Some(atlas::HEIGHT),
            },
            atlas_texture.size(),
        );
        let atlas_view = atlas_texture.create_view(&Default::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("glyph linear sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        // A one-texel placeholder keeps the optional texture binding valid.
        // No character pipeline, mesh buffer, or full render targets exist yet.
        let placeholder = texture(
            &device,
            "unused character texture",
            1,
            1,
            KNIGHT_FORMAT,
            wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let knight_view = placeholder.create_view(&Default::default());
        let scene_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("glyph camera and luminance ramp"),
            contents: bytemuck::bytes_of(&SceneUniform {
                camera_view: [0.0, 0.0, 640.0, 640.0],
                ramp: atlas.ramp,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let glyph_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("glyph resources"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
                texture_entry(1, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                texture_entry(3, false),
            ],
        });
        let glyph_bind = glyph_bind_group(
            &device,
            &glyph_layout,
            &scene_uniform,
            &atlas_view,
            &sampler,
            &knight_view,
        );
        let glyph_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("instanced glyph shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/glyph.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("glyph pipeline layout"),
            bind_group_layouts: &[&glyph_layout],
            push_constant_ranges: &[],
        });
        const GLYPH_ATTRIBUTES: [wgpu::VertexAttribute; 7] = wgpu::vertex_attr_array![0=>Float32x2,1=>Float32x2,2=>Float32x4,3=>Float32,4=>Float32,5=>Uint32,6=>Uint32];
        let glyph_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("independent glyph quads"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &glyph_shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<GlyphInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &GLYPH_ATTRIBUTES,
                }],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(depth_state()),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &glyph_shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });
        let instance_capacity = 48 * 1024;
        let instance_buffer = vertex_buffer(&device, "glyph instances", instance_capacity);
        if let Some(error) = device.pop_error_scope().await {
            return Err(format!("initialize spatial GPU pipelines: {error}"));
        }
        device
            .poll(wgpu::PollType::Poll)
            .map_err(|e| format!("poll GPU initialization: {e}"))?;
        check_errors(&errors)?;
        Ok(Self {
            adapter,
            device,
            queue,
            format,
            errors,
            glyph_pipeline,
            glyph_bind,
            scene_uniform,
            ramp: atlas.ramp,
            instance_buffer,
            instance_capacity,
            instance_scratch: Vec::new(),
            depth_target: None,
            glyph_layout,
            atlas_view,
            sampler,
            knight: None,
        })
    }

    fn validate_size(&self, width: u32, height: u32) -> Result<(), String> {
        let max = self.device.limits().max_texture_dimension_2d;
        if width == 0 || height == 0 || width > max || height > max {
            return Err(format!(
                "render size {width}x{height} must be nonzero and at most {max} per axis"
            ));
        }
        Ok(())
    }

    /// Render any FrameSource in one submission; no demo-specific frame is required.
    /// Asynchronous device errors are reported here or on the next call;
    /// capture also waits for completion.
    pub fn render(
        &mut self,
        target: &wgpu::TextureView,
        width: u32,
        height: u32,
        frame: &impl FrameSource,
    ) -> Result<(), String> {
        self.render_frame(target, width, height, &frame.frame_view())
    }

    /// Render an independently authored scene. No character is added unless
    /// `frame.knight` is Some. The target must match this renderer's format/size.
    pub fn render_scene(
        &mut self,
        target: &wgpu::TextureView,
        width: u32,
        height: u32,
        frame: &SceneFrame,
    ) -> Result<(), String> {
        self.render_frame(target, width, height, &FrameView::from(frame))
    }

    fn render_frame(
        &mut self,
        target: &wgpu::TextureView,
        width: u32,
        height: u32,
        frame: &FrameView<'_>,
    ) -> Result<(), String> {
        self.validate_size(width, height)?;
        validate_frame(frame)?;
        check_errors(&self.errors)?;
        let view_width = frame.view_height * width as f32 / height as f32;
        if !view_width.is_finite() || view_width <= 0.0 {
            return Err("invalid projected view width".into());
        }
        let count = frame
            .instances
            .len()
            .checked_add(if frame.knight.is_some() {
                (knight::COLS * knight::ROWS) as usize
            } else {
                0
            })
            .ok_or("glyph count overflow")?;
        let count_u32 = u32::try_from(count).map_err(|_| "too many glyph instances")?;
        let bytes = (count as u64)
            .checked_mul(std::mem::size_of::<GlyphInstance>() as u64)
            .ok_or("glyph buffer size overflow")?;
        let capacity = buffer_capacity(bytes, self.device.limits().max_buffer_size)?;
        if bytes > self.instance_capacity {
            self.instance_buffer = vertex_buffer(&self.device, "glyph instances", capacity);
            self.instance_capacity = capacity;
        }
        self.instance_scratch.clear();
        self.instance_scratch.extend_from_slice(frame.instances);
        if let Some(knight) = frame.knight {
            self.instance_scratch.extend(knight::cells(knight));
        }
        self.queue.write_buffer(
            &self.instance_buffer,
            0,
            bytemuck::cast_slice(&self.instance_scratch),
        );
        self.queue.write_buffer(
            &self.scene_uniform,
            0,
            bytemuck::bytes_of(&SceneUniform {
                camera_view: [
                    frame.camera[0],
                    frame.camera[1],
                    view_width,
                    frame.view_height,
                ],
                ramp: self.ramp,
            }),
        );
        if self
            .depth_target
            .as_ref()
            .is_none_or(|depth| depth.width() != width || depth.height() != height)
        {
            self.depth_target = Some(texture(
                &self.device,
                "spatial depth",
                width,
                height,
                DEPTH_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT,
            ));
        }
        let depth_view = self
            .depth_target
            .as_ref()
            .ok_or("missing spatial depth target")?
            .create_view(&Default::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("spatial frame"),
            });
        if let Some(pose) = frame.knight {
            if self.knight.is_none() {
                let pass = knight_pass::KnightPass::new(&self.device);
                self.glyph_bind = glyph_bind_group(
                    &self.device,
                    &self.glyph_layout,
                    &self.scene_uniform,
                    &self.atlas_view,
                    &self.sampler,
                    &pass.texture_view(),
                );
                self.knight = Some(pass);
            }
            self.knight
                .as_mut()
                .unwrap()
                .render(&self.device, &self.queue, &mut encoder, pose)?;
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("depth-composited glyph spatial"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: frame.background[0],
                            g: frame.background[1],
                            b: frame.background[2],
                            a: frame.background[3],
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&self.glyph_pipeline);
            pass.set_bind_group(0, &self.glyph_bind, &[]);
            if count_u32 > 0 {
                pass.set_vertex_buffer(0, self.instance_buffer.slice(..bytes));
                pass.draw(0..6, 0..count_u32);
            }
        }
        self.queue.submit([encoder.finish()]);
        self.device
            .poll(wgpu::PollType::Poll)
            .map_err(|e| format!("poll GPU frame: {e}"))?;
        check_errors(&self.errors)
    }

    /// Render and write an sRGB RGBA PNG. Row padding and BGRA surfaces are
    /// handled explicitly. Errors include GPU/device failures, mapping and I/O.
    pub fn capture_png(
        &mut self,
        width: u32,
        height: u32,
        frame: &impl FrameSource,
        path: &Path,
    ) -> Result<(), String> {
        self.capture_frame(width, height, &frame.frame_view(), path)
    }

    /// Capture a generic scene using the same pipeline as `render_scene`.
    /// Requires a graphics adapter even without a window.
    pub fn capture_scene_png(
        &mut self,
        width: u32,
        height: u32,
        frame: &SceneFrame,
        path: &Path,
    ) -> Result<(), String> {
        self.capture_frame(width, height, &FrameView::from(frame), path)
    }

    fn capture_frame(
        &mut self,
        width: u32,
        height: u32,
        frame: &FrameView<'_>,
        path: &Path,
    ) -> Result<(), String> {
        self.validate_size(width, height)?;
        validate_frame(frame)?;
        let bgra = match self.format {
            wgpu::TextureFormat::Rgba8UnormSrgb => false,
            wgpu::TextureFormat::Bgra8UnormSrgb => true,
            _ => return Err(format!("unsupported capture format {:?}", self.format)),
        };
        let (row_bytes, padded_bytes, total) = readback_layout(width, height)?;
        if total > self.device.limits().max_buffer_size {
            return Err("capture exceeds GPU readback buffer limit".into());
        }
        check_errors(&self.errors)?;
        let target = texture(
            &self.device,
            "PNG capture",
            width,
            height,
            self.format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        self.render_frame(
            &target.create_view(&Default::default()),
            width,
            height,
            frame,
        )?;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PNG padded readback"),
            size: total,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("PNG copy"),
            });
        encoder.copy_texture_to_buffer(
            target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes),
                    rows_per_image: Some(height),
                },
            },
            target.size(),
        );
        let submission = self.queue.submit([encoder.finish()]);
        check_errors(&self.errors)?;
        let (tx, rx) = mpsc::sync_channel(1);
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(GPU_TIMEOUT),
            })
            .map_err(|e| format!("wait for PNG capture: {e}"))?;
        check_errors(&self.errors)?;
        rx.recv_timeout(GPU_TIMEOUT)
            .map_err(|e| format!("PNG readback callback: {e}"))?
            .map_err(|e| format!("map PNG readback: {e}"))?;
        let mapped = buffer.slice(..).get_mapped_range();
        let pixels = unpack_rows(&mapped, row_bytes, padded_bytes, height, bgra)?;
        drop(mapped);
        buffer.unmap();
        let file = File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
        let mut encoder = png::Encoder::new(BufWriter::new(file), width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
        let mut writer = encoder
            .write_header()
            .map_err(|e| format!("write PNG header: {e}"))?;
        writer
            .write_image_data(&pixels)
            .map_err(|e| format!("write PNG pixels: {e}"))?;
        writer.finish().map_err(|e| format!("finish PNG: {e}"))?;
        check_errors(&self.errors)
    }
}

fn buffer_capacity(bytes: u64, limit: u64) -> Result<u64, String> {
    if bytes > limit {
        return Err(format!(
            "GPU buffer requires {bytes} bytes, limit is {limit}"
        ));
    }
    Ok(bytes
        .max(4)
        .checked_next_power_of_two()
        .unwrap_or(limit)
        .min(limit))
}

fn validate_frame(frame: &FrameView<'_>) -> Result<(), String> {
    let knight = frame.knight.unwrap_or_default();
    if !frame.background.iter().all(|v| (0.0..=1.0).contains(v)) {
        return Err("frame background must be finite linear RGBA in 0..=1".into());
    }
    if !frame.view_height.is_finite()
        || frame.view_height <= 0.0
        || !frame
            .camera
            .iter()
            .chain(&knight.position)
            .all(|v| v.is_finite())
        || ![
            frame.time,
            knight.time,
            knight.yaw,
            knight.phase,
            knight.movement,
        ]
        .iter()
        .all(|v| v.is_finite())
    {
        return Err(
            "frame camera, scale, time and knight pose must be finite (view height > 0)".into(),
        );
    }
    for (i, glyph) in frame.instances.iter().enumerate() {
        if !glyph
            .position
            .iter()
            .chain(&glyph.size)
            .chain(&glyph.color)
            .all(|v| v.is_finite())
            || !glyph.rotation.is_finite()
            || !(0.0..=1.0).contains(&glyph.depth)
            || glyph.size.iter().any(|&size| size < 0.0)
            || glyph.kind > 2
            || (glyph.kind == 0 && glyph.glyph > 127)
            || (glyph.kind == 2
                && (frame.knight.is_none() || glyph.glyph >= knight::COLS * knight::ROWS))
        {
            return Err(format!(
                "invalid glyph instance {i}: non-finite/out-of-range geometry or glyph"
            ));
        }
    }
    Ok(())
}

fn readback_layout(width: u32, height: u32) -> Result<(u32, u32, u64), String> {
    if width == 0 || height == 0 {
        return Err("PNG dimensions must be nonzero".into());
    }
    let row = width.checked_mul(4).ok_or("PNG row size overflow")?;
    let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = row
        .checked_add(alignment - 1)
        .ok_or("PNG row alignment overflow")?
        / alignment
        * alignment;
    Ok((row, padded, u64::from(padded) * u64::from(height)))
}

fn unpack_rows(
    bytes: &[u8],
    row: u32,
    padded: u32,
    height: u32,
    bgra: bool,
) -> Result<Vec<u8>, String> {
    if !row.is_multiple_of(4)
        || padded < row
        || padded == 0
        || (bytes.len() as u64) < u64::from(padded) * u64::from(height)
    {
        return Err("invalid PNG readback layout".into());
    }
    let size = usize::try_from(u64::from(row) * u64::from(height))
        .map_err(|_| "PNG image too large for host")?;
    let mut result = Vec::with_capacity(size);
    for source in bytes.chunks_exact(padded as usize).take(height as usize) {
        result.extend_from_slice(&source[..row as usize]);
    }
    if bgra {
        for pixel in result.as_chunks_mut::<4>().0 {
            pixel.swap(0, 2);
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn frame() -> SceneFrame {
        SceneFrame {
            knight: Some(KnightPose::default()),
            ..SceneFrame::default()
        }
    }
    #[test]
    fn instance_layout_matches_shader_attributes() {
        assert_eq!(std::mem::size_of::<GlyphInstance>(), 48);
        assert_eq!(std::mem::offset_of!(GlyphInstance, rotation), 32);
        assert_eq!(std::mem::offset_of!(GlyphInstance, glyph), 40);
        assert_eq!(std::mem::offset_of!(GlyphInstance, kind), 44);
        assert_eq!(std::mem::size_of::<SceneUniform>(), 80);
    }
    #[test]
    fn capture_padding_and_bgra_conversion() {
        assert_eq!(readback_layout(65, 2).unwrap(), (260, 512, 1024));
        assert_eq!(readback_layout(64, 3).unwrap(), (256, 256, 768));
        assert!(readback_layout(0, 1).is_err());
        assert!(readback_layout(u32::MAX, 1).is_err());
        let source = [1, 2, 3, 4, 99, 99, 99, 99, 5, 6, 7, 8, 99, 99, 99, 99];
        assert_eq!(
            unpack_rows(&source, 4, 8, 2, false).unwrap(),
            [1, 2, 3, 4, 5, 6, 7, 8]
        );
        assert_eq!(
            unpack_rows(&source, 4, 8, 2, true).unwrap(),
            [3, 2, 1, 4, 7, 6, 5, 8]
        );
        assert!(unpack_rows(&source, 4, 8, 3, false).is_err());
    }
    #[test]
    fn invalid_frame_and_oversized_buffers_are_rejected() {
        let mut frame = frame();
        assert!(validate_frame(&FrameView::from(&frame)).is_ok());
        frame.knight.as_mut().unwrap().phase = f32::NAN;
        assert!(validate_frame(&FrameView::from(&frame)).is_err());
        frame.knight.as_mut().unwrap().phase = 0.0;
        frame.camera.view_height = 0.0;
        assert!(validate_frame(&FrameView::from(&frame)).is_err());
        frame.camera.view_height = 640.0;
        frame.instances.push(GlyphInstance {
            position: [0.0; 2],
            size: [1.0; 2],
            color: [1.0; 4],
            rotation: 0.0,
            depth: 0.5,
            glyph: 65,
            kind: 0,
        });
        assert!(validate_frame(&FrameView::from(&frame)).is_ok());
        frame.instances[0].depth = f32::NAN;
        assert!(validate_frame(&FrameView::from(&frame)).is_err());
        assert_eq!(buffer_capacity(100, 1000).unwrap(), 128);
        assert_eq!(buffer_capacity(999, 1000).unwrap(), 1000);
        assert!(buffer_capacity(1001, 1000).is_err());
    }

    #[test]
    fn generic_frame_validation_and_borrowed_adapter() {
        let mut generic = SceneFrame::default();
        assert!(validate_frame(&FrameView::from(&generic)).is_ok());
        generic.camera.view_height = 0.0;
        assert!(validate_frame(&FrameView::from(&generic)).is_err());
        generic.camera.view_height = 640.0;
        generic.knight = Some(KnightPose {
            yaw: f32::NAN,
            ..KnightPose::default()
        });
        assert!(validate_frame(&FrameView::from(&generic)).is_err());
        generic.knight = None;
        assert!(validate_frame(&FrameView::from(&generic)).is_ok());
        let source = frame();
        let view = source.frame_view();
        assert_eq!(view.knight, source.knight);
        assert_eq!(view.background, [0.0, 0.0, 0.0, 1.0]);
        assert!(std::ptr::eq(view.instances, source.instances.as_slice()));
        let mut view = source.frame_view();
        view.background[0] = f64::NAN;
        assert!(validate_frame(&view).is_err());
        view.background = [0.0, 0.0, 0.0, 1.0];
        view.knight = None;
        let cells: Vec<_> = knight::cells(KnightPose::default()).collect();
        view.instances = &cells;
        assert!(
            validate_frame(&view).is_err(),
            "character cells require an enabled character pass"
        );
    }

    fn decode(path: &Path) -> Vec<u8> {
        let decoder = png::Decoder::new(File::open(path).unwrap());
        let mut reader = decoder.read_info().unwrap();
        let mut bytes = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut bytes).unwrap();
        bytes.truncate(info.buffer_size());
        bytes
    }

    #[test]
    fn asynchronous_gpu_errors_are_retained() {
        let errors: Errors = Arc::new(Mutex::new(None));
        assert!(check_errors(&errors).is_ok());
        record_error(&errors, "original GPU failure".into());
        record_error(&errors, "consequent error".into());
        for _ in 0..2 {
            assert_eq!(check_errors(&errors), Err("original GPU failure".into()));
        }
    }

    #[test]
    #[ignore = "requires native GPU; run with --ignored --nocapture"]
    fn gpu_character_resources_are_optional_and_lazy() {
        let instance = wgpu::Instance::default();
        let mut renderer = pollster::block_on(GpuRenderer::new(&instance, None)).unwrap();
        assert!(renderer.knight.is_none());
        let path =
            std::env::temp_dir().join(format!("cosplay-lazy-knight-{}.png", std::process::id()));
        let mut frame = SceneFrame::default();
        renderer.capture_scene_png(64, 64, &frame, &path).unwrap();
        assert!(
            renderer.knight.is_none(),
            "glyph-only scenes must not allocate character resources"
        );
        frame.knight = Some(KnightPose::default());
        renderer.capture_scene_png(64, 64, &frame, &path).unwrap();
        assert!(renderer.knight.is_some());
        frame.knight = None;
        renderer.capture_scene_png(64, 64, &frame, &path).unwrap();
        assert!(decode(&path)
            .as_chunks::<4>()
            .0
            .iter()
            .all(|p| *p == [0, 0, 0, 255]));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    #[ignore = "requires native GPU; run with --ignored --nocapture"]
    fn gpu_glyph_rotation_depth_transparency_resize_and_errors() {
        let instance = wgpu::Instance::default();
        let mut renderer = pollster::block_on(GpuRenderer::new(&instance, None)).unwrap();
        let directory = std::env::temp_dir().join(format!("cosplay-glyphs-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let mut frame = frame();
        frame.camera.view_height = 100.0;
        frame.knight.as_mut().unwrap().position = [2000.0, 2000.0]; // Isolate the sprite pipeline.
        let quad = GlyphInstance {
            position: [0.0; 2],
            size: [20.0; 2],
            color: [0.0, 0.0, 1.0, 1.0],
            rotation: 0.0,
            depth: 0.2,
            glyph: 32,
            kind: 1,
        };
        frame.instances = vec![
            quad,
            // A later but farther red backing must not overwrite blue.
            GlyphInstance {
                color: [1.0, 0.0, 0.0, 1.0],
                depth: 0.8,
                ..quad
            },
            // Equal depth is last-write-wins, giving green.
            GlyphInstance {
                color: [0.0, 1.0, 0.0, 1.0],
                ..quad
            },
            // A nearer blank atlas cell must discard, not erase the backing.
            GlyphInstance {
                color: [1.0; 4],
                depth: 0.1,
                kind: 0,
                ..quad
            },
            GlyphInstance {
                position: [-30.0, 0.0],
                size: [30.0, 4.0],
                color: [1.0, 1.0, 0.0, 1.0],
                rotation: std::f32::consts::FRAC_PI_2,
                ..quad
            },
            GlyphInstance {
                position: [30.0, 0.0],
                size: [20.0, 30.0],
                color: [1.0; 4],
                glyph: u32::from(b'#'),
                kind: 0,
                ..quad
            },
        ];
        let path = directory.join("glyphs.png");
        renderer.capture_png(101, 100, &frame, &path).unwrap();
        let pixels = decode(&path);
        let pixel = |x: usize, y: usize| &pixels[(y * 101 + x) * 4..(y * 101 + x) * 4 + 4];
        assert_eq!(pixel(50, 50), [0, 255, 0, 255]);
        assert_eq!(pixel(20, 40), [255, 255, 0, 255]);
        assert_eq!(
            pixel(30, 50),
            pixel(0, 0),
            "rotation must turn the long quad vertical"
        );
        let ink = (35..65)
            .flat_map(|y| (70..91).map(move |x| (x, y)))
            .filter(|&(x, y)| pixel(x, y)[0] > 240)
            .count();
        assert!(ink > 10, "normal glyphs must sample the ASCII atlas");
        let resized = directory.join("resized.png");
        renderer.capture_png(202, 200, &frame, &resized).unwrap();
        let resized = decode(&resized);
        assert_eq!(
            &resized[(100 * 202 + 101) * 4..(100 * 202 + 101) * 4 + 4],
            [0, 255, 0, 255]
        );
        assert!(renderer.capture_png(0, 100, &frame, &path).is_err());
        assert!(
            renderer.capture_png(101, 100, &frame, &directory).is_err(),
            "directory is not a writable PNG file"
        );

        // An incompatible caller-supplied target must return a GPU error, not
        // panic or silently return a successful blank image. Device errors stick.
        let wrong_target = texture(
            &renderer.device,
            "deliberately wrong format",
            101,
            100,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        assert!(renderer
            .render(
                &wrong_target.create_view(&Default::default()),
                101,
                100,
                &frame
            )
            .is_err());
        assert!(renderer
            .capture_png(101, 100, &frame, &directory.join("must-not-succeed.png"))
            .is_err());
    }

    /// Explicit opt-in: this test requires a real available GPU adapter. It must
    /// fail, not silently skip or draw a CPU substitute, when requested without one.
    #[test]
    #[ignore = "requires native GPU; run with --ignored --nocapture"]
    fn gpu_capture_has_live_animated_knight() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let mut renderer =
            pollster::block_on(GpuRenderer::new(&instance, None)).expect("native GPU renderer");
        eprintln!("capture adapter: {:?}", renderer.adapter.get_info());
        let directory = std::env::temp_dir().join(format!("cosplay-knight-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let mut frame = frame();
        frame.camera.center = [0.0, 70.0];
        frame.camera.view_height = 230.0;
        let first = directory.join("idle.png");
        let repeat = directory.join("idle-repeat.png");
        let turned = directory.join("walking.png");
        renderer.capture_png(481, 480, &frame, &first).unwrap();
        renderer.capture_png(481, 480, &frame, &repeat).unwrap();
        frame.knight.as_mut().unwrap().yaw = 0.75;
        frame.knight.as_mut().unwrap().phase = 1.1;
        frame.knight.as_mut().unwrap().movement = 1.0;
        frame.knight.as_mut().unwrap().time = 2.0;
        renderer.capture_png(481, 480, &frame, &turned).unwrap();
        let idle = decode(&first);
        assert_eq!(
            idle,
            decode(&repeat),
            "identical input captures must be deterministic"
        );
        assert!(
            idle.as_chunks::<4>()
                .0
                .iter()
                .filter(|p| p[0] > 100 && p[1] > 100 && p[2] > 100)
                .count()
                > 200,
            "knight must have visible lit glyph pixels"
        );
        assert_ne!(
            idle,
            decode(&turned),
            "3D pose/yaw must change rendered glyphs"
        );
        eprintln!("validated live knight PNGs: {}", directory.display());
    }
}
