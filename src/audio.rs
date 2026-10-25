// SPDX-License-Identifier: Apache-2.0
//! Sound decoding, beat analysis, and optional playback.
//!
//! This module is intentionally usable with or without the `audio` Cargo
//! feature. Decoding, [`AudioClip`] inspection, [`BeatEnvelope`], and
//! [`SoundRegistry`] work in both configurations. Only the methods that start
//! playback need `rodio`; without the feature they return
//! [`AudioError::Playback`] instead of opening a device.
//!
//! ```no_run
//! use cosplay::audio::{AudioClip, BeatEnvelope, Sound};
//! # fn main() -> Result<(), cosplay::audio::AudioError> {
//! let clip = AudioClip::load("music.wav")?;
//! let envelope = BeatEnvelope::new(&clip);
//! let mut sound = Sound::new(clip);
//! sound.play()?;
//! # Ok(())
//! # }
//! ```
//!
//! The decoders are pure in-memory parsers for the uncompressed formats that
//! CosPlay accepts: WAV (PCM and IEEE float), AIFF/AIFF-C PCM, and AU/SND PCM.
//! Samples are converted to interleaved `f32` in `-1.0..=1.0`; non-finite float
//! input is replaced with silence and finite float input is clamped.
//!
//! [`Sound`] is deliberately single-threaded. `rodio::OutputStream` is not
//! guaranteed to be `Send`, and the output stream must outlive its `Sink`.
//! Use [`SoundRegistry`] as an owner that can be passed around within the
//! thread that owns the sounds; it is not a global static and requires no
//! unsafe code.

use std::collections::BTreeMap;
use std::fmt;
use std::io;
use std::path::Path;
use std::time::Duration;

#[cfg(feature = "audio")]
use std::sync::Arc;

#[cfg(feature = "audio")]
use rodio::buffer::SamplesBuffer;
#[cfg(feature = "audio")]
use rodio::{Sink, Source};

/// Duration of one beat-analysis amplitude window.
pub const BEAT_WINDOW_MS: u32 = 10;

/// Default length of the sliding RMS threshold window.
pub const DEFAULT_THRESHOLD_TAIL_MS: u64 = 2_000;

/// Default multiplier applied to the RMS threshold.
pub const DEFAULT_THRESHOLD_GAIN: f32 = 1.0;

/// Errors produced while loading, validating, or playing audio.
#[derive(Debug)]
pub enum AudioError {
    /// Reading a sound file failed.
    Io(io::Error),
    /// The file extension or codec is not one this module supports.
    UnsupportedFormat(String),
    /// The container or sample format is malformed.
    InvalidFormat(String),
    /// The container declares or requires more bytes than are present.
    Truncated(String),
    /// A decoded allocation would be unreasonably large or could not be made.
    TooLarge(String),
    /// The output device or playback backend reported an error.
    Playback(String),
    /// A registry operation referred to an ID that is not present.
    UnknownSound(u64),
}

impl fmt::Display for AudioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "audio I/O error: {err}"),
            Self::UnsupportedFormat(msg) => write!(f, "unsupported audio format: {msg}"),
            Self::InvalidFormat(msg) => write!(f, "invalid audio format: {msg}"),
            Self::Truncated(msg) => write!(f, "truncated audio data: {msg}"),
            Self::TooLarge(msg) => write!(f, "audio data is too large: {msg}"),
            Self::Playback(msg) => write!(f, "audio playback error: {msg}"),
            Self::UnknownSound(id) => write!(f, "unknown sound id: {id}"),
        }
    }
}

impl std::error::Error for AudioError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for AudioError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

/// A decoded, immutable-in-practice sound clip.
///
/// `samples` are interleaved by channel and normalized to `-1.0..=1.0`.
/// `origin` is the path used by [`AudioClip::load`], or `None` for clips decoded
/// directly from bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioClip {
    /// Interleaved normalized samples.
    pub samples: Vec<f32>,
    /// Samples per second per channel.
    pub sample_rate: u32,
    /// Number of interleaved channels.
    pub channels: u16,
    /// Optional source path or URI for diagnostics.
    pub origin: Option<String>,
}

impl AudioClip {
    /// Loads a WAV, AIFF/AIFF-C, or AU/SND file, choosing the decoder by the
    /// path's extension (case-insensitive).
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, AudioError> {
        let path = path.as_ref();
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .ok_or_else(|| AudioError::UnsupportedFormat("missing file extension".into()))?;
        let data = std::fs::read(path)?;
        Self::from_bytes_with_origin(&data, extension, Some(path.to_string_lossy().into_owned()))
    }

    /// Alias for [`AudioClip::load`].
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, AudioError> {
        Self::load(path)
    }

    /// Decodes an in-memory sound. `extension` may include a leading dot and is
    /// case-insensitive (`wav`, `.WAV`, `aiff`, `aifc`, `au`, `snd`).
    pub fn from_bytes(data: &[u8], extension: &str) -> Result<Self, AudioError> {
        Self::from_bytes_with_origin(data, extension, None)
    }

    /// Decodes in-memory bytes and attaches an origin for diagnostics.
    pub fn from_bytes_with_origin(
        data: &[u8],
        extension: &str,
        origin: Option<String>,
    ) -> Result<Self, AudioError> {
        let mut clip = Self::decode(data, extension)?;
        clip.origin = origin;
        Ok(clip)
    }

    /// Decodes in-memory bytes using an explicit extension.
    pub fn decode(data: &[u8], extension: &str) -> Result<Self, AudioError> {
        let ext = extension
            .trim()
            .trim_start_matches('.')
            .to_ascii_lowercase();
        let mut clip = match ext.as_str() {
            "wav" | "wave" => Self::decode_wav(data)?,
            "aif" | "aiff" | "aifc" => Self::decode_aiff(data)?,
            "au" | "snd" => Self::decode_au(data)?,
            _ => {
                return Err(AudioError::UnsupportedFormat(format!(
                    "extension {extension:?}"
                )))
            }
        };
        clip.origin = None;
        Ok(clip)
    }

    /// Decodes a WAV (`RIFF`/`WAVE`) clip containing PCM or 32-bit IEEE float.
    pub fn decode_wav(data: &[u8]) -> Result<Self, AudioError> {
        decode_wav(data)
    }

    /// Decodes an AIFF or AIFF-C clip containing big-endian PCM.
    ///
    /// AIFF-C `NONE`/`twos` are big-endian, `sowt` is little-endian 16-bit
    /// PCM (a common extension), and 8-bit AIFF samples are signed.
    pub fn decode_aiff(data: &[u8]) -> Result<Self, AudioError> {
        decode_aiff(data)
    }

    /// Decodes an AU/SND clip containing signed linear PCM or 32-bit float.
    pub fn decode_au(data: &[u8]) -> Result<Self, AudioError> {
        decode_au(data)
    }

    /// Number of complete interleaved frames in the clip.
    pub fn frames(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.samples.len() / usize::from(self.channels)
        }
    }

    /// Playback duration, computed without floating point.
    pub fn duration(&self) -> Duration {
        duration_for_frames(self.frames(), self.sample_rate)
    }

    /// Returns `true` when the clip contains no complete frames.
    pub fn is_empty(&self) -> bool {
        self.frames() == 0
    }

    /// Checks structural invariants without modifying the clip.
    ///
    /// Zero samples are valid; zero channels/sample rate and a partial trailing
    /// frame are not. Existing samples are not re-normalized.
    pub fn validate(&self) -> Result<(), AudioError> {
        if self.channels == 0 {
            return Err(AudioError::InvalidFormat(
                "channel count must be greater than zero".into(),
            ));
        }
        if self.sample_rate == 0 {
            return Err(AudioError::InvalidFormat(
                "sample rate must be greater than zero".into(),
            ));
        }
        if !self
            .samples
            .len()
            .is_multiple_of(usize::from(self.channels))
        {
            return Err(AudioError::InvalidFormat(format!(
                "{} samples cannot be split into {} interleaved channels",
                self.samples.len(),
                self.channels
            )));
        }
        Ok(())
    }
}

impl From<&AudioClip> for AudioClip {
    fn from(clip: &AudioClip) -> Self {
        clip.clone()
    }
}

fn duration_for_frames(frames: usize, sample_rate: u32) -> Duration {
    if sample_rate == 0 {
        return Duration::ZERO;
    }
    let rate = u128::from(sample_rate);
    let frames = frames as u128;
    let secs = frames / rate;
    let remainder = frames % rate;
    let nanos = remainder * 1_000_000_000 / rate;
    let secs = u64::try_from(secs).unwrap_or(u64::MAX);
    Duration::new(secs, nanos as u32)
}

fn invalid(message: impl Into<String>) -> AudioError {
    AudioError::InvalidFormat(message.into())
}

fn truncated(message: impl Into<String>) -> AudioError {
    AudioError::Truncated(message.into())
}

fn checked_range<'a>(
    data: &'a [u8],
    start: usize,
    len: usize,
    what: &str,
) -> Result<&'a [u8], AudioError> {
    let end = start
        .checked_add(len)
        .ok_or_else(|| truncated(format!("{what} range overflows")))?;
    data.get(start..end)
        .ok_or_else(|| truncated(format!("{what} extends past end of file")))
}

fn u16_le(data: &[u8], offset: usize) -> Result<u16, AudioError> {
    let bytes = checked_range(data, offset, 2, "16-bit little-endian field")?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn u16_be(data: &[u8], offset: usize) -> Result<u16, AudioError> {
    let bytes = checked_range(data, offset, 2, "16-bit big-endian field")?;
    Ok(u16::from_be_bytes([bytes[0], bytes[1]]))
}

fn u32_le(data: &[u8], offset: usize) -> Result<u32, AudioError> {
    let bytes = checked_range(data, offset, 4, "32-bit little-endian field")?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn u32_be(data: &[u8], offset: usize) -> Result<u32, AudioError> {
    let bytes = checked_range(data, offset, 4, "32-bit big-endian field")?;
    Ok(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn part<'a>(data: &'a [u8], start: usize, len: usize, what: &str) -> Result<&'a [u8], AudioError> {
    checked_range(data, start, len, what)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Endian {
    Little,
    Big,
}

#[derive(Clone, Copy, Debug)]
struct PcmLayout {
    channels: u16,
    sample_width: usize,
    frame_stride: usize,
    endian: Endian,
    unsigned_8_bit: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WavKind {
    Pcm,
    Float,
}

#[derive(Clone, Copy, Debug)]
struct WavFormat {
    kind: WavKind,
    channels: u16,
    sample_rate: u32,
    bits: u16,
    block_align: u16,
}

fn decode_wav(data: &[u8]) -> Result<AudioClip, AudioError> {
    if data.len() < 12 {
        return Err(truncated("WAV header is shorter than 12 bytes"));
    }
    if &data[0..4] != b"RIFF" {
        return Err(invalid("missing RIFF header"));
    }
    if &data[8..12] != b"WAVE" {
        return Err(invalid("RIFF container is not WAVE"));
    }

    let mut format = None;
    let mut data_ranges = Vec::new();
    let mut offset = 12usize;

    while offset.saturating_add(8) <= data.len() {
        let id = &data[offset..offset + 4];
        let size = u32_le(data, offset + 4)? as usize;
        let payload_start = offset + 8;
        let payload_len = if id == b"data" && size == u32::MAX as usize {
            data.len().saturating_sub(payload_start)
        } else {
            size
        };
        let payload_end = payload_start
            .checked_add(payload_len)
            .ok_or_else(|| truncated("WAV chunk length overflows"))?;
        if payload_end > data.len() {
            return Err(truncated(format!(
                "WAV chunk {:?} extends past end of file",
                String::from_utf8_lossy(id)
            )));
        }

        if id == b"fmt " && format.is_none() {
            format = Some(parse_wav_format(&data[payload_start..payload_end])?);
        } else if id == b"data" {
            data_ranges.push((payload_start, payload_end));
        }

        if size == u32::MAX as usize {
            break;
        }
        let padded_end = payload_end
            .checked_add(size & 1)
            .ok_or_else(|| truncated("WAV chunk padding overflows"))?;
        if padded_end > data.len() {
            if payload_end == data.len() {
                break;
            }
            return Err(truncated("WAV chunk is missing its padding byte"));
        }
        offset = padded_end;
    }

    let format = format.ok_or_else(|| invalid("WAV file has no fmt chunk"))?;
    if data_ranges.is_empty() {
        return Err(invalid("WAV file has no data chunk"));
    }

    let bytes_per_sample = usize::from(format.bits / 8);
    let expected_align = usize::from(format.channels)
        .checked_mul(bytes_per_sample)
        .ok_or_else(|| AudioError::TooLarge("WAV frame size overflows".into()))?;
    if usize::from(format.block_align) != expected_align {
        return Err(invalid(format!(
            "WAV block alignment {} does not match {} channels at {} bits",
            format.block_align, format.channels, format.bits
        )));
    }

    let mut total_bytes = 0usize;
    for &(start, end) in &data_ranges {
        total_bytes = total_bytes
            .checked_add(end - start)
            .ok_or_else(|| AudioError::TooLarge("WAV data length overflows".into()))?;
    }
    if !total_bytes.is_multiple_of(expected_align) {
        return Err(truncated("WAV data ends in a partial frame"));
    }
    let frames = total_bytes / expected_align;
    let sample_count = frames
        .checked_mul(usize::from(format.channels))
        .ok_or_else(|| AudioError::TooLarge("WAV sample count overflows".into()))?;
    let mut samples = Vec::new();
    samples.try_reserve_exact(sample_count).map_err(|err| {
        AudioError::TooLarge(format!("cannot allocate {sample_count} samples: {err}"))
    })?;

    for &(start, end) in &data_ranges {
        let chunk = &data[start..end];
        if !chunk.len().is_multiple_of(expected_align) {
            return Err(truncated("WAV data chunk ends in a partial frame"));
        }
        let chunk_frames = chunk.len() / expected_align;
        match format.kind {
            WavKind::Pcm => decode_linear_frames(
                chunk,
                chunk_frames,
                PcmLayout {
                    channels: format.channels,
                    sample_width: bytes_per_sample,
                    frame_stride: expected_align,
                    endian: Endian::Little,
                    unsigned_8_bit: true,
                },
                &mut samples,
            )?,
            WavKind::Float => decode_float_frames(
                chunk,
                chunk_frames,
                format.channels,
                expected_align,
                Endian::Little,
                &mut samples,
            )?,
        }
    }

    Ok(AudioClip {
        samples,
        sample_rate: format.sample_rate,
        channels: format.channels,
        origin: None,
    })
}

fn parse_wav_format(payload: &[u8]) -> Result<WavFormat, AudioError> {
    if payload.len() < 16 {
        return Err(truncated("WAV fmt chunk is shorter than 16 bytes"));
    }
    let mut tag = u16_le(payload, 0)?;
    let channels = u16_le(payload, 2)?;
    let sample_rate = u32_le(payload, 4)?;
    let block_align = u16_le(payload, 12)?;
    let bits = u16_le(payload, 14)?;

    if tag == 0xFFFE {
        if payload.len() < 40 {
            return Err(truncated("WAVE_FORMAT_EXTENSIBLE fmt chunk is too short"));
        }
        tag = u16_le(payload, 24)?;
    }
    let kind = match tag {
        1 => WavKind::Pcm,
        3 => WavKind::Float,
        other => {
            return Err(AudioError::UnsupportedFormat(format!(
                "WAV format tag {other}"
            )))
        }
    };
    if channels == 0 {
        return Err(invalid("WAV channel count must be greater than zero"));
    }
    if sample_rate == 0 {
        return Err(invalid("WAV sample rate must be greater than zero"));
    }
    if bits != 8 && bits != 16 && bits != 24 && bits != 32 {
        return Err(AudioError::UnsupportedFormat(format!(
            "WAV sample width {bits} bits"
        )));
    }
    if kind == WavKind::Float && bits != 32 {
        return Err(AudioError::UnsupportedFormat(format!(
            "WAV IEEE float width {bits} bits"
        )));
    }
    Ok(WavFormat {
        kind,
        channels,
        sample_rate,
        bits,
        block_align,
    })
}

fn decode_linear_frames(
    data: &[u8],
    frames: usize,
    layout: PcmLayout,
    out: &mut Vec<f32>,
) -> Result<(), AudioError> {
    if layout.sample_width == 0 {
        return Err(invalid("sample width must be greater than zero"));
    }
    let needed = frames
        .checked_mul(layout.frame_stride)
        .ok_or_else(|| AudioError::TooLarge("audio frame count overflows".into()))?;
    if data.len() < needed {
        return Err(truncated(format!(
            "need {needed} bytes for {frames} frames but only {} are present",
            data.len()
        )));
    }
    let channels = usize::from(layout.channels);
    for frame in 0..frames {
        let base = frame * layout.frame_stride;
        for channel in 0..channels {
            let offset = base + channel * layout.sample_width;
            let sample = &data[offset..offset + layout.sample_width];
            out.push(linear_sample(
                sample,
                layout.sample_width,
                layout.endian,
                layout.unsigned_8_bit,
            ));
        }
    }
    Ok(())
}

fn linear_sample(bytes: &[u8], width: usize, endian: Endian, unsigned_8_bit: bool) -> f32 {
    match width {
        1 => {
            if unsigned_8_bit {
                (f32::from(bytes[0]) - 128.0) / 128.0
            } else {
                f32::from(bytes[0] as i8) / 128.0
            }
        }
        2 => {
            let value = match endian {
                Endian::Little => i16::from_le_bytes([bytes[0], bytes[1]]),
                Endian::Big => i16::from_be_bytes([bytes[0], bytes[1]]),
            };
            f32::from(value) / 32_768.0
        }
        3 => {
            let raw = match endian {
                Endian::Little => {
                    i32::from(bytes[0]) | (i32::from(bytes[1]) << 8) | (i32::from(bytes[2]) << 16)
                }
                Endian::Big => {
                    i32::from(bytes[2]) | (i32::from(bytes[1]) << 8) | (i32::from(bytes[0]) << 16)
                }
            };
            let signed = if raw & 0x0080_0000 != 0 {
                raw - 0x0100_0000
            } else {
                raw
            };
            signed as f32 / 8_388_608.0
        }
        4 => {
            let value = match endian {
                Endian::Little => i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                Endian::Big => i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            };
            value as f32 / 2_147_483_648.0
        }
        _ => 0.0,
    }
}

fn decode_float_frames(
    data: &[u8],
    frames: usize,
    channels: u16,
    frame_stride: usize,
    endian: Endian,
    out: &mut Vec<f32>,
) -> Result<(), AudioError> {
    let needed = frames
        .checked_mul(frame_stride)
        .ok_or_else(|| AudioError::TooLarge("audio frame count overflows".into()))?;
    if data.len() < needed {
        return Err(truncated(format!(
            "need {needed} bytes for {frames} float frames but only {} are present",
            data.len()
        )));
    }
    let channels = usize::from(channels);
    for frame in 0..frames {
        let base = frame * frame_stride;
        for channel in 0..channels {
            let offset = base + channel * 4;
            let bytes = [
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ];
            let bits = match endian {
                Endian::Little => u32::from_le_bytes(bytes),
                Endian::Big => u32::from_be_bytes(bytes),
            };
            out.push(sanitize_sample(f32::from_bits(bits)));
        }
    }
    Ok(())
}

fn sanitize_sample(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

#[derive(Clone, Copy, Debug)]
struct AiffFormat {
    channels: u16,
    frames: usize,
    bits: u16,
    sample_rate: u32,
    compression: [u8; 4],
}

fn decode_aiff(data: &[u8]) -> Result<AudioClip, AudioError> {
    if data.len() < 12 {
        return Err(truncated("AIFF header is shorter than 12 bytes"));
    }
    if &data[0..4] != b"FORM" {
        return Err(invalid("missing FORM header"));
    }
    let form_kind = &data[8..12];
    let aifc = match form_kind {
        b"AIFF" => false,
        b"AIFC" => true,
        _ => return Err(invalid("FORM container is not AIFF or AIFC")),
    };

    let mut format = None;
    let mut sound_range = None;
    let mut offset = 12usize;
    while offset.saturating_add(8) <= data.len() {
        let id = &data[offset..offset + 4];
        let size = u32_be(data, offset + 4)? as usize;
        let payload_start = offset + 8;
        let payload_end = payload_start
            .checked_add(size)
            .ok_or_else(|| truncated("AIFF chunk length overflows"))?;
        if payload_end > data.len() {
            return Err(truncated(format!(
                "AIFF chunk {:?} extends past end of file",
                String::from_utf8_lossy(id)
            )));
        }
        if id == b"COMM" && format.is_none() {
            format = Some(parse_aiff_format(data, payload_start, payload_end, aifc)?);
        } else if id == b"SSND" && sound_range.is_none() {
            sound_range = Some((payload_start, payload_end));
        }
        let padded_end = payload_end
            .checked_add(size & 1)
            .ok_or_else(|| truncated("AIFF chunk padding overflows"))?;
        if padded_end > data.len() {
            if payload_end == data.len() {
                break;
            }
            return Err(truncated("AIFF chunk is missing its padding byte"));
        }
        offset = padded_end;
    }

    let format = format.ok_or_else(|| invalid("AIFF file has no COMM chunk"))?;
    let sample_width = usize::from(format.bits / 8);
    let frame_stride = usize::from(format.channels)
        .checked_mul(sample_width)
        .ok_or_else(|| AudioError::TooLarge("AIFF frame size overflows".into()))?;
    let needed = format
        .frames
        .checked_mul(frame_stride)
        .ok_or_else(|| AudioError::TooLarge("AIFF sample count overflows".into()))?;
    let sample_count = format
        .frames
        .checked_mul(usize::from(format.channels))
        .ok_or_else(|| AudioError::TooLarge("AIFF sample count overflows".into()))?;

    let mut samples = Vec::new();

    if let Some((payload_start, payload_end)) = sound_range {
        if payload_end - payload_start < 8 {
            return Err(truncated("AIFF SSND chunk is shorter than its header"));
        }
        let data_offset = u32_be(data, payload_start)? as usize;
        let sound_start = payload_start
            .checked_add(8)
            .and_then(|value| value.checked_add(data_offset))
            .ok_or_else(|| truncated("AIFF SSND offset overflows"))?;
        if sound_start > payload_end {
            return Err(truncated("AIFF SSND offset points past its chunk"));
        }
        let sound = &data[sound_start..payload_end];
        if sound.len() < needed {
            return Err(truncated(format!(
                "AIFF needs {needed} sample bytes but SSND contains {}",
                sound.len()
            )));
        }
        samples.try_reserve_exact(sample_count).map_err(|err| {
            AudioError::TooLarge(format!("cannot allocate {sample_count} samples: {err}"))
        })?;
        let (endian, unsigned_8_bit) = match &format.compression {
            b"NONE" | b"twos" => (Endian::Big, false),
            b"sowt" => (Endian::Little, false),
            other => {
                return Err(AudioError::UnsupportedFormat(format!(
                    "AIFF-C compression {:?}",
                    String::from_utf8_lossy(other)
                )))
            }
        };
        decode_linear_frames(
            &sound[..needed],
            format.frames,
            PcmLayout {
                channels: format.channels,
                sample_width,
                frame_stride,
                endian,
                unsigned_8_bit,
            },
            &mut samples,
        )?;
    } else if format.frames != 0 {
        return Err(invalid("AIFF file has no SSND chunk"));
    }

    Ok(AudioClip {
        samples,
        sample_rate: format.sample_rate,
        channels: format.channels,
        origin: None,
    })
}

fn parse_aiff_format(
    data: &[u8],
    payload_start: usize,
    payload_end: usize,
    aifc: bool,
) -> Result<AiffFormat, AudioError> {
    let minimum = if aifc { 22 } else { 18 };
    if payload_end.saturating_sub(payload_start) < minimum {
        return Err(truncated(format!(
            "AIFF{} COMM chunk is shorter than {minimum} bytes",
            if aifc { "-C" } else { "" }
        )));
    }
    let channels = u16_be(data, payload_start)?;
    let frames = u32_be(data, payload_start + 2)? as usize;
    let bits = u16_be(data, payload_start + 6)?;
    let rate_bytes = part(data, payload_start + 8, 10, "AIFF sample rate")?;
    let sample_rate = parse_extended_80(rate_bytes)?;
    let compression = if aifc {
        let bytes = part(data, payload_start + 18, 4, "AIFF-C compression type")?;
        [bytes[0], bytes[1], bytes[2], bytes[3]]
    } else {
        *b"NONE"
    };

    if channels == 0 {
        return Err(invalid("AIFF channel count must be greater than zero"));
    }
    if bits != 8 && bits != 16 && bits != 24 && bits != 32 {
        return Err(AudioError::UnsupportedFormat(format!(
            "AIFF sample width {bits} bits"
        )));
    }
    if compression == *b"sowt" && bits != 16 {
        return Err(AudioError::UnsupportedFormat(
            "AIFF-C sowt compression is only supported at 16 bits".into(),
        ));
    }
    if aifc && !matches!(&compression, b"NONE" | b"twos" | b"sowt") {
        return Err(AudioError::UnsupportedFormat(format!(
            "AIFF-C compression {:?}",
            String::from_utf8_lossy(&compression)
        )));
    }

    Ok(AiffFormat {
        channels,
        frames,
        bits,
        sample_rate,
        compression,
    })
}

fn parse_extended_80(bytes: &[u8]) -> Result<u32, AudioError> {
    if bytes.len() < 10 {
        return Err(truncated("80-bit extended float is shorter than 10 bytes"));
    }
    let sign = bytes[0] & 0x80 != 0;
    let exponent = (u16::from(bytes[0] & 0x7F) << 8) | u16::from(bytes[1]);
    let mantissa = u64::from_be_bytes([
        bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7], bytes[8], bytes[9],
    ]);
    if exponent == 0 && mantissa == 0 {
        return Err(invalid("AIFF sample rate is zero"));
    }
    if exponent == 0x7FFF {
        return Err(invalid("AIFF sample rate is not finite"));
    }
    let value = mantissa as f64 * 2f64.powi(i32::from(exponent) - 16_383 - 63);
    if sign || !value.is_finite() || value <= 0.0 || value > f64::from(u32::MAX) {
        return Err(invalid("AIFF sample rate is not a positive u32"));
    }
    let rounded = value.round();
    if rounded < 1.0 {
        return Err(invalid("AIFF sample rate rounds to zero"));
    }
    Ok(rounded as u32)
}

fn decode_au(data: &[u8]) -> Result<AudioClip, AudioError> {
    if data.len() < 24 {
        return Err(truncated("AU header is shorter than 24 bytes"));
    }
    if &data[0..4] != b".snd" {
        return Err(invalid("missing .snd header"));
    }
    let data_offset = u32_be(data, 4)? as usize;
    let declared_size = u32_be(data, 8)?;
    let encoding = u32_be(data, 12)?;
    let sample_rate = u32_be(data, 16)?;
    let channels_u32 = u32_be(data, 20)?;

    if data_offset < 24 {
        return Err(invalid("AU data offset points inside the header"));
    }
    if data_offset > data.len() {
        return Err(truncated("AU data offset points past end of file"));
    }
    let channels = u16::try_from(channels_u32)
        .map_err(|_| AudioError::UnsupportedFormat(format!("{channels_u32} AU channels")))?;
    if channels == 0 {
        return Err(invalid("AU channel count must be greater than zero"));
    }
    if sample_rate == 0 {
        return Err(invalid("AU sample rate must be greater than zero"));
    }
    let available = data.len() - data_offset;
    let data_size = if declared_size == u32::MAX {
        available
    } else {
        let declared = declared_size as usize;
        if declared > available {
            return Err(truncated(format!(
                "AU header declares {declared} data bytes but only {available} are present"
            )));
        }
        declared
    };

    let (sample_width, endian, is_float, unsigned_8_bit) = match encoding {
        2 => (1usize, Endian::Big, false, false),
        3 => (2usize, Endian::Big, false, false),
        4 => (3usize, Endian::Big, false, false),
        5 => (4usize, Endian::Big, false, false),
        6 => (4usize, Endian::Big, true, false),
        other => {
            return Err(AudioError::UnsupportedFormat(format!(
                "AU encoding {other}"
            )))
        }
    };
    let frame_stride = usize::from(channels)
        .checked_mul(sample_width)
        .ok_or_else(|| AudioError::TooLarge("AU frame size overflows".into()))?;
    let sound = &data[data_offset..data_offset + data_size];
    if !sound.len().is_multiple_of(frame_stride) {
        return Err(truncated("AU data ends in a partial frame"));
    }
    let frames = sound.len() / frame_stride;
    let sample_count = frames
        .checked_mul(usize::from(channels))
        .ok_or_else(|| AudioError::TooLarge("AU sample count overflows".into()))?;
    let mut samples = Vec::new();
    samples.try_reserve_exact(sample_count).map_err(|err| {
        AudioError::TooLarge(format!("cannot allocate {sample_count} samples: {err}"))
    })?;

    if is_float {
        decode_float_frames(sound, frames, channels, frame_stride, endian, &mut samples)?;
    } else {
        decode_linear_frames(
            sound,
            frames,
            PcmLayout {
                channels,
                sample_width,
                frame_stride,
                endian,
                unsigned_8_bit,
            },
            &mut samples,
        )?;
    }

    Ok(AudioClip {
        samples,
        sample_rate,
        channels,
        origin: None,
    })
}

/// Smoothing applied to beat brightness.
///
/// The variants follow CosPlay's `CPBeatShaderSmoothing`:
/// - [`SmoothNone`](Self::SmoothNone) is binary: below threshold is `0`,
///   at/above threshold is `1`.
/// - [`SmoothUp`](Self::SmoothUp) is `0` below threshold and the continuous
///   amplitude at/above it.
/// - [`SmoothDown`](Self::SmoothDown) is the continuous amplitude below the
///   threshold and `1` at/above it.
/// - [`SmoothBoth`](Self::SmoothBoth) always uses the continuous amplitude.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Smoothing {
    /// Binary output.
    SmoothNone,
    /// Continuous above the threshold, zero below it.
    #[default]
    SmoothUp,
    /// Continuous below the threshold, one at/above it.
    SmoothDown,
    /// Continuous output everywhere.
    SmoothBoth,
}

/// Alias matching the wording used by CosPlay's beat shader documentation.
pub type BeatSmoothing = Smoothing;

pub use Smoothing::{SmoothBoth, SmoothDown, SmoothNone, SmoothUp};

/// Precomputed beat envelope for a clip.
///
/// The clip is split into approximately 10 ms windows. Each window's RMS
/// amplitude is min/max normalized to `0.0..=1.0`. At a given playback time the
/// threshold is the RMS of up to the configured tail of preceding windows
/// (default 2000 ms), multiplied by [`Self::threshold_gain`]. On the first pass
/// the preceding window history starts empty, matching a live sliding queue;
/// later passes wrap around the clip.
///
/// A one-window clip whose signal is non-silent normalizes to `1.0`; a
/// completely silent clip stays at `0.0`. Silent windows always report `0.0`.
#[derive(Clone, Debug)]
pub struct BeatEnvelope {
    windows: Vec<f32>,
    duration: Duration,
    window_duration: Duration,
    threshold_tail: Duration,
    threshold_gain: f32,
    smoothing: Smoothing,
}

impl BeatEnvelope {
    /// Analyzes `clip` with the default tail, gain, and smoothing.
    pub fn new(clip: impl std::borrow::Borrow<AudioClip>) -> Self {
        Self::analyze(clip.borrow())
    }

    /// Analyzes `clip` with explicit options.
    pub fn analyze_with(
        clip: &AudioClip,
        smoothing: Smoothing,
        threshold_gain: f32,
        threshold_tail: Duration,
    ) -> Self {
        Self::analyze(clip).with_options(smoothing, threshold_gain, threshold_tail)
    }

    /// Analyzes `clip` with the default options.
    pub fn analyze(clip: &AudioClip) -> Self {
        let mut envelope = Self {
            windows: Vec::new(),
            duration: clip.duration(),
            window_duration: Duration::from_millis(u64::from(BEAT_WINDOW_MS)),
            threshold_tail: Duration::from_millis(DEFAULT_THRESHOLD_TAIL_MS),
            threshold_gain: DEFAULT_THRESHOLD_GAIN,
            smoothing: Smoothing::default(),
        };
        if clip.channels == 0 || clip.sample_rate == 0 || clip.frames() == 0 {
            return envelope;
        }

        let window_frames = usize::try_from(
            (u64::from(clip.sample_rate) * u64::from(BEAT_WINDOW_MS)).div_ceil(1_000),
        )
        .unwrap_or(usize::MAX)
        .max(1);
        let frames = clip.frames();
        let window_count = frames.div_ceil(window_frames);
        let channels = usize::from(clip.channels);
        let mut amplitudes = Vec::new();
        if amplitudes.try_reserve_exact(window_count).is_err() {
            return envelope;
        }

        for window in 0..window_count {
            let start_frame = window * window_frames;
            let end_frame = (start_frame + window_frames).min(frames);
            let start = start_frame.saturating_mul(channels);
            let end = end_frame.saturating_mul(channels).min(clip.samples.len());
            let mut sum = 0.0f64;
            let mut count = 0usize;
            for &sample in &clip.samples[start..end] {
                let sample = if sample.is_finite() { sample } else { 0.0 };
                sum += f64::from(sample) * f64::from(sample);
                count += 1;
            }
            let rms = if count == 0 {
                0.0
            } else {
                (sum / count as f64).sqrt() as f32
            };
            amplitudes.push(rms);
        }

        let mut minimum = f32::INFINITY;
        let mut maximum = f32::NEG_INFINITY;
        for &amplitude in &amplitudes {
            minimum = minimum.min(amplitude);
            maximum = maximum.max(amplitude);
        }
        let gain = maximum - minimum;
        if gain > 1.0e-9 {
            for amplitude in &mut amplitudes {
                *amplitude = ((*amplitude - minimum) / gain).clamp(0.0, 1.0);
            }
        } else if maximum > 1.0e-9 {
            // A constant non-silent window has no dynamic range to normalize;
            // treat it as fully lit rather than dividing by zero.
            amplitudes.fill(1.0);
        } else {
            amplitudes.fill(0.0);
        }

        envelope.window_duration = duration_for_frames(window_frames, clip.sample_rate);
        envelope.windows = amplitudes;
        envelope
    }

    /// Number of analyzed amplitude windows.
    pub fn window_count(&self) -> usize {
        self.windows.len()
    }

    /// Nominal duration of one analysis window (~10 ms).
    pub fn window_duration(&self) -> Duration {
        self.window_duration
    }

    /// Clip duration used for wrapping.
    pub fn duration(&self) -> Duration {
        self.duration
    }

    /// Returns `true` when there are no windows to sample.
    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }

    /// The normalized per-window amplitudes, mainly useful for diagnostics and
    /// tests.
    pub fn normalized_amplitudes(&self) -> &[f32] {
        &self.windows
    }

    /// Current threshold multiplier.
    pub fn threshold_gain(&self) -> f32 {
        self.threshold_gain
    }

    /// Sets the threshold multiplier. Non-finite or negative values reset to
    /// the default of `1.0`.
    pub fn set_threshold_gain(&mut self, gain: f32) {
        self.threshold_gain = if gain.is_finite() && gain >= 0.0 {
            gain
        } else {
            DEFAULT_THRESHOLD_GAIN
        };
    }

    /// Builder form of [`Self::set_threshold_gain`].
    pub fn with_threshold_gain(mut self, gain: f32) -> Self {
        self.set_threshold_gain(gain);
        self
    }

    /// Current sliding threshold window.
    pub fn threshold_tail(&self) -> Duration {
        self.threshold_tail
    }

    /// Sets the sliding RMS threshold window. A zero tail disables the
    /// threshold (it is treated as zero).
    pub fn set_threshold_tail(&mut self, tail: Duration) {
        self.threshold_tail = tail;
    }

    /// Builder form of [`Self::set_threshold_tail`].
    pub fn with_threshold_tail(mut self, tail: Duration) -> Self {
        self.set_threshold_tail(tail);
        self
    }

    /// Sets the sliding threshold window in milliseconds.
    pub fn set_threshold_tail_ms(&mut self, tail_ms: u64) {
        self.threshold_tail = Duration::from_millis(tail_ms);
    }

    /// Builder form of [`Self::set_threshold_tail_ms`].
    pub fn with_threshold_tail_ms(mut self, tail_ms: u64) -> Self {
        self.set_threshold_tail_ms(tail_ms);
        self
    }

    /// Current smoothing mode.
    pub fn smoothing(&self) -> Smoothing {
        self.smoothing
    }

    /// Sets the smoothing mode.
    pub fn set_smoothing(&mut self, smoothing: Smoothing) {
        self.smoothing = smoothing;
    }

    /// Builder form of [`Self::set_smoothing`].
    pub fn with_smoothing(mut self, smoothing: Smoothing) -> Self {
        self.set_smoothing(smoothing);
        self
    }

    /// Builder convenience for all options at once.
    pub fn with_options(
        mut self,
        smoothing: Smoothing,
        threshold_gain: f32,
        threshold_tail: Duration,
    ) -> Self {
        self.smoothing = smoothing;
        self.set_threshold_gain(threshold_gain);
        self.threshold_tail = threshold_tail;
        self
    }

    /// Brightness in `0.0..=1.0` at the given playback time.
    ///
    /// Time wraps at the clip duration, so a looping sound stays in sync. Times
    /// beyond the clip's end therefore restart from the beginning. Empty clips
    /// always return `0.0`.
    pub fn brightness_at(&self, elapsed: Duration) -> f32 {
        let window_count = self.windows.len();
        if window_count == 0 {
            return 0.0;
        }
        let duration_ns = self.duration.as_nanos();
        let elapsed_ns = elapsed.as_nanos();
        let cycle = elapsed_ns.checked_div(duration_ns).unwrap_or(0);
        let phase_ns = elapsed_ns.checked_rem(duration_ns).unwrap_or(0);
        let window_ns = u128::from(BEAT_WINDOW_MS) * 1_000_000;
        let index = ((phase_ns / window_ns) as usize).min(window_count - 1);
        let amplitude = self.windows[index];
        if !amplitude.is_finite() || amplitude <= 0.0 {
            return 0.0;
        }
        let threshold = self.threshold_before(index, cycle);
        let value = match self.smoothing {
            Smoothing::SmoothNone => {
                if amplitude < threshold {
                    0.0
                } else {
                    1.0
                }
            }
            Smoothing::SmoothUp => {
                if amplitude < threshold {
                    0.0
                } else {
                    amplitude
                }
            }
            Smoothing::SmoothDown => {
                if amplitude < threshold {
                    amplitude
                } else {
                    1.0
                }
            }
            Smoothing::SmoothBoth => amplitude,
        };
        value.clamp(0.0, 1.0)
    }

    fn threshold_before(&self, index: usize, cycle: u128) -> f32 {
        let window_count = self.windows.len();
        if window_count == 0 {
            return 0.0;
        }
        let window_ns = u128::from(BEAT_WINDOW_MS) * 1_000_000;
        let tail_windows =
            usize::try_from(self.threshold_tail.as_nanos() / window_ns).unwrap_or(usize::MAX);
        if tail_windows == 0 {
            return 0.0;
        }
        let count = if cycle == 0 {
            tail_windows.min(index)
        } else {
            tail_windows.min(window_count)
        };
        if count == 0 {
            return 0.0;
        }
        let mut sum = 0.0f64;
        for back in 1..=count {
            let previous = (index + window_count - back % window_count) % window_count;
            let value = f64::from(self.windows[previous]);
            sum += value * value;
        }
        let threshold = (sum / count as f64).sqrt() as f32;
        threshold * self.threshold_gain
    }
}

fn clamp_volume(value: f32) -> f32 {
    if value.is_nan() {
        0.0
    } else {
        value.clamp(0.0, 2.0)
    }
}

fn clamp_speed(value: f32) -> f32 {
    if value.is_nan() {
        1.0
    } else {
        value.clamp(0.1, 4.0)
    }
}

#[cfg(feature = "audio")]
fn frames_for_duration(duration: Duration, sample_rate: u32) -> usize {
    let frames = duration.as_nanos().saturating_mul(u128::from(sample_rate)) / 1_000_000_000;
    usize::try_from(frames).unwrap_or(usize::MAX)
}

#[cfg(feature = "audio")]
fn wrap_duration(value: Duration, modulus: Duration) -> Duration {
    if modulus.is_zero() {
        return Duration::ZERO;
    }
    let nanos = value.as_nanos() % modulus.as_nanos();
    let secs = u64::try_from(nanos / 1_000_000_000).unwrap_or(u64::MAX);
    let subsec = (nanos % 1_000_000_000) as u32;
    Duration::new(secs, subsec)
}

/// A decoded sound plus optional playback state.
///
/// Creating a `Sound` (and [`Sound::new`] in particular) never opens an audio
/// device. The first call that actually starts playback opens the default
/// output stream; this keeps decoding and headless tests device-free.
///
/// `rodio::OutputStream` is not guaranteed to be `Send`, so `Sound` is
/// deliberately single-threaded. Keep it on the thread that created it, or use
/// [`SoundRegistry`] to own a group of sounds on that thread.
pub struct Sound {
    clip: AudioClip,
    beat: BeatEnvelope,
    volume: f32,
    speed: f32,
    looping: bool,
    position: Duration,
    disposed: bool,
    // The sink must be dropped before the stream that feeds it.
    #[cfg(feature = "audio")]
    sink: Option<Sink>,
    #[cfg(feature = "audio")]
    stream: Option<rodio::OutputStream>,
    #[cfg(feature = "audio")]
    source_offset: Duration,
}

impl Sound {
    /// Wraps a decoded clip. No device is opened.
    pub fn new(clip: impl Into<AudioClip>) -> Self {
        let clip = clip.into();
        let beat = BeatEnvelope::new(&clip);
        Self {
            clip,
            beat,
            volume: 1.0,
            speed: 1.0,
            looping: false,
            position: Duration::ZERO,
            disposed: false,
            #[cfg(feature = "audio")]
            sink: None,
            #[cfg(feature = "audio")]
            stream: None,
            #[cfg(feature = "audio")]
            source_offset: Duration::ZERO,
        }
    }

    /// The underlying clip.
    pub fn clip(&self) -> &AudioClip {
        &self.clip
    }

    /// The clip's precomputed beat envelope.
    pub fn beat_envelope(&self) -> &BeatEnvelope {
        &self.beat
    }

    /// Whether playback is currently running.
    pub fn is_playing(&self) -> bool {
        if self.disposed {
            return false;
        }
        #[cfg(feature = "audio")]
        {
            self.sink
                .as_ref()
                .is_some_and(|sink| !sink.empty() && !sink.is_paused())
        }
        #[cfg(not(feature = "audio"))]
        {
            false
        }
    }

    /// Starts or resumes playback. Opens the default output device on first
    /// use and returns [`AudioError::Playback`] if that fails or if this build
    /// does not have the `audio` feature.
    pub fn play(&mut self) -> Result<(), AudioError> {
        if self.disposed {
            return Err(AudioError::Playback("sound has been disposed".into()));
        }
        #[cfg(feature = "audio")]
        {
            if let Some(sink) = &self.sink {
                if !sink.empty() {
                    sink.play();
                    return Ok(());
                }
                // Natural end: like CosPlay, rewind before playing again.
                self.position = Duration::ZERO;
            }
            let duration = self.clip.duration();
            if !self.looping && !duration.is_zero() && self.position >= duration {
                self.position = Duration::ZERO;
            }
            self.start_source_at(self.position, false)
        }
        #[cfg(not(feature = "audio"))]
        {
            Err(AudioError::Playback(
                "rebuild with the `audio` feature to enable playback".into(),
            ))
        }
    }

    /// Stops, rewinds, and starts playback from the beginning.
    pub fn replay(&mut self) -> Result<(), AudioError> {
        self.stop();
        self.play()
    }

    /// Sets whether playback loops. If a sound is already running, its source
    /// is rebuilt at the current position so the change takes effect now.
    pub fn loop_clip(&mut self, looping: bool) {
        if self.looping == looping {
            return;
        }
        self.looping = looping;
        #[cfg(feature = "audio")]
        {
            let active = self.sink.as_ref().is_some_and(|sink| !sink.empty());
            if active {
                let paused = self.sink.as_ref().is_some_and(Sink::is_paused);
                let position = self.current_time();
                if self.start_source_at(position, paused).is_err() {
                    self.stop();
                }
            }
        }
    }

    /// Whether looping is enabled for future playback.
    pub fn is_looping(&self) -> bool {
        self.looping
    }

    /// Pauses current playback.
    pub fn pause(&mut self) {
        #[cfg(feature = "audio")]
        if let Some(sink) = &self.sink {
            sink.pause();
        }
    }

    /// Resumes a paused sound. Does nothing if nothing is active.
    pub fn resume(&mut self) {
        #[cfg(feature = "audio")]
        if let Some(sink) = &self.sink {
            if !sink.empty() {
                sink.play();
            }
        }
    }

    /// Stops playback and rewinds to the start. The output stream is retained
    /// for later playback, but the sink (and its queued audio) is dropped.
    pub fn stop(&mut self) {
        #[cfg(feature = "audio")]
        if let Some(sink) = self.sink.take() {
            sink.stop();
        }
        self.position = Duration::ZERO;
    }

    /// Pauses a playing sound, otherwise starts/restarts it.
    pub fn toggle(&mut self) -> Result<(), AudioError> {
        if self.is_playing() {
            self.pause();
            Ok(())
        } else {
            self.play()
        }
    }

    /// Sets volume, clamped to `0.0..=2.0`. NaN becomes `0.0`.
    pub fn set_volume(&mut self, volume: f32) {
        self.volume = clamp_volume(volume);
        #[cfg(feature = "audio")]
        if let Some(sink) = &self.sink {
            sink.set_volume(self.volume);
        }
    }

    /// Current volume.
    pub fn volume(&self) -> f32 {
        self.volume
    }

    /// Sets playback speed, clamped to `0.1..=4.0`. NaN becomes `1.0`.
    ///
    /// Rodio changes pitch as well as rate; it does not time-stretch.
    pub fn set_speed(&mut self, speed: f32) {
        self.speed = clamp_speed(speed);
        #[cfg(feature = "audio")]
        if let Some(sink) = &self.sink {
            sink.set_speed(self.speed);
        }
    }

    /// Current playback speed.
    pub fn speed(&self) -> f32 {
        self.speed
    }

    /// Seeks to `elapsed`, clamped to the clip duration.
    ///
    /// Seeking a running sound recreates its in-memory source at the requested
    /// position, preserving pause state, volume, and speed.
    pub fn seek(&mut self, elapsed: Duration) -> Result<(), AudioError> {
        if self.disposed {
            return Err(AudioError::Playback("sound has been disposed".into()));
        }
        let target = elapsed.min(self.clip.duration());
        #[cfg(feature = "audio")]
        {
            let paused = self
                .sink
                .as_ref()
                .filter(|sink| !sink.empty())
                .map(Sink::is_paused);
            if let Some(paused) = paused {
                self.start_source_at(target, paused)?;
            } else {
                // Drop an ended/empty sink so a later play starts at `target`
                // rather than treating it as an automatic rewind.
                self.sink = None;
                self.position = target;
            }
        }
        #[cfg(not(feature = "audio"))]
        {
            self.position = target;
        }
        Ok(())
    }

    /// Seeks to the start.
    pub fn rewind(&mut self) -> Result<(), AudioError> {
        self.seek(Duration::ZERO)
    }

    /// Current playback position. After a sound ends naturally this is
    /// reported as zero, matching CosPlay's automatic rewind.
    pub fn current_time(&self) -> Duration {
        #[cfg(feature = "audio")]
        {
            if let Some(sink) = &self.sink {
                if sink.empty() {
                    return Duration::ZERO;
                }
                let duration = self.clip.duration();
                let elapsed = self.source_offset.saturating_add(sink.get_pos());
                if duration.is_zero() {
                    return Duration::ZERO;
                }
                return if self.looping {
                    wrap_duration(elapsed, duration)
                } else {
                    elapsed.min(duration)
                };
            }
        }
        self.position.min(self.clip.duration())
    }

    /// Total clip duration.
    pub fn duration(&self) -> Duration {
        self.clip.duration()
    }

    /// Stops playback and releases the output stream. The `Sound` remains
    /// queryable but cannot be played again.
    pub fn dispose(&mut self) {
        if self.disposed {
            return;
        }
        #[cfg(feature = "audio")]
        {
            if let Some(sink) = self.sink.take() {
                sink.stop();
            }
            self.stream = None;
        }
        self.position = Duration::ZERO;
        self.disposed = true;
    }

    /// Whether [`Sound::dispose`] has been called.
    pub fn is_disposed(&self) -> bool {
        self.disposed
    }

    #[cfg(feature = "audio")]
    fn start_source_at(&mut self, position: Duration, paused: bool) -> Result<(), AudioError> {
        if self.clip.channels == 0 || self.clip.sample_rate == 0 {
            return Err(AudioError::InvalidFormat(
                "cannot play a clip with zero channels or sample rate".into(),
            ));
        }
        if self.stream.is_none() {
            let mut stream = rodio::OutputStreamBuilder::open_default_stream().map_err(|err| {
                AudioError::Playback(format!("failed to open audio output: {err}"))
            })?;
            // CosPlay owns lifecycle management; the default drop log is noise.
            stream.log_on_drop(false);
            self.stream = Some(stream);
        }
        let duration = self.clip.duration();
        let start = if self.looping {
            wrap_duration(position.min(duration), duration)
        } else {
            position.min(duration)
        };
        let sink = {
            let stream = self
                .stream
                .as_ref()
                .ok_or_else(|| AudioError::Playback("audio output stream is unavailable".into()))?;
            Sink::connect_new(stream.mixer())
        };
        sink.set_volume(self.volume);
        sink.set_speed(self.speed);
        if self.looping {
            sink.append(LoopBuffer::new(
                Arc::from(self.clip.samples.clone()),
                self.clip.channels,
                self.clip.sample_rate,
                start,
            ));
        } else {
            let frame = frames_for_duration(start, self.clip.sample_rate).min(self.clip.frames());
            let sample_index = frame.saturating_mul(usize::from(self.clip.channels));
            let data = self
                .clip
                .samples
                .get(sample_index..)
                .unwrap_or(&[])
                .to_vec();
            sink.append(SamplesBuffer::new(
                self.clip.channels,
                self.clip.sample_rate,
                data,
            ));
        }
        if paused {
            sink.pause();
        }
        self.source_offset = start;
        self.position = start;
        self.sink = Some(sink);
        Ok(())
    }
}

#[cfg(feature = "audio")]
struct LoopBuffer {
    data: Arc<[f32]>,
    channels: u16,
    sample_rate: u32,
    frame_count: usize,
    position: usize,
}

#[cfg(feature = "audio")]
impl LoopBuffer {
    fn new(data: Arc<[f32]>, channels: u16, sample_rate: u32, start: Duration) -> Self {
        let frame_count = if channels == 0 {
            0
        } else {
            data.len() / usize::from(channels)
        };
        let start_frame = if frame_count == 0 {
            0
        } else {
            frames_for_duration(start, sample_rate) % frame_count
        };
        Self {
            data,
            channels,
            sample_rate,
            frame_count,
            position: start_frame.saturating_mul(usize::from(channels)),
        }
    }
}

#[cfg(feature = "audio")]
impl Iterator for LoopBuffer {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.frame_count == 0 || self.data.is_empty() {
            return None;
        }
        let sample = *self.data.get(self.position)?;
        self.position += 1;
        if self.position >= self.data.len() {
            self.position = 0;
        }
        Some(sample)
    }
}

#[cfg(feature = "audio")]
impl Source for LoopBuffer {
    fn current_span_len(&self) -> Option<usize> {
        None
    }

    fn channels(&self) -> rodio::ChannelCount {
        self.channels
    }

    fn sample_rate(&self) -> rodio::SampleRate {
        self.sample_rate
    }

    fn total_duration(&self) -> Option<Duration> {
        None
    }

    fn try_seek(&mut self, pos: Duration) -> Result<(), rodio::source::SeekError> {
        if self.frame_count != 0 {
            let frame = frames_for_duration(pos, self.sample_rate) % self.frame_count;
            self.position = frame.saturating_mul(usize::from(self.channels));
        }
        Ok(())
    }
}

/// Owned registry of sounds, keyed by stable `u64` IDs.
///
/// CosPlay keeps a process-global track list. A safe Rust equivalent cannot
/// store a non-`Sync` `Sink` in a mutable static without `unsafe`, so this type
/// makes ownership explicit: create one registry on the audio thread and share
/// it through normal Rust ownership. IDs start at `1` and are never reused by
/// [`SoundRegistry::insert`] during a registry's lifetime.
#[derive(Default)]
pub struct SoundRegistry {
    next_id: u64,
    sounds: BTreeMap<u64, Sound>,
}

impl SoundRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self {
            next_id: 1,
            sounds: BTreeMap::new(),
        }
    }

    /// Inserts a sound and returns its new ID.
    pub fn insert(&mut self, sound: Sound) -> u64 {
        let mut id = self.next_id.max(1);
        while self.sounds.contains_key(&id) {
            id = id.wrapping_add(1).max(1);
        }
        self.next_id = id.wrapping_add(1).max(1);
        self.sounds.insert(id, sound);
        id
    }

    /// Alias for [`SoundRegistry::insert`].
    pub fn add(&mut self, sound: Sound) -> u64 {
        self.insert(sound)
    }

    /// Inserts with an explicit ID, returning any sound that was replaced.
    pub fn insert_with_id(&mut self, id: u64, sound: Sound) -> Option<Sound> {
        self.sounds.insert(id, sound)
    }

    /// Removes a sound, returning it if present.
    pub fn remove(&mut self, id: u64) -> Option<Sound> {
        self.sounds.remove(&id)
    }

    /// Alias for [`SoundRegistry::remove`].
    pub fn remove_id(&mut self, id: u64) -> Option<Sound> {
        self.remove(id)
    }

    /// Borrows a sound by ID.
    pub fn get(&self, id: u64) -> Option<&Sound> {
        self.sounds.get(&id)
    }

    /// Mutably borrows a sound by ID.
    pub fn get_mut(&mut self, id: u64) -> Option<&mut Sound> {
        self.sounds.get_mut(&id)
    }

    /// Tests whether an ID exists.
    pub fn contains(&self, id: u64) -> bool {
        self.sounds.contains_key(&id)
    }

    /// Number of registered sounds.
    pub fn len(&self) -> usize {
        self.sounds.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.sounds.is_empty()
    }

    /// Iterates over `(id, sound)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (u64, &Sound)> + '_ {
        self.sounds.iter().map(|(&id, sound)| (id, sound))
    }

    /// Mutably iterates over `(id, sound)` pairs.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (u64, &mut Sound)> + '_ {
        self.sounds.iter_mut().map(|(&id, sound)| (id, sound))
    }

    /// Calls `f` for every sound in ID order.
    pub fn foreach(&mut self, mut f: impl FnMut(&mut Sound)) {
        for sound in self.sounds.values_mut() {
            f(sound);
        }
    }

    /// Calls `f` for every `(id, sound)` pair in ID order.
    pub fn foreach_with_id(&mut self, mut f: impl FnMut(u64, &mut Sound)) {
        for (&id, sound) in &mut self.sounds {
            f(id, sound);
        }
    }

    /// Stops and rewinds every registered sound.
    pub fn stop_all(&mut self) {
        for sound in self.sounds.values_mut() {
            sound.stop();
        }
    }

    /// Disposes every registered sound and empties the registry.
    pub fn dispose_all(&mut self) {
        for sound in self.sounds.values_mut() {
            sound.dispose();
        }
        self.sounds.clear();
    }

    /// Starts a registered sound.
    pub fn play_id(&mut self, id: u64) -> Result<(), AudioError> {
        self.sound_mut(id)?.play()
    }

    /// Replays a registered sound from the start.
    pub fn replay_id(&mut self, id: u64) -> Result<(), AudioError> {
        self.sound_mut(id)?.replay()
    }

    /// Toggles a registered sound.
    pub fn toggle_id(&mut self, id: u64) -> Result<(), AudioError> {
        self.sound_mut(id)?.toggle()
    }

    /// Pauses a registered sound.
    pub fn pause_id(&mut self, id: u64) -> Result<(), AudioError> {
        self.sound_mut(id)?.pause();
        Ok(())
    }

    /// Resumes a registered sound.
    pub fn resume_id(&mut self, id: u64) -> Result<(), AudioError> {
        self.sound_mut(id)?.resume();
        Ok(())
    }

    /// Stops a registered sound.
    pub fn stop_id(&mut self, id: u64) -> Result<(), AudioError> {
        self.sound_mut(id)?.stop();
        Ok(())
    }

    /// Seeks a registered sound.
    pub fn seek_id(&mut self, id: u64, elapsed: Duration) -> Result<(), AudioError> {
        self.sound_mut(id)?.seek(elapsed)
    }

    /// Rewinds a registered sound.
    pub fn rewind_id(&mut self, id: u64) -> Result<(), AudioError> {
        self.sound_mut(id)?.rewind()
    }

    /// Sets a registered sound's volume.
    pub fn set_volume_id(&mut self, id: u64, volume: f32) -> Result<(), AudioError> {
        self.sound_mut(id)?.set_volume(volume);
        Ok(())
    }

    /// Gets a registered sound's volume.
    pub fn volume_id(&self, id: u64) -> Result<f32, AudioError> {
        Ok(self.sound(id)?.volume())
    }

    /// Sets a registered sound's playback speed.
    pub fn set_speed_id(&mut self, id: u64, speed: f32) -> Result<(), AudioError> {
        self.sound_mut(id)?.set_speed(speed);
        Ok(())
    }

    /// Gets a registered sound's current playback time.
    pub fn current_time_id(&self, id: u64) -> Result<Duration, AudioError> {
        Ok(self.sound(id)?.current_time())
    }

    /// Gets a registered sound's clip duration.
    pub fn duration_id(&self, id: u64) -> Result<Duration, AudioError> {
        Ok(self.sound(id)?.duration())
    }

    /// Disposes a registered sound, removing it.
    pub fn dispose_id(&mut self, id: u64) -> Result<(), AudioError> {
        let mut sound = self
            .sounds
            .remove(&id)
            .ok_or(AudioError::UnknownSound(id))?;
        sound.dispose();
        Ok(())
    }

    fn sound(&self, id: u64) -> Result<&Sound, AudioError> {
        self.sounds.get(&id).ok_or(AudioError::UnknownSound(id))
    }

    fn sound_mut(&mut self, id: u64) -> Result<&mut Sound, AudioError> {
        self.sounds.get_mut(&id).ok_or(AudioError::UnknownSound(id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clip(samples: Vec<f32>, sample_rate: u32, channels: u16) -> AudioClip {
        AudioClip {
            samples,
            sample_rate,
            channels,
            origin: None,
        }
    }

    fn push_u16_le(out: &mut Vec<u8>, value: u16) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn make_wav(tag: u16, bits: u16, channels: u16, rate: u32, data: &[u8]) -> Vec<u8> {
        let width = u32::from(bits / 8);
        let block_align = u32::from(channels) * width;
        let byte_rate = rate * block_align;
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(b"WAVE");
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        push_u16_le(&mut out, tag);
        push_u16_le(&mut out, channels);
        out.extend_from_slice(&rate.to_le_bytes());
        out.extend_from_slice(&byte_rate.to_le_bytes());
        push_u16_le(&mut out, block_align as u16);
        push_u16_le(&mut out, bits);
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(data);
        if !data.len().is_multiple_of(2) {
            out.push(0);
        }
        let riff_size = (out.len() - 8) as u32;
        out[4..8].copy_from_slice(&riff_size.to_le_bytes());
        out
    }

    fn make_wav_extensible(tag: u16, bits: u16, channels: u16, rate: u32, data: &[u8]) -> Vec<u8> {
        let width = u32::from(bits / 8);
        let block_align = u32::from(channels) * width;
        let byte_rate = rate * block_align;
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(b"WAVE");
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&40u32.to_le_bytes());
        push_u16_le(&mut out, 0xFFFE);
        push_u16_le(&mut out, channels);
        out.extend_from_slice(&rate.to_le_bytes());
        out.extend_from_slice(&byte_rate.to_le_bytes());
        push_u16_le(&mut out, block_align as u16);
        push_u16_le(&mut out, bits);
        push_u16_le(&mut out, 22);
        push_u16_le(&mut out, bits);
        out.extend_from_slice(&0u32.to_le_bytes());
        push_u16_le(&mut out, tag);
        out.extend_from_slice(&[
            0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71, 0x00, 0x00,
        ]);
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(data);
        if !data.len().is_multiple_of(2) {
            out.push(0);
        }
        let riff_size = (out.len() - 8) as u32;
        out[4..8].copy_from_slice(&riff_size.to_le_bytes());
        out
    }

    fn extended_80(value: u32) -> [u8; 10] {
        assert!(value > 0);
        let msb = 31 - value.leading_zeros();
        let exponent = 16_383 + msb;
        let mantissa = u64::from(value) << (63 - msb);
        let mut out = [0u8; 10];
        out[0] = ((exponent >> 8) & 0x7f) as u8;
        out[1] = exponent as u8;
        out[2..].copy_from_slice(&mantissa.to_be_bytes());
        out
    }

    fn make_aiff(
        aifc: bool,
        compression: &[u8; 4],
        bits: u16,
        channels: u16,
        rate: u32,
        frames: u32,
        data: &[u8],
    ) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"FORM");
        out.extend_from_slice(&0u32.to_be_bytes());
        out.extend_from_slice(if aifc { b"AIFC" } else { b"AIFF" });
        out.extend_from_slice(b"COMM");
        let comm_size = if aifc { 22u32 } else { 18u32 };
        out.extend_from_slice(&comm_size.to_be_bytes());
        out.extend_from_slice(&channels.to_be_bytes());
        out.extend_from_slice(&frames.to_be_bytes());
        out.extend_from_slice(&bits.to_be_bytes());
        out.extend_from_slice(&extended_80(rate));
        if aifc {
            out.extend_from_slice(compression);
        }
        out.extend_from_slice(b"SSND");
        out.extend_from_slice(&((data.len() + 8) as u32).to_be_bytes());
        out.extend_from_slice(&0u32.to_be_bytes());
        out.extend_from_slice(&0u32.to_be_bytes());
        out.extend_from_slice(data);
        if !data.len().is_multiple_of(2) {
            out.push(0);
        }
        let form_size = (out.len() - 8) as u32;
        out[4..8].copy_from_slice(&form_size.to_be_bytes());
        out
    }

    fn make_au(encoding: u32, rate: u32, channels: u32, data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b".snd");
        out.extend_from_slice(&24u32.to_be_bytes());
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        out.extend_from_slice(&encoding.to_be_bytes());
        out.extend_from_slice(&rate.to_be_bytes());
        out.extend_from_slice(&channels.to_be_bytes());
        out.extend_from_slice(data);
        out
    }

    fn approx(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1.0e-6,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn wav_8_bit_is_unsigned_and_normalized() {
        let wav = make_wav(1, 8, 1, 8_000, &[0, 64, 128, 192, 255]);
        let decoded = AudioClip::from_bytes(&wav, ".WAV").unwrap();
        assert_eq!(decoded.sample_rate, 8_000);
        assert_eq!(decoded.channels, 1);
        assert_eq!(decoded.frames(), 5);
        approx(decoded.samples[0], -1.0);
        approx(decoded.samples[1], -0.5);
        approx(decoded.samples[2], 0.0);
        approx(decoded.samples[3], 0.5);
        approx(decoded.samples[4], 127.0 / 128.0);
        assert_eq!(decoded.duration(), Duration::from_micros(625));
    }

    #[test]
    fn wav_16_bit_stereo_and_multichannel() {
        let wav = make_wav(
            1,
            16,
            2,
            44_100,
            &[0x00, 0x40, 0x00, 0xC0, 0x00, 0x80, 0xFF, 0x7F],
        );
        let decoded = AudioClip::decode_wav(&wav).unwrap();
        assert_eq!(decoded.channels, 2);
        assert_eq!(decoded.frames(), 2);
        approx(decoded.samples[0], 0.5);
        approx(decoded.samples[1], -0.5);
        approx(decoded.samples[2], -1.0);
        approx(decoded.samples[3], 32_767.0 / 32_768.0);

        let wav = make_wav(1, 16, 3, 32_000, &[0x00, 0x40, 0x00, 0xC0, 0x00, 0x20]);
        let decoded = AudioClip::decode_wav(&wav).unwrap();
        assert_eq!(decoded.channels, 3);
        assert_eq!(decoded.frames(), 1);
        approx(decoded.samples[0], 0.5);
        approx(decoded.samples[1], -0.5);
        approx(decoded.samples[2], 0.25);
    }

    #[test]
    fn wav_24_32_and_float_formats() {
        let wav = make_wav(
            1,
            24,
            1,
            48_000,
            &[0, 0, 0, 0, 0, 0x40, 0, 0, 0x80, 0xFF, 0xFF, 0x7F],
        );
        let decoded = AudioClip::decode_wav(&wav).unwrap();
        approx(decoded.samples[0], 0.0);
        approx(decoded.samples[1], 0.5);
        approx(decoded.samples[2], -1.0);
        approx(decoded.samples[3], 8_388_607.0 / 8_388_608.0);

        let wav = make_wav(
            1,
            32,
            1,
            48_000,
            &[
                0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x80, 0xFF, 0xFF, 0xFF, 0x7F,
            ],
        );
        let decoded = AudioClip::decode_wav(&wav).unwrap();
        approx(decoded.samples[0], 0.5);
        approx(decoded.samples[1], -1.0);
        approx(decoded.samples[2], 2_147_483_647.0 / 2_147_483_648.0);

        let mut data = Vec::new();
        for value in [0.25f32, -0.5, 1.5, -2.0, f32::NAN, f32::INFINITY] {
            data.extend_from_slice(&value.to_le_bytes());
        }
        let wav = make_wav(3, 32, 1, 44_100, &data);
        let decoded = AudioClip::decode_wav(&wav).unwrap();
        approx(decoded.samples[0], 0.25);
        approx(decoded.samples[1], -0.5);
        approx(decoded.samples[2], 1.0);
        approx(decoded.samples[3], -1.0);
        approx(decoded.samples[4], 0.0);
        approx(decoded.samples[5], 0.0);
    }

    #[test]
    fn wav_extensible_pcm_is_accepted() {
        let wav = make_wav_extensible(1, 16, 1, 16_000, &[0x00, 0x40, 0x00, 0xC0]);
        let decoded = AudioClip::decode_wav(&wav).unwrap();
        assert_eq!(decoded.sample_rate, 16_000);
        assert_eq!(decoded.frames(), 2);
        approx(decoded.samples[0], 0.5);
        approx(decoded.samples[1], -0.5);
    }

    #[test]
    fn wav_zero_samples_is_valid_but_bad_headers_are_not() {
        let wav = make_wav(1, 16, 1, 8_000, &[]);
        let decoded = AudioClip::decode_wav(&wav).unwrap();
        assert!(decoded.is_empty());
        assert_eq!(decoded.duration(), Duration::ZERO);

        for bad in [
            make_wav(1, 16, 0, 8_000, &[]),
            make_wav(1, 16, 1, 0, &[]),
            make_wav(1, 12, 1, 8_000, &[0, 0]),
            make_wav(6, 8, 1, 8_000, &[0]),
        ] {
            assert!(AudioClip::decode_wav(&bad).is_err());
        }

        let mut truncated = make_wav(1, 16, 1, 8_000, &[0, 0, 0, 0]);
        truncated.truncate(truncated.len() - 1);
        assert!(AudioClip::decode_wav(&truncated).is_err());

        let mut bad_size = make_wav(1, 16, 1, 8_000, &[0, 0]);
        let data_pos = bad_size
            .windows(4)
            .position(|window| window == b"data")
            .unwrap();
        bad_size[data_pos + 4..data_pos + 8].copy_from_slice(&99u32.to_le_bytes());
        assert!(AudioClip::decode_wav(&bad_size).is_err());

        assert!(matches!(
            AudioClip::from_bytes(&[], "mp3"),
            Err(AudioError::UnsupportedFormat(_))
        ));
    }

    #[test]
    fn aiff_pcm_8_16_24_and_32_bit_big_endian() {
        let aiff = make_aiff(
            false,
            b"NONE",
            16,
            2,
            44_100,
            2,
            &[0x40, 0x00, 0xC0, 0x00, 0x80, 0x00, 0x7F, 0xFF],
        );
        let decoded = AudioClip::decode_aiff(&aiff).unwrap();
        assert_eq!(decoded.channels, 2);
        assert_eq!(decoded.sample_rate, 44_100);
        approx(decoded.samples[0], 0.5);
        approx(decoded.samples[1], -0.5);
        approx(decoded.samples[2], -1.0);
        approx(decoded.samples[3], 32_767.0 / 32_768.0);

        let aiff = make_aiff(
            false,
            b"NONE",
            8,
            1,
            8_000,
            5,
            &[0x00, 0x40, 0x7F, 0x80, 0xC0],
        );
        let decoded = AudioClip::decode_aiff(&aiff).unwrap();
        approx(decoded.samples[0], 0.0);
        approx(decoded.samples[1], 0.5);
        approx(decoded.samples[2], 127.0 / 128.0);
        approx(decoded.samples[3], -1.0);
        approx(decoded.samples[4], -0.5);

        let aiff = make_aiff(
            false,
            b"NONE",
            24,
            1,
            48_000,
            4,
            &[
                0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x80, 0x00, 0x00, 0x7F, 0xFF, 0xFF,
            ],
        );
        let decoded = AudioClip::decode_aiff(&aiff).unwrap();
        approx(decoded.samples[0], 0.0);
        approx(decoded.samples[1], 0.5);
        approx(decoded.samples[2], -1.0);
        approx(decoded.samples[3], 8_388_607.0 / 8_388_608.0);

        let aiff = make_aiff(
            false,
            b"NONE",
            32,
            1,
            48_000,
            3,
            &[
                0x40, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x7F, 0xFF, 0xFF, 0xFF,
            ],
        );
        let decoded = AudioClip::decode_aiff(&aiff).unwrap();
        approx(decoded.samples[0], 0.5);
        approx(decoded.samples[1], -1.0);
        approx(decoded.samples[2], 2_147_483_647.0 / 2_147_483_648.0);
    }

    #[test]
    fn aiff_c_none_and_sowt_and_invalid_compression() {
        let aifc = make_aiff(true, b"NONE", 16, 1, 8_000, 2, &[0x40, 0x00, 0xC0, 0x00]);
        let decoded = AudioClip::decode_aiff(&aifc).unwrap();
        approx(decoded.samples[0], 0.5);
        approx(decoded.samples[1], -0.5);

        let sowt = make_aiff(true, b"sowt", 16, 1, 8_000, 2, &[0x00, 0x40, 0x00, 0xC0]);
        let decoded = AudioClip::decode_aiff(&sowt).unwrap();
        approx(decoded.samples[0], 0.5);
        approx(decoded.samples[1], -0.5);

        let ima4 = make_aiff(true, b"ima4", 16, 1, 8_000, 0, &[]);
        assert!(matches!(
            AudioClip::decode_aiff(&ima4),
            Err(AudioError::UnsupportedFormat(_))
        ));

        let mut bad = aifc;
        bad.truncate(30);
        assert!(AudioClip::decode_aiff(&bad).is_err());
    }

    #[test]
    fn aiff_rejects_missing_or_short_sound_data() {
        let no_ssnd = {
            let mut bytes = make_aiff(false, b"NONE", 16, 1, 8_000, 1, &[0, 1]);
            let ssnd = bytes
                .windows(4)
                .position(|window| window == b"SSND")
                .unwrap();
            bytes.truncate(ssnd);
            let size = (bytes.len() - 8) as u32;
            bytes[4..8].copy_from_slice(&size.to_be_bytes());
            bytes
        };
        assert!(AudioClip::decode_aiff(&no_ssnd).is_err());

        let short = make_aiff(false, b"NONE", 16, 1, 8_000, 2, &[0, 1]);
        assert!(matches!(
            AudioClip::decode_aiff(&short),
            Err(AudioError::Truncated(_))
        ));

        // A huge declared frame count must be rejected from the available data
        // before a huge sample buffer is reserved.
        let huge = make_aiff(false, b"NONE", 16, 1, 8_000, u32::MAX, &[0, 1]);
        assert!(matches!(
            AudioClip::decode_aiff(&huge),
            Err(AudioError::Truncated(_))
        ));
    }

    #[test]
    fn au_pcm_and_float_formats() {
        let au = make_au(
            3,
            44_100,
            2,
            &[0x40, 0x00, 0xC0, 0x00, 0x80, 0x00, 0x7F, 0xFF],
        );
        let decoded = AudioClip::decode_au(&au).unwrap();
        assert_eq!(decoded.channels, 2);
        assert_eq!(decoded.sample_rate, 44_100);
        approx(decoded.samples[0], 0.5);
        approx(decoded.samples[1], -0.5);
        approx(decoded.samples[2], -1.0);
        approx(decoded.samples[3], 32_767.0 / 32_768.0);

        let au = make_au(2, 8_000, 1, &[0x00, 0x40, 0x7F, 0x80, 0xC0]);
        let decoded = AudioClip::decode_au(&au).unwrap();
        approx(decoded.samples[0], 0.0);
        approx(decoded.samples[1], 0.5);
        approx(decoded.samples[2], 127.0 / 128.0);
        approx(decoded.samples[3], -1.0);
        approx(decoded.samples[4], -0.5);

        let au = make_au(
            4,
            48_000,
            1,
            &[0x40, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00],
        );
        let decoded = AudioClip::decode_au(&au).unwrap();
        approx(decoded.samples[0], 0.5);
        approx(decoded.samples[1], -1.0);
        approx(decoded.samples[2], 0.0);

        let au = make_au(
            5,
            48_000,
            1,
            &[0x40, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00],
        );
        let decoded = AudioClip::decode_au(&au).unwrap();
        approx(decoded.samples[0], 0.5);
        approx(decoded.samples[1], -1.0);

        let mut data = Vec::new();
        for value in [0.25f32, -0.5, 1.5, -2.0, f32::NAN] {
            data.extend_from_slice(&value.to_be_bytes());
        }
        let au = make_au(6, 44_100, 1, &data);
        let decoded = AudioClip::decode_au(&au).unwrap();
        approx(decoded.samples[0], 0.25);
        approx(decoded.samples[1], -0.5);
        approx(decoded.samples[2], 1.0);
        approx(decoded.samples[3], -1.0);
        approx(decoded.samples[4], 0.0);
    }

    #[test]
    fn au_zero_samples_and_invalid_headers() {
        let au = make_au(3, 8_000, 1, &[]);
        let decoded = AudioClip::decode_au(&au).unwrap();
        assert!(decoded.is_empty());

        for bad in [
            make_au(1, 8_000, 1, &[0, 0]),
            make_au(3, 8_000, 0, &[]),
            make_au(3, 0, 1, &[]),
            make_au(7, 8_000, 1, &[0; 8]),
        ] {
            assert!(AudioClip::decode_au(&bad).is_err());
        }

        let mut bad_size = make_au(3, 8_000, 1, &[0, 0]);
        bad_size[8..12].copy_from_slice(&100u32.to_be_bytes());
        assert!(matches!(
            AudioClip::decode_au(&bad_size),
            Err(AudioError::Truncated(_))
        ));

        let mut short = make_au(3, 8_000, 1, &[0, 0]);
        short.truncate(23);
        assert!(matches!(
            AudioClip::decode_au(&short),
            Err(AudioError::Truncated(_))
        ));

        let mut offset = make_au(2, 8_000, 1, &[0x40]);
        offset.splice(24..24, [0u8; 3]);
        offset[4..8].copy_from_slice(&27u32.to_be_bytes());
        let decoded = AudioClip::decode_au(&offset).unwrap();
        approx(decoded.samples[0], 0.5);
    }

    #[test]
    fn audio_error_implements_std_error_and_converts_io_errors() {
        let error: AudioError = io::Error::new(io::ErrorKind::NotFound, "missing file").into();
        assert!(matches!(error, AudioError::Io(_)));
        assert!(error.to_string().contains("missing file"));
        assert!(std::error::Error::source(&error).is_some());
        assert!(AudioError::UnsupportedFormat("x".into())
            .to_string()
            .contains("unsupported audio format"));
        assert!(AudioError::InvalidFormat("x".into())
            .to_string()
            .contains("invalid audio format"));
        assert!(AudioError::Truncated("x".into())
            .to_string()
            .contains("truncated audio data"));
        assert!(AudioError::TooLarge("x".into())
            .to_string()
            .contains("too large"));
        assert!(AudioError::Playback("x".into())
            .to_string()
            .contains("playback error"));
        assert_eq!(
            AudioError::UnknownSound(7).to_string(),
            "unknown sound id: 7"
        );
    }

    #[test]
    fn load_sets_origin_and_uses_extension() {
        let path = std::env::temp_dir().join(format!(
            "cosplay-audio-test-{}-{}.wav",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, make_wav(1, 16, 1, 8_000, &[0, 0x40])).unwrap();
        let decoded = AudioClip::load(&path).unwrap();
        assert_eq!(
            decoded.origin.as_deref(),
            Some(path.to_string_lossy().as_ref())
        );
        approx(decoded.samples[0], 0.5);
        let _ = std::fs::remove_file(path);
        assert!(decoded.validate().is_ok());
        assert!(clip(vec![0.0, 1.0], 8_000, 1).validate().is_ok());
        assert!(clip(vec![0.0], 8_000, 2).validate().is_err());
        assert!(clip(vec![], 0, 1).validate().is_err());
        assert!(clip(vec![], 8_000, 0).validate().is_err());
    }

    #[test]
    fn beat_envelope_normalizes_and_detects_thresholds() {
        let mut samples = vec![1.0f32; 10];
        samples.extend(vec![0.5f32; 10]);
        let clip = clip(samples, 1_000, 1);
        let envelope = BeatEnvelope::new(&clip).with_smoothing(Smoothing::SmoothNone);

        assert_eq!(envelope.window_count(), 2);
        assert_eq!(envelope.window_duration(), Duration::from_millis(10));
        assert_eq!(envelope.duration(), Duration::from_millis(20));
        assert_eq!(envelope.threshold_tail(), Duration::from_millis(2_000));
        assert_eq!(envelope.threshold_gain(), 1.0);
        approx(envelope.normalized_amplitudes()[0], 1.0);
        approx(envelope.normalized_amplitudes()[1], 0.0);

        assert_eq!(envelope.brightness_at(Duration::ZERO), 1.0);
        assert_eq!(envelope.brightness_at(Duration::from_millis(10)), 0.0);
        assert_eq!(envelope.brightness_at(Duration::from_millis(15)), 0.0);
        assert_eq!(envelope.brightness_at(Duration::from_millis(20)), 1.0);
        assert_eq!(
            envelope.brightness_at(Duration::from_millis(25)),
            envelope.brightness_at(Duration::from_millis(5))
        );
    }

    #[test]
    fn beat_smoothing_modes_are_deterministic() {
        let mut samples = vec![1.0f32; 10];
        samples.extend(vec![0.5f32; 10]);
        samples.extend(vec![0.75f32; 10]);
        let clip = clip(samples, 1_000, 1);
        let at_third = Duration::from_millis(20);

        let none = BeatEnvelope::new(&clip).with_smoothing(Smoothing::SmoothNone);
        let up = BeatEnvelope::new(&clip).with_smoothing(Smoothing::SmoothUp);
        let down = BeatEnvelope::new(&clip).with_smoothing(Smoothing::SmoothDown);
        let both = BeatEnvelope::new(&clip).with_smoothing(Smoothing::SmoothBoth);

        assert_eq!(none.brightness_at(at_third), 0.0);
        assert_eq!(up.brightness_at(at_third), 0.0);
        approx(down.brightness_at(at_third), 0.5);
        approx(both.brightness_at(at_third), 0.5);

        assert_eq!(none.brightness_at(Duration::ZERO), 1.0);
        assert_eq!(up.brightness_at(Duration::ZERO), 1.0);
        assert_eq!(down.brightness_at(Duration::ZERO), 1.0);
        assert_eq!(both.brightness_at(Duration::ZERO), 1.0);

        let amplified = none
            .clone()
            .with_threshold_gain(0.5)
            .with_threshold_tail(Duration::from_secs(2));
        assert_eq!(amplified.brightness_at(at_third), 1.0);
        let disabled_threshold = none.clone().with_threshold_tail(Duration::ZERO);
        assert_eq!(disabled_threshold.threshold_tail(), Duration::ZERO);
        assert_eq!(disabled_threshold.brightness_at(at_third), 1.0);

        let configured = BeatEnvelope::new(&clip).with_options(
            Smoothing::SmoothBoth,
            0.75,
            Duration::from_millis(20),
        );
        assert_eq!(configured.smoothing(), Smoothing::SmoothBoth);
        assert_eq!(configured.threshold_gain(), 0.75);
        assert_eq!(configured.threshold_tail(), Duration::from_millis(20));
        let mut configured = configured;
        configured.set_smoothing(Smoothing::SmoothDown);
        configured.set_threshold_gain(f32::NAN);
        configured.set_threshold_gain(-1.0);
        assert_eq!(configured.smoothing(), Smoothing::SmoothDown);
        assert_eq!(configured.threshold_gain(), DEFAULT_THRESHOLD_GAIN);
    }

    #[test]
    fn beat_envelope_handles_empty_silent_and_short_clips() {
        let empty = clip(vec![], 8_000, 1);
        let envelope = BeatEnvelope::analyze(&empty);
        assert!(envelope.is_empty());
        assert_eq!(envelope.window_count(), 0);
        assert_eq!(envelope.duration(), Duration::ZERO);
        assert_eq!(envelope.brightness_at(Duration::ZERO), 0.0);
        assert_eq!(envelope.brightness_at(Duration::from_secs(10)), 0.0);

        let silent = clip(vec![0.0; 20], 1_000, 1);
        let envelope = BeatEnvelope::new(&silent);
        assert_eq!(envelope.window_count(), 2);
        assert_eq!(envelope.brightness_at(Duration::ZERO), 0.0);
        assert_eq!(envelope.brightness_at(Duration::from_millis(10)), 0.0);

        let short = clip(vec![0.25; 5], 1_000, 1);
        let envelope = BeatEnvelope::new(&short);
        assert_eq!(envelope.window_count(), 1);
        assert_eq!(envelope.duration(), Duration::from_millis(5));
        approx(envelope.normalized_amplitudes()[0], 1.0);
        assert_eq!(envelope.brightness_at(Duration::ZERO), 1.0);
        assert_eq!(envelope.brightness_at(Duration::from_millis(4)), 1.0);
        assert_eq!(envelope.brightness_at(Duration::from_millis(5)), 1.0);
        assert_eq!(envelope.brightness_at(Duration::from_secs(1)), 1.0);

        let stereo = clip(vec![0.5; 4], 1_000, 2);
        let envelope = BeatEnvelope::new(&stereo);
        assert_eq!(envelope.window_count(), 1);
        assert_eq!(envelope.brightness_at(Duration::ZERO), 1.0);

        for mode in [
            Smoothing::SmoothNone,
            Smoothing::SmoothUp,
            Smoothing::SmoothDown,
            Smoothing::SmoothBoth,
        ] {
            let value = BeatEnvelope::new(&short)
                .with_smoothing(mode)
                .brightness_at(Duration::ZERO);
            assert!((0.0..=1.0).contains(&value));
        }
    }

    #[test]
    fn sound_controls_work_without_a_device() {
        let clip = clip(vec![0.1; 100], 1_000, 1);
        let borrowed = Sound::new(&clip);
        assert_eq!(borrowed.clip(), &clip);
        let mut sound = Sound::new(clip.clone());
        assert_eq!(sound.clip(), &clip);
        assert_eq!(sound.duration(), Duration::from_millis(100));
        assert_eq!(sound.current_time(), Duration::ZERO);
        assert!(!sound.is_playing());
        assert!(!sound.is_looping());
        assert_eq!(sound.volume(), 1.0);
        assert_eq!(sound.speed(), 1.0);
        assert_eq!(sound.beat_envelope().window_count(), 10);

        sound.set_volume(-1.0);
        assert_eq!(sound.volume(), 0.0);
        sound.set_volume(9.0);
        assert_eq!(sound.volume(), 2.0);
        sound.set_volume(f32::NAN);
        assert_eq!(sound.volume(), 0.0);
        sound.set_volume(1.25);
        assert_eq!(sound.volume(), 1.25);

        sound.set_speed(0.0);
        assert_eq!(sound.speed(), 0.1);
        sound.set_speed(10.0);
        assert_eq!(sound.speed(), 4.0);
        sound.set_speed(f32::NAN);
        assert_eq!(sound.speed(), 1.0);
        sound.set_speed(2.0);

        sound.loop_clip(true);
        assert!(sound.is_looping());
        sound.seek(Duration::from_millis(50)).unwrap();
        assert_eq!(sound.current_time(), Duration::from_millis(50));
        sound.seek(Duration::from_secs(10)).unwrap();
        assert_eq!(sound.current_time(), Duration::from_millis(100));
        sound.rewind().unwrap();
        assert_eq!(sound.current_time(), Duration::ZERO);
        sound.stop();
        assert_eq!(sound.current_time(), Duration::ZERO);
        assert!(!sound.is_disposed());
        sound.dispose();
        assert!(sound.is_disposed());
        assert!(matches!(sound.play(), Err(AudioError::Playback(_))));
        sound.dispose();
    }

    #[cfg(not(feature = "audio"))]
    #[test]
    fn playback_without_the_feature_reports_an_error() {
        let mut sound = Sound::new(clip(vec![0.0; 10], 1_000, 1));
        assert!(matches!(
            sound.play(),
            Err(AudioError::Playback(message)) if message.contains("audio")
        ));
        assert!(!sound.is_playing());
    }

    #[cfg(feature = "audio")]
    #[test]
    fn loop_buffer_wraps_and_supports_seeking() {
        let mut source = LoopBuffer::new(
            Arc::from(vec![1.0, 2.0, 3.0, 4.0]),
            2,
            1_000,
            Duration::ZERO,
        );
        assert_eq!(source.next(), Some(1.0));
        assert_eq!(source.next(), Some(2.0));
        assert_eq!(source.next(), Some(3.0));
        assert_eq!(source.next(), Some(4.0));
        assert_eq!(source.next(), Some(1.0));

        source.try_seek(Duration::from_millis(1)).unwrap();
        assert_eq!(source.next(), Some(3.0));
        assert_eq!(source.next(), Some(4.0));
        assert_eq!(source.next(), Some(1.0));

        source.try_seek(Duration::from_millis(100)).unwrap();
        assert_eq!(source.next(), Some(1.0));

        let mut empty = LoopBuffer::new(Arc::from(Vec::new()), 1, 8_000, Duration::ZERO);
        assert_eq!(empty.next(), None);
        empty.try_seek(Duration::from_secs(1)).unwrap();
        assert_eq!(empty.next(), None);
    }

    #[cfg(feature = "audio")]
    #[test]
    fn invalid_clip_is_rejected_before_opening_a_device() {
        let mut sound = Sound::new(clip(vec![], 8_000, 0));
        assert!(matches!(sound.play(), Err(AudioError::InvalidFormat(_))));
        let mut sound = Sound::new(clip(vec![], 0, 1));
        assert!(matches!(sound.play(), Err(AudioError::InvalidFormat(_))));
    }

    #[test]
    fn registry_owns_and_controls_sounds() {
        let mut registry = SoundRegistry::new();
        assert!(registry.is_empty());
        let first = registry.insert(Sound::new(clip(vec![0.1; 20], 1_000, 1)));
        let second = registry.add(Sound::new(clip(vec![0.2; 20], 1_000, 1)));
        assert_ne!(first, second);
        assert_eq!(registry.len(), 2);
        assert!(registry.contains(first));
        assert!(registry.get(first).is_some());
        assert_eq!(registry.iter().count(), 2);

        registry.foreach(|sound| sound.set_volume(0.5));
        assert_eq!(registry.volume_id(first).unwrap(), 0.5);
        assert_eq!(registry.volume_id(second).unwrap(), 0.5);
        registry.foreach_with_id(|id, sound| {
            sound.set_speed(if id == first { 1.5 } else { 0.5 });
        });
        assert_eq!(registry.get(first).unwrap().speed(), 1.5);
        assert_eq!(registry.get(second).unwrap().speed(), 0.5);

        registry.set_volume_id(first, 1.75).unwrap();
        assert_eq!(registry.volume_id(first).unwrap(), 1.75);
        registry.seek_id(first, Duration::from_millis(10)).unwrap();
        assert_eq!(
            registry.current_time_id(first).unwrap(),
            Duration::from_millis(10)
        );
        assert_eq!(
            registry.duration_id(first).unwrap(),
            Duration::from_millis(20)
        );
        assert!(matches!(
            registry.play_id(999),
            Err(AudioError::UnknownSound(999))
        ));

        let removed = registry.remove(second).unwrap();
        assert_eq!(removed.duration(), Duration::from_millis(20));
        assert!(!registry.contains(second));
        assert!(registry
            .insert_with_id(42, Sound::new(clip(vec![0.3; 10], 1_000, 1)))
            .is_none());
        assert!(registry.contains(42));
        registry.dispose_id(42).unwrap();
        assert!(!registry.contains(42));

        registry.stop_all();
        assert_eq!(registry.current_time_id(first).unwrap(), Duration::ZERO);
        registry.dispose_all();
        assert!(registry.is_empty());

        let mut default = SoundRegistry::default();
        let id = default.insert(Sound::new(clip(vec![0.0; 1], 1_000, 1)));
        assert!(id >= 1);
        assert_eq!(default.len(), 1);
    }
}
