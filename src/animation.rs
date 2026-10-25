// SPDX-License-Identifier: Apache-2.0
//! Renderer-independent animation and ASCII video playback.
//!
//! [`Animation`] is a mutable playback cursor over a sequence of key frames.
//! It mirrors CosPlay's time-based and film-strip animations: callers supply a
//! monotonically increasing frame timestamp, and [`Animation::key_frame`]
//! decides which image should be shown. Nothing in this module draws to a
//! canvas or touches a terminal.
//!
//! [`VideoClip`] is a definition made of equally sized frames played at a fixed
//! frame rate. [`VideoPlayer`] owns a shared [`Video`] definition and provides
//! deterministic playback control. Its [`VideoPlayer::update`] method emits a
//! [`VideoEvent`] on every call, including while paused or after the final
//! frame, so a renderer can always draw the current frame.
//!
//! Time is supplied by the caller as an absolute timestamp, typically elapsed
//! time since the game started. Animations switch frames when the elapsed time
//! since the previous switch is *strictly greater* than the frame duration,
//! matching CosPlay's upstream behavior.

use crate::graphics::Color;
use crate::image::Image;
use std::error::Error;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

/// One frame produced by an [`Animation`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnimationKeyFrame {
    /// ID of the animation that produced this key frame.
    pub animation_id: String,
    /// The image to render.
    pub image: Arc<Image>,
    /// Zero-based index of this frame in the animation.
    pub index: usize,
}

/// An error returned while constructing an [`Animation`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimationError {
    /// The animation contains no frames.
    Empty,
    /// A frame has a zero duration.
    ZeroDuration {
        /// Index of the invalid frame.
        index: usize,
    },
    /// Bouncing was requested without looping.
    BounceWithoutLoop,
    /// The animation ID was empty or contained only whitespace.
    IdEmpty,
}

impl fmt::Display for AnimationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("animation must contain at least one frame"),
            Self::ZeroDuration { index } => {
                write!(f, "animation frame {index} has a zero duration")
            }
            Self::BounceWithoutLoop => f.write_str("a bouncing animation must also be looping"),
            Self::IdEmpty => f.write_str("animation ID cannot be empty"),
        }
    }
}

impl Error for AnimationError {}

/// A mutable animation playback cursor.
///
/// The frame definitions are immutable, but playback state (current index,
/// direction, playing flag, and timer) changes as [`Animation::key_frame`] is
/// called. Use [`Animation::reset`] to return to the initial state.
#[derive(Debug, Clone)]
pub struct Animation {
    id: String,
    frames: Vec<(Arc<Image>, Duration)>,
    looping: bool,
    bouncing: bool,
    index: usize,
    direction: i8,
    playing: bool,
    last_frame_time: Option<Duration>,
}

impl Animation {
    /// Creates a time-based animation.
    ///
    /// Each frame is shown for its own duration. `loop_` controls whether
    /// playback wraps around; `bounce` makes a looping animation reverse
    /// direction at both ends instead of jumping back to the first frame.
    ///
    /// # Errors
    ///
    /// Returns [`AnimationError::IdEmpty`] for a blank ID,
    /// [`AnimationError::Empty`] when `frames` is empty,
    /// [`AnimationError::ZeroDuration`] for a zero frame duration, and
    /// [`AnimationError::BounceWithoutLoop`] when `bounce` is true but `loop_`
    /// is false.
    pub fn time_based(
        id: impl Into<String>,
        loop_: bool,
        bounce: bool,
        frames: Vec<(Arc<Image>, Duration)>,
    ) -> Result<Self, AnimationError> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(AnimationError::IdEmpty);
        }
        if frames.is_empty() {
            return Err(AnimationError::Empty);
        }
        if let Some(index) = frames.iter().position(|(_, duration)| duration.is_zero()) {
            return Err(AnimationError::ZeroDuration { index });
        }
        if bounce && !loop_ {
            return Err(AnimationError::BounceWithoutLoop);
        }

        Ok(Self {
            id,
            frames,
            looping: loop_,
            bouncing: bounce,
            index: 0,
            direction: 1,
            playing: true,
            last_frame_time: None,
        })
    }

    /// Creates a film-strip animation with one duration shared by all frames.
    ///
    /// This is a convenience wrapper around [`Animation::time_based`] and
    /// returns the same validation errors.
    pub fn film_strip(
        id: impl Into<String>,
        frame: Duration,
        loop_: bool,
        bounce: bool,
        images: Vec<Arc<Image>>,
    ) -> Result<Self, AnimationError> {
        let frames = images.into_iter().map(|image| (image, frame)).collect();
        Self::time_based(id, loop_, bounce, frames)
    }

    /// Returns the animation ID.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the number of key frames.
    pub fn key_frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Returns whether the animation loops.
    pub fn is_looping(&self) -> bool {
        self.looping
    }

    /// Returns whether a looping animation bounces at its ends.
    pub fn is_bouncing(&self) -> bool {
        self.bouncing
    }

    /// Returns whether playback is currently running.
    pub fn playing(&self) -> bool {
        self.playing
    }

    /// Alias for [`Animation::playing`].
    pub fn is_playing(&self) -> bool {
        self.playing
    }

    /// Starts or pauses playback.
    ///
    /// Changing the state also restarts the frame timer, so a paused animation
    /// does not immediately advance when resumed.
    pub fn set_playing(&mut self, playing: bool) {
        if playing != self.playing {
            self.playing = playing;
            self.last_frame_time = None;
        }
    }

    /// Returns the current key-frame index.
    pub fn current_index(&self) -> usize {
        self.index
    }

    /// Returns the duration of a key frame, or [`Duration::ZERO`] when `index`
    /// is outside the animation.
    pub fn frame_duration(&self, index: usize) -> Duration {
        self.frames
            .get(index)
            .map_or(Duration::ZERO, |(_, duration)| *duration)
    }

    /// Resets playback to the first frame and starts it again.
    pub fn reset(&mut self) {
        self.index = 0;
        self.direction = 1;
        self.playing = true;
        self.last_frame_time = None;
    }

    /// Returns the key frame that should be rendered at `frame_time`.
    ///
    /// `frame_time` is an absolute, monotonically increasing timestamp. The
    /// first call starts the timer and returns the current frame. Later calls
    /// advance only when the time since the previous advance is strictly
    /// greater than the current frame's duration.
    ///
    /// A non-looping animation returns `None` after the last frame's duration
    /// has elapsed, and remains stopped until [`Animation::reset`] or
    /// [`Animation::set_playing`] is called. A paused animation also returns
    /// `None`.
    pub fn key_frame(&mut self, frame_time: Duration) -> Option<AnimationKeyFrame> {
        if !self.playing {
            return None;
        }

        match self.last_frame_time {
            None => self.last_frame_time = Some(frame_time),
            Some(last_time) => {
                if frame_time.saturating_sub(last_time) > self.frames[self.index].1 {
                    self.last_frame_time = Some(frame_time);
                    if !self.advance() {
                        return None;
                    }
                }
            }
        }

        Some(AnimationKeyFrame {
            animation_id: self.id.clone(),
            image: Arc::clone(&self.frames[self.index].0),
            index: self.index,
        })
    }

    /// Returns whether `key_frame` is the first frame of this animation.
    ///
    /// A key frame from another animation is never considered first.
    pub fn is_first_frame(&self, key_frame: &AnimationKeyFrame) -> bool {
        key_frame.animation_id == self.id && key_frame.index == 0
    }

    /// Returns whether `key_frame` is the last frame of this animation.
    ///
    /// A key frame from another animation is never considered last.
    pub fn is_last_frame(&self, key_frame: &AnimationKeyFrame) -> bool {
        key_frame.animation_id == self.id && key_frame.index == self.frames.len().saturating_sub(1)
    }

    /// Advances one frame, returning `false` when a non-looping animation has
    /// finished.
    fn advance(&mut self) -> bool {
        let last_index = self.frames.len() - 1;

        if self.index == last_index {
            if !self.looping {
                self.playing = false;
                return false;
            }
            if self.bouncing {
                self.direction = -1;
                self.index = last_index.saturating_sub(1);
            } else {
                self.direction = 1;
                self.index = 0;
            }
        } else if self.index == 0 {
            self.direction = 1;
            self.index = 1;
        } else {
            self.index = if self.direction < 0 {
                self.index - 1
            } else {
                self.index + 1
            };
        }

        true
    }
}

/// A shared video definition.
///
/// Implementations expose equally sized frames and a frame rate. The default
/// [`Video::fps`] is 30, so a minimal implementation only needs the four
/// required methods.
pub trait Video {
    /// Returns the video ID.
    fn id(&self) -> &str;

    /// Returns the number of frames.
    fn frame_count(&self) -> usize;

    /// Returns the width and height shared by every frame.
    fn frame_dim(&self) -> (u16, u16);

    /// Returns the frame at `index`, or `None` when out of range.
    fn frame(&self, index: usize) -> Option<Arc<Image>>;

    /// Returns the playback frame rate.
    ///
    /// Implementations should return a finite value greater than zero. A
    /// player falls back to 30 FPS for an invalid value.
    fn fps(&self) -> f32 {
        30.0
    }
}

/// An error returned while constructing a [`VideoClip`].
#[derive(Debug, Clone, PartialEq)]
pub enum VideoError {
    /// The video contains no frames.
    Empty,
    /// The video ID was empty or contained only whitespace.
    IdEmpty,
    /// The frame rate was not finite or was not greater than zero.
    InvalidFps {
        /// The rejected frame rate.
        fps: f32,
    },
    /// A frame's dimensions differ from the first frame's dimensions.
    DimensionMismatch {
        /// Index of the mismatched frame.
        index: usize,
        /// Dimensions of the first frame.
        expected: (u16, u16),
        /// Dimensions of the mismatched frame.
        actual: (u16, u16),
    },
}

impl fmt::Display for VideoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("video must contain at least one frame"),
            Self::IdEmpty => f.write_str("video ID cannot be empty"),
            Self::InvalidFps { fps } => {
                write!(f, "invalid video frame rate {fps} (must be finite and > 0)")
            }
            Self::DimensionMismatch {
                index,
                expected,
                actual,
            } => write!(
                f,
                "video frame {index} is {}x{}, expected {}x{}",
                actual.0, actual.1, expected.0, expected.1
            ),
        }
    }
}

impl Error for VideoError {}

/// A concrete video definition backed by in-memory images.
#[derive(Debug, Clone)]
pub struct VideoClip {
    id: String,
    frames: Vec<Arc<Image>>,
    fps: f32,
    frame_dim: (u16, u16),
    frame_duration: Duration,
}

impl VideoClip {
    /// Creates a video clip.
    ///
    /// Every frame must have the same dimensions as the first frame. `fps`
    /// must be finite and greater than zero.
    ///
    /// # Errors
    ///
    /// Returns [`VideoError::IdEmpty`] for a blank ID, [`VideoError::Empty`]
    /// when `frames` is empty, [`VideoError::InvalidFps`] for an invalid frame
    /// rate, and [`VideoError::DimensionMismatch`] when frames differ in size.
    pub fn new(
        id: impl Into<String>,
        frames: Vec<Arc<Image>>,
        fps: f32,
    ) -> Result<Self, VideoError> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(VideoError::IdEmpty);
        }
        if frames.is_empty() {
            return Err(VideoError::Empty);
        }
        if !fps.is_finite() || fps <= 0.0 {
            return Err(VideoError::InvalidFps { fps });
        }

        let frame_dim = (frames[0].width(), frames[0].height());
        for (index, frame) in frames.iter().enumerate().skip(1) {
            let actual = (frame.width(), frame.height());
            if actual != frame_dim {
                return Err(VideoError::DimensionMismatch {
                    index,
                    expected: frame_dim,
                    actual,
                });
            }
        }

        let frame_duration = duration_from_fps(fps);
        Ok(Self {
            id,
            frames,
            fps,
            frame_dim,
            frame_duration,
        })
    }

    /// Creates a video clip from plain-text frames.
    ///
    /// Each entry is converted with [`Image::from_text`]. Frames must therefore
    /// end up with identical dimensions; use newline-separated text with
    /// consistent line lengths to control this explicitly.
    ///
    /// # Errors
    ///
    /// Returns any error from [`VideoClip::new`], including a dimension
    /// mismatch between the rendered text frames.
    pub fn from_text_frames(
        id: impl Into<String>,
        texts: &[&str],
        fg: Color,
        bg: Option<Color>,
        transparent_spaces: bool,
        fps: f32,
    ) -> Result<Self, VideoError> {
        let frames = texts
            .iter()
            .map(|text| Arc::new(Image::from_text(text, fg, bg, transparent_spaces)))
            .collect();
        Self::new(id, frames, fps)
    }

    /// Returns the frame rate.
    pub fn fps(&self) -> f32 {
        self.fps
    }

    /// Returns the video ID.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the number of frames.
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Returns the width and height shared by every frame.
    pub fn frame_dim(&self) -> (u16, u16) {
        self.frame_dim
    }

    /// Returns the frame at `index`, or `None` when out of range.
    pub fn frame(&self, index: usize) -> Option<Arc<Image>> {
        self.frames.get(index).cloned()
    }

    /// Returns the duration of one frame.
    pub fn frame_duration(&self) -> Duration {
        self.frame_duration
    }

    /// Returns the total duration at the configured frame rate.
    pub fn duration(&self) -> Duration {
        let seconds = self.frames.len() as f64 / f64::from(self.fps);
        Duration::try_from_secs_f64(seconds).unwrap_or(Duration::MAX)
    }
}

impl Video for VideoClip {
    fn id(&self) -> &str {
        self.id()
    }

    fn frame_count(&self) -> usize {
        self.frame_count()
    }

    fn frame_dim(&self) -> (u16, u16) {
        self.frame_dim()
    }

    fn frame(&self, index: usize) -> Option<Arc<Image>> {
        self.frame(index)
    }

    fn fps(&self) -> f32 {
        self.fps
    }
}

/// An event emitted by [`VideoPlayer::update`] for the frame to render.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoEvent {
    /// Zero-based index of the frame.
    pub index: usize,
    /// The image to render.
    pub image: Arc<Image>,
    /// Whether playback was running for this event.
    pub playing: bool,
}

impl VideoEvent {
    /// Converts the event into its `(index, image)` pair.
    pub fn into_frame(self) -> (usize, Arc<Image>) {
        (self.index, self.image)
    }
}

/// A deterministic player for a shared [`Video`] definition.
///
/// Playback uses absolute caller-supplied timestamps. Frame changes occur when
/// the time since the previous change is strictly greater than one frame
/// duration, matching [`Animation`] and CosPlay's upstream timing.
pub struct VideoPlayer {
    video: Arc<dyn Video>,
    playing: bool,
    looping: bool,
    index: usize,
    last_frame_time: Option<Duration>,
    frame_duration: Duration,
}

impl VideoPlayer {
    /// Creates a player that owns a shared video definition.
    ///
    /// `autoplay` starts playback immediately and `looping` wraps from the last
    /// frame to the first. A video with no frames never starts and produces no
    /// events.
    pub fn new(video: Arc<dyn Video>, autoplay: bool, looping: bool) -> Self {
        let has_frames = video.frame_count() > 0;
        Self {
            frame_duration: duration_from_fps(video.fps()),
            video,
            playing: autoplay && has_frames,
            looping,
            index: 0,
            last_frame_time: None,
        }
    }

    /// Returns the shared video definition.
    pub fn video(&self) -> &Arc<dyn Video> {
        &self.video
    }

    /// Returns the video ID.
    pub fn id(&self) -> &str {
        self.video.id()
    }

    /// Returns the number of video frames.
    pub fn frame_count(&self) -> usize {
        self.video.frame_count()
    }

    /// Returns the frame dimensions.
    pub fn frame_dim(&self) -> (u16, u16) {
        self.video.frame_dim()
    }

    /// Returns the duration of one frame.
    pub fn frame_duration(&self) -> Duration {
        self.frame_duration
    }

    /// Returns whether playback is running.
    pub fn is_playing(&self) -> bool {
        self.playing
    }

    /// Returns whether playback loops.
    pub fn is_looping(&self) -> bool {
        self.looping
    }

    /// Starts or resumes playback from the current frame.
    pub fn play(&mut self) {
        if self.video.frame_count() == 0 {
            self.playing = false;
            return;
        }
        if !self.playing {
            self.playing = true;
            self.last_frame_time = None;
        }
    }

    /// Pauses playback without changing the current frame.
    pub fn stop(&mut self) {
        self.playing = false;
        self.last_frame_time = None;
    }

    /// Toggles between playing and paused.
    pub fn toggle(&mut self) {
        if self.playing {
            self.stop();
        } else {
            self.play();
        }
    }

    /// Returns the current frame index.
    pub fn current_index(&self) -> usize {
        self.index
    }

    /// Seeks to a frame index.
    ///
    /// Returns `false` and leaves the position unchanged when `index` is out of
    /// range. Seeking restarts the frame timer but does not change whether the
    /// player is playing.
    pub fn seek_frame(&mut self, index: usize) -> bool {
        if index >= self.video.frame_count() {
            return false;
        }
        self.index = index;
        self.last_frame_time = None;
        true
    }

    /// Rewinds to the first frame, preserving the playing state.
    pub fn rewind(&mut self) {
        self.index = 0;
        self.last_frame_time = None;
    }

    /// Updates playback and returns the frame to render.
    ///
    /// An event is returned on every call while the video has a current frame,
    /// including when paused or when a non-looping video has reached its last
    /// frame. Paused playback emits the current frame with
    /// [`VideoEvent::playing`] set to `false`. The first call while playing
    /// starts the frame timer.
    pub fn update(&mut self, frame_time: Duration) -> Option<VideoEvent> {
        let frame_count = self.video.frame_count();
        if frame_count == 0 {
            self.playing = false;
            return None;
        }
        if self.index >= frame_count {
            self.index = frame_count - 1;
        }

        if self.playing {
            match self.last_frame_time {
                None => self.last_frame_time = Some(frame_time),
                Some(last_time) => {
                    if frame_time.saturating_sub(last_time) > self.frame_duration {
                        self.last_frame_time = Some(frame_time);
                        if self.index + 1 < frame_count {
                            self.index += 1;
                        } else if self.looping {
                            self.index = 0;
                        } else {
                            self.playing = false;
                        }
                    }
                }
            }
        }

        let image = self.video.frame(self.index)?;
        Some(VideoEvent {
            index: self.index,
            image,
            playing: self.playing,
        })
    }

    /// Updates playback and returns only the `(index, image)` pair.
    ///
    /// This is a convenience wrapper around [`VideoPlayer::update`].
    pub fn update_frame(&mut self, frame_time: Duration) -> Option<(usize, Arc<Image>)> {
        self.update(frame_time).map(VideoEvent::into_frame)
    }

    /// Returns the current frame without advancing playback.
    pub fn current_event(&self) -> Option<VideoEvent> {
        self.video.frame(self.index).map(|image| VideoEvent {
            index: self.index,
            image,
            playing: self.playing,
        })
    }
}

impl fmt::Debug for VideoPlayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VideoPlayer")
            .field("id", &self.video.id())
            .field("playing", &self.playing)
            .field("looping", &self.looping)
            .field("index", &self.index)
            .field("frame_duration", &self.frame_duration)
            .finish()
    }
}

fn duration_from_fps(fps: f32) -> Duration {
    if !fps.is_finite() || fps <= 0.0 {
        return Duration::from_secs_f64(1.0 / 30.0);
    }
    let duration = Duration::try_from_secs_f64(1.0 / f64::from(fps)).unwrap_or(Duration::MAX);
    duration.max(Duration::from_nanos(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graphics::Pixel;

    fn image(ch: char) -> Arc<Image> {
        Arc::new(Image::new(2, 2, Pixel::new(ch, Color::WHITE, None)))
    }

    fn time_frames() -> Vec<(Arc<Image>, Duration)> {
        vec![
            (image('a'), Duration::from_millis(100)),
            (image('b'), Duration::from_millis(100)),
            (image('c'), Duration::from_millis(100)),
        ]
    }

    fn animation() -> Animation {
        Animation::time_based("ani", true, false, time_frames()).unwrap()
    }

    #[test]
    fn time_based_validates_arguments() {
        assert_eq!(
            Animation::time_based("", true, false, time_frames()).unwrap_err(),
            AnimationError::IdEmpty
        );
        assert_eq!(
            Animation::time_based("   ", true, false, time_frames()).unwrap_err(),
            AnimationError::IdEmpty
        );
        assert_eq!(
            Animation::time_based("ani", true, false, Vec::new()).unwrap_err(),
            AnimationError::Empty
        );

        let frames = vec![
            (image('a'), Duration::from_millis(10)),
            (image('b'), Duration::ZERO),
        ];
        assert_eq!(
            Animation::time_based("ani", true, false, frames).unwrap_err(),
            AnimationError::ZeroDuration { index: 1 }
        );
        assert_eq!(
            Animation::time_based("ani", false, true, time_frames()).unwrap_err(),
            AnimationError::BounceWithoutLoop
        );

        let error = Animation::time_based("ani", false, true, Vec::new()).unwrap_err();
        assert_eq!(error, AnimationError::Empty);
        assert!(!error.to_string().is_empty());
    }

    #[test]
    fn film_strip_uses_one_duration_for_every_frame() {
        let images = vec![image('a'), image('b'), image('c')];
        let animation =
            Animation::film_strip("strip", Duration::from_millis(50), true, true, images).unwrap();

        assert_eq!(animation.id(), "strip");
        assert_eq!(animation.key_frame_count(), 3);
        assert!(animation.is_looping());
        assert!(animation.is_bouncing());
        assert!(animation.playing());
        assert_eq!(animation.current_index(), 0);
        assert_eq!(animation.frame_duration(0), Duration::from_millis(50));
        assert_eq!(animation.frame_duration(2), Duration::from_millis(50));
        assert_eq!(animation.frame_duration(3), Duration::ZERO);

        assert_eq!(
            Animation::film_strip("strip", Duration::ZERO, true, false, Vec::new()).unwrap_err(),
            AnimationError::Empty
        );
        assert_eq!(
            Animation::film_strip("strip", Duration::ZERO, true, false, vec![image('a')])
                .unwrap_err(),
            AnimationError::ZeroDuration { index: 0 }
        );
        assert_eq!(
            Animation::film_strip(
                "strip",
                Duration::from_millis(1),
                false,
                true,
                vec![image('a')]
            )
            .unwrap_err(),
            AnimationError::BounceWithoutLoop
        );
    }

    #[test]
    fn looping_animation_wraps_and_uses_strict_timing() {
        let mut animation = animation();

        let first = animation.key_frame(Duration::ZERO).unwrap();
        assert_eq!(first.index, 0);
        assert_eq!(first.animation_id, "ani");
        assert_eq!(first.image.as_ref(), image('a').as_ref());
        assert_eq!(animation.current_index(), 0);

        // Equal to the duration is not enough to switch.
        assert_eq!(
            animation
                .key_frame(Duration::from_millis(100))
                .unwrap()
                .index,
            0
        );
        assert_eq!(
            animation
                .key_frame(Duration::from_millis(101))
                .unwrap()
                .index,
            1
        );
        assert_eq!(
            animation
                .key_frame(Duration::from_millis(201))
                .unwrap()
                .index,
            1
        );
        assert_eq!(
            animation
                .key_frame(Duration::from_millis(202))
                .unwrap()
                .index,
            2
        );
        assert_eq!(
            animation
                .key_frame(Duration::from_millis(303))
                .unwrap()
                .index,
            0
        );
        assert_eq!(animation.current_index(), 0);
        assert!(animation.playing());
    }

    #[test]
    fn time_based_uses_each_frames_own_duration() {
        let mut animation = Animation::time_based(
            "mixed",
            true,
            false,
            vec![
                (image('a'), Duration::from_millis(10)),
                (image('b'), Duration::from_millis(100)),
            ],
        )
        .unwrap();

        assert_eq!(animation.key_frame(Duration::ZERO).unwrap().index, 0);
        assert_eq!(
            animation
                .key_frame(Duration::from_millis(10))
                .unwrap()
                .index,
            0
        );
        assert_eq!(
            animation
                .key_frame(Duration::from_millis(11))
                .unwrap()
                .index,
            1
        );
        assert_eq!(
            animation
                .key_frame(Duration::from_millis(111))
                .unwrap()
                .index,
            1
        );
        assert_eq!(
            animation
                .key_frame(Duration::from_millis(112))
                .unwrap()
                .index,
            0
        );
        assert_eq!(animation.frame_duration(1), Duration::from_millis(100));
    }

    #[test]
    fn bouncing_animation_reverses_at_both_ends() {
        let mut animation = Animation::time_based("bounce", true, true, time_frames()).unwrap();
        let mut indices = Vec::new();
        let mut time = 0_u64;

        indices.push(
            animation
                .key_frame(Duration::from_millis(time))
                .unwrap()
                .index,
        );
        for _ in 0..8 {
            time += 101;
            indices.push(
                animation
                    .key_frame(Duration::from_millis(time))
                    .unwrap()
                    .index,
            );
        }

        assert_eq!(indices, vec![0, 1, 2, 1, 0, 1, 2, 1, 0]);
        assert!(animation.playing());
    }

    #[test]
    fn single_frame_bounce_stays_on_the_only_frame() {
        let mut animation = Animation::time_based(
            "one",
            true,
            true,
            vec![(image('a'), Duration::from_millis(10))],
        )
        .unwrap();

        assert_eq!(animation.key_frame(Duration::ZERO).unwrap().index, 0);
        assert_eq!(
            animation
                .key_frame(Duration::from_millis(11))
                .unwrap()
                .index,
            0
        );
        assert_eq!(
            animation
                .key_frame(Duration::from_millis(22))
                .unwrap()
                .index,
            0
        );
        assert_eq!(animation.current_index(), 0);
        assert!(animation.playing());
    }

    #[test]
    fn non_looping_animation_hides_after_last_duration() {
        let mut animation = Animation::time_based("once", false, false, time_frames()).unwrap();

        assert_eq!(animation.key_frame(Duration::ZERO).unwrap().index, 0);
        assert_eq!(
            animation
                .key_frame(Duration::from_millis(101))
                .unwrap()
                .index,
            1
        );
        assert_eq!(
            animation
                .key_frame(Duration::from_millis(202))
                .unwrap()
                .index,
            2
        );

        // Last frame is returned until its own duration has elapsed.
        assert_eq!(
            animation
                .key_frame(Duration::from_millis(302))
                .unwrap()
                .index,
            2
        );
        assert!(animation.key_frame(Duration::from_millis(303)).is_none());
        assert!(!animation.playing());
        assert_eq!(animation.current_index(), 2);
        assert!(animation.key_frame(Duration::from_millis(1_000)).is_none());
    }

    #[test]
    fn pause_resume_and_reset_restart_the_frame_timer() {
        let mut animation = Animation::time_based("once", false, false, time_frames()).unwrap();
        animation.key_frame(Duration::ZERO).unwrap();
        animation.key_frame(Duration::from_millis(101)).unwrap();
        assert_eq!(animation.current_index(), 1);

        animation.set_playing(false);
        assert!(!animation.playing());
        assert!(animation.key_frame(Duration::from_millis(500)).is_none());
        assert_eq!(animation.current_index(), 1);

        animation.set_playing(true);
        assert!(animation.playing());
        assert_eq!(
            animation
                .key_frame(Duration::from_millis(1_000))
                .unwrap()
                .index,
            1
        );
        assert_eq!(
            animation
                .key_frame(Duration::from_millis(1_101))
                .unwrap()
                .index,
            2
        );

        animation.reset();
        assert!(animation.playing());
        assert_eq!(animation.current_index(), 0);
        assert_eq!(
            animation
                .key_frame(Duration::from_millis(2_000))
                .unwrap()
                .index,
            0
        );
    }

    #[test]
    fn first_last_and_foreign_frame_checks() {
        let mut animation = animation();
        let first = animation.key_frame(Duration::ZERO).unwrap();
        let last = AnimationKeyFrame {
            animation_id: "ani".to_owned(),
            image: image('c'),
            index: 2,
        };
        let foreign = AnimationKeyFrame {
            animation_id: "other".to_owned(),
            image: image('a'),
            index: 0,
        };

        assert!(animation.is_first_frame(&first));
        assert!(!animation.is_last_frame(&first));
        assert!(!animation.is_first_frame(&last));
        assert!(animation.is_last_frame(&last));
        assert!(!animation.is_first_frame(&foreign));
        assert!(!animation.is_last_frame(&foreign));
    }

    fn clip() -> VideoClip {
        VideoClip::new("clip", vec![image('a'), image('b'), image('c')], 10.0).unwrap()
    }

    #[test]
    fn video_clip_validates_inputs_and_exposes_metadata() {
        assert_eq!(
            VideoClip::new("", vec![image('a')], 10.0).unwrap_err(),
            VideoError::IdEmpty
        );
        assert_eq!(
            VideoClip::new("clip", Vec::new(), 10.0).unwrap_err(),
            VideoError::Empty
        );

        for fps in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                matches!(
                    VideoClip::new("clip", vec![image('a')], fps),
                    Err(VideoError::InvalidFps { .. })
                ),
                "fps {fps} should be rejected"
            );
        }

        let mismatch = VideoClip::new(
            "clip",
            vec![
                image('a'),
                Arc::new(Image::new(3, 2, Pixel::new('b', Color::WHITE, None))),
            ],
            10.0,
        )
        .unwrap_err();
        assert_eq!(
            mismatch,
            VideoError::DimensionMismatch {
                index: 1,
                expected: (2, 2),
                actual: (3, 2),
            }
        );
        assert!(!mismatch.to_string().is_empty());

        let clip = clip();
        assert_eq!(clip.id(), "clip");
        assert_eq!(clip.frame_count(), 3);
        assert_eq!(clip.frame_dim(), (2, 2));
        assert_eq!(clip.fps(), 10.0);
        assert_eq!(clip.frame_duration(), Duration::from_millis(100));
        assert_eq!(clip.duration(), Duration::from_millis(300));
        assert!(clip.frame(0).is_some());
        assert!(clip.frame(3).is_none());
    }

    #[test]
    fn video_clip_from_text_frames_renders_images() {
        let clip =
            VideoClip::from_text_frames("text", &["a b", "c d"], Color::WHITE, None, true, 4.0)
                .unwrap();

        assert_eq!(clip.frame_dim(), (3, 1));
        assert_eq!(clip.frame_count(), 2);
        let first = clip.frame(0).unwrap();
        assert_eq!(first.row_string(0).unwrap(), "a b");
        assert!(first.get(1, 0).unwrap().is_transparent());
        assert_eq!(clip.frame(1).unwrap().row_string(0).unwrap(), "c d");

        let mismatch =
            VideoClip::from_text_frames("text", &["a", "long"], Color::WHITE, None, false, 4.0)
                .unwrap_err();
        assert!(matches!(
            mismatch,
            VideoError::DimensionMismatch { index: 1, .. }
        ));
    }

    #[test]
    fn video_player_loops_and_emits_an_event_every_update() {
        let mut player = VideoPlayer::new(Arc::new(clip()), true, true);
        assert_eq!(player.id(), "clip");
        assert_eq!(player.frame_count(), 3);
        assert_eq!(player.frame_dim(), (2, 2));
        assert_eq!(player.frame_duration(), Duration::from_millis(100));
        assert!(player.is_playing());
        assert!(player.is_looping());

        let event = player.update(Duration::ZERO).unwrap();
        assert_eq!((event.index, event.playing), (0, true));

        // Strictly greater than the frame duration is required.
        assert_eq!(player.update(Duration::from_millis(100)).unwrap().index, 0);
        let event = player.update(Duration::from_millis(101)).unwrap();
        assert_eq!((event.index, event.playing), (1, true));
        assert_eq!(player.current_index(), 1);
        assert_eq!(player.update(Duration::from_millis(202)).unwrap().index, 2);
        assert_eq!(player.update(Duration::from_millis(303)).unwrap().index, 0);

        let (index, image) = player.update_frame(Duration::from_millis(404)).unwrap();
        assert_eq!(index, 1);
        assert_eq!(image.as_ref(), self::image('b').as_ref());
        assert_eq!(player.current_event().unwrap().index, 1);
    }

    #[test]
    fn video_player_supports_pause_toggle_seek_and_rewind() {
        let mut player = VideoPlayer::new(Arc::new(clip()), false, true);
        assert!(!player.is_playing());

        let event = player.update(Duration::ZERO).unwrap();
        assert_eq!((event.index, event.playing), (0, false));

        player.play();
        assert!(player.is_playing());
        assert_eq!(
            player.update(Duration::from_millis(1_000)).unwrap().index,
            0
        );
        assert_eq!(
            player.update(Duration::from_millis(1_101)).unwrap().index,
            1
        );

        player.stop();
        assert!(!player.is_playing());
        let event = player.update(Duration::from_millis(5_000)).unwrap();
        assert_eq!((event.index, event.playing), (1, false));
        assert_eq!(player.current_index(), 1);

        player.toggle();
        assert!(player.is_playing());
        player.rewind();
        assert_eq!(player.current_index(), 0);
        assert!(player.is_playing());
        assert_eq!(
            player.update(Duration::from_millis(6_000)).unwrap().index,
            0
        );

        assert!(player.seek_frame(2));
        assert_eq!(player.current_index(), 2);
        assert!(!player.seek_frame(3));
        assert_eq!(player.current_index(), 2);
        assert_eq!(
            player.update(Duration::from_millis(7_000)).unwrap().index,
            2
        );
        assert_eq!(
            player.update(Duration::from_millis(7_101)).unwrap().index,
            0
        );
    }

    #[test]
    fn non_looping_video_freezes_on_last_frame() {
        let mut player = VideoPlayer::new(Arc::new(clip()), true, false);
        assert_eq!(player.update(Duration::ZERO).unwrap().index, 0);
        assert_eq!(player.update(Duration::from_millis(101)).unwrap().index, 1);
        assert_eq!(player.update(Duration::from_millis(202)).unwrap().index, 2);

        let event = player.update(Duration::from_millis(303)).unwrap();
        assert_eq!((event.index, event.playing), (2, false));
        assert!(!player.is_playing());
        assert_eq!(player.current_index(), 2);

        let event = player.update(Duration::from_millis(9_999)).unwrap();
        assert_eq!((event.index, event.playing), (2, false));

        player.play();
        assert!(player.is_playing());
        assert_eq!(
            player.update(Duration::from_millis(10_000)).unwrap().index,
            2
        );
        assert_eq!(
            player.update(Duration::from_millis(10_101)).unwrap().index,
            2
        );
    }

    struct CustomVideo {
        frames: Vec<Arc<Image>>,
    }

    impl Video for CustomVideo {
        fn id(&self) -> &str {
            "custom"
        }

        fn frame_count(&self) -> usize {
            self.frames.len()
        }

        fn frame_dim(&self) -> (u16, u16) {
            (2, 2)
        }

        fn frame(&self, index: usize) -> Option<Arc<Image>> {
            self.frames.get(index).cloned()
        }
    }

    #[test]
    fn video_player_accepts_custom_video_implementations() {
        let video = Arc::new(CustomVideo {
            frames: vec![image('x'), image('y')],
        });
        let mut player = VideoPlayer::new(video, true, true);
        assert_eq!(player.id(), "custom");
        assert_eq!(player.frame_duration(), Duration::from_secs_f64(1.0 / 30.0));
        assert_eq!(player.update(Duration::ZERO).unwrap().index, 0);
        assert_eq!(player.update(Duration::from_millis(34)).unwrap().index, 1);
        assert_eq!(player.update(Duration::from_millis(68)).unwrap().index, 0);

        let empty = Arc::new(CustomVideo { frames: Vec::new() });
        let mut empty_player = VideoPlayer::new(empty, true, true);
        assert!(!empty_player.is_playing());
        assert!(empty_player.update(Duration::ZERO).is_none());
        empty_player.play();
        assert!(!empty_player.is_playing());
        assert!(!empty_player.seek_frame(0));
        assert!(empty_player.current_event().is_none());
    }
}
