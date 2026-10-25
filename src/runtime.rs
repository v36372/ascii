// SPDX-License-Identifier: Apache-2.0
//! Reusable runtime services shared by the engine, scenes, and scene objects.
//!
//! The services in this module are deliberately independent of terminal I/O and
//! of the engine's object storage. They provide the pieces of a frame context
//! that need their own deterministic state: delayed user-data caches, object
//! messages, timers, logging, rendering statistics, singleton predicates, and
//! static game metadata.
//!
//! Time is always supplied by the caller. This keeps headless tests and replay
//! runs deterministic and avoids reading a wall clock inside the services.

use crate::graphics::{Color, Pixel};
use std::any::Any;
use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::time::{Duration, Instant};

/// Nanoseconds in one millisecond.
pub const NANOS_PER_MILLIS: u64 = 1_000_000;
/// Nanoseconds in one second.
pub const NANOS_PER_SECOND: u64 = 1_000_000_000;
/// Milliseconds in one second.
pub const MILLIS_PER_SECOND: u64 = 1_000;

/// Converts a duration to whole milliseconds, saturating at [`u64::MAX`].
pub fn duration_millis(duration: Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}

/// Converts a duration to whole nanoseconds, saturating at [`u64::MAX`].
pub fn duration_nanos(duration: Duration) -> u64 {
    duration.as_nanos().min(u128::from(u64::MAX)) as u64
}

/// Computes an integer frames-per-second value from a frame duration.
///
/// A zero duration has no meaningful rate and returns `0`. Rates above
/// [`u32::MAX`] saturate.
pub fn fps_from_duration(duration: Duration) -> u32 {
    let nanos = duration.as_nanos();
    let fps = u128::from(NANOS_PER_SECOND)
        .checked_div(nanos)
        .unwrap_or(0)
        .min(u128::from(u32::MAX));
    fps as u32
}

/// The default number of frame samples retained by [`RenderStats`].
pub const DEFAULT_FPS_WINDOW: usize = 300;

/// A map-like container for game-wide and scene-wide user data.
///
/// Mutations are queued. [`Cache::get`] and [`Cache::contains`] only observe
/// values from the last [`Cache::apply`], so scene objects see a stable snapshot
/// for the whole frame. The engine calls [`Cache::apply`] at a frame boundary.
/// Operations are applied in call order, so a `put("x", 1)` followed by
/// `remove("x")` leaves the key absent when both become visible together.
///
/// Keys are stored in a [`BTreeMap`] so filtering and value mapping visit keys
/// in sorted order, independent of hashing seeds or insertion order.
#[derive(Default)]
pub struct Cache {
    values: BTreeMap<String, Box<dyn Any>>,
    pending: Vec<CacheOp>,
}

enum CacheOp {
    Put(String, Box<dyn Any>),
    Remove(String),
    Clear,
    Filter(Box<FilterFn>),
    MapValues(Box<MapValuesFn>),
}

type FilterFn = dyn FnMut(&str, &dyn Any) -> bool;
type MapValuesFn = dyn FnMut(&str, &dyn Any) -> Box<dyn Any>;

impl Cache {
    /// Creates an empty cache with no pending operations.
    pub fn new() -> Self {
        Self::default()
    }

    /// Queues a value for insertion.
    ///
    /// The value replaces any current or earlier queued value for `key` when
    /// [`Cache::apply`] reaches this operation.
    pub fn put<T: Any>(&mut self, key: impl Into<String>, value: T) {
        self.pending.push(CacheOp::Put(key.into(), Box::new(value)));
    }

    /// Returns the applied value for `key`, or `None` if it is absent or has a
    /// different type than `T`.
    pub fn get<T: Any>(&self, key: &str) -> Option<&T> {
        self.values.get(key)?.downcast_ref::<T>()
    }

    /// Clones the applied value for `key`, if present and of type `T`.
    pub fn get_cloned<T: Clone + Any>(&self, key: &str) -> Option<T> {
        self.get::<T>(key).cloned()
    }

    /// Tests whether an applied value exists for `key`.
    pub fn contains(&self, key: &str) -> bool {
        self.values.contains_key(key)
    }

    /// Queues removal of `key`.
    pub fn remove(&mut self, key: impl Into<String>) {
        self.pending.push(CacheOp::Remove(key.into()));
    }

    /// Queues removal of all applied entries.
    ///
    /// Operations queued after this call still apply normally.
    pub fn clear(&mut self) {
        self.pending.push(CacheOp::Clear);
    }

    /// Queues a deterministic filter over the applied entries.
    ///
    /// The predicate is evaluated when [`Cache::apply`] runs, in ascending key
    /// order. Entries for which it returns `false` are removed.
    pub fn filter<F>(&mut self, predicate: F)
    where
        F: FnMut(&str, &dyn Any) -> bool + 'static,
    {
        self.pending.push(CacheOp::Filter(Box::new(predicate)));
    }

    /// Queues a deterministic transformation of every applied value.
    ///
    /// The function runs when [`Cache::apply`] runs, in ascending key order. It
    /// receives the key and the current boxed value and returns its replacement.
    pub fn map_values<F>(&mut self, transform: F)
    where
        F: FnMut(&str, &dyn Any) -> Box<dyn Any> + 'static,
    {
        self.pending.push(CacheOp::MapValues(Box::new(transform)));
    }

    /// Applies every queued operation in call order and returns their count.
    ///
    /// New operations queued by a callback during application are left pending
    /// for the next call.
    pub fn apply(&mut self) -> usize {
        let pending = std::mem::take(&mut self.pending);
        let count = pending.len();
        for op in pending {
            match op {
                CacheOp::Put(key, value) => {
                    self.values.insert(key, value);
                }
                CacheOp::Remove(key) => {
                    self.values.remove(&key);
                }
                CacheOp::Clear => {
                    self.values.clear();
                }
                CacheOp::Filter(mut predicate) => {
                    self.values
                        .retain(|key, value| predicate(key, value.as_ref()));
                }
                CacheOp::MapValues(mut transform) => {
                    for (key, value) in &mut self.values {
                        *value = transform(key, value.as_ref());
                    }
                }
            }
        }
        count
    }

    /// Immediately clears applied values and discards pending operations.
    ///
    /// This is intended for scene resets, where work queued by the outgoing
    /// scene must not leak into the incoming scene's cache.
    pub fn clear_now(&mut self) {
        self.values.clear();
        self.pending.clear();
    }

    /// Number of currently applied entries.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns `true` if no applied entries exist.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Number of operations waiting for [`Cache::apply`].
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }
}

/// A deterministic mailbox keyed by scene-object ID.
///
/// Messages remain queued until the recipient calls [`MessageQueue::receive`]
/// or the queue is cleared, even if the ID does not currently exist. Sends to
/// an existing mailbox append in send order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MessageQueue<T> {
    mailboxes: BTreeMap<String, Vec<T>>,
}

impl<T> MessageQueue<T> {
    /// Creates an empty message queue.
    pub fn new() -> Self {
        Self {
            mailboxes: BTreeMap::new(),
        }
    }

    /// Appends all `messages` to the mailbox for `id`.
    ///
    /// Empty message lists do not create a mailbox.
    pub fn send(&mut self, id: impl Into<String>, messages: impl IntoIterator<Item = T>) {
        let messages: Vec<T> = messages.into_iter().collect();
        if messages.is_empty() {
            return;
        }
        self.mailboxes
            .entry(id.into())
            .or_default()
            .extend(messages);
    }

    /// Appends one message to the mailbox for `id`.
    pub fn send_one(&mut self, id: impl Into<String>, message: T) {
        self.mailboxes.entry(id.into()).or_default().push(message);
    }

    /// Removes and returns every message for `id` in send order.
    ///
    /// An unknown or already drained ID returns an empty vector.
    pub fn receive(&mut self, id: &str) -> Vec<T> {
        self.mailboxes.remove(id).unwrap_or_default()
    }

    /// Number of pending messages for one ID.
    pub fn len_for(&self, id: &str) -> usize {
        self.mailboxes.get(id).map_or(0, Vec::len)
    }

    /// Total number of pending messages across all IDs.
    pub fn len(&self) -> usize {
        self.mailboxes.values().map(Vec::len).sum()
    }

    /// Returns `true` if no mailbox contains a pending message.
    pub fn is_empty(&self) -> bool {
        self.mailboxes.values().all(Vec::is_empty)
    }

    /// Discards all messages for every ID.
    pub fn clear(&mut self) {
        self.mailboxes.clear();
    }

    /// Discards the messages for one ID and returns how many were removed.
    pub fn clear_for(&mut self, id: &str) -> usize {
        self.mailboxes
            .remove(id)
            .map_or(0, |messages| messages.len())
    }
}

/// Timers and next-frame callbacks scoped to one active scene.
///
/// Delayed timers are stored in `(due_ms, sequence)` order. The monotonically
/// increasing sequence breaks ties in scheduling order, so equal deadlines are
/// deterministic. Callbacks are `FnOnce` and run only when a `drain*`, `run*`,
/// or [`TimerQueue::flush_all`] method executes them.
///
/// The engine runs due timers before object updates, then runs next-frame
/// callbacks. On a scene switch it calls [`TimerQueue::flush_all`]: pending
/// `run_later` callbacks execute immediately, while queued next-frame callbacks
/// are discarded, matching CosPlay's scene-switch behavior.
#[derive(Default)]
pub struct TimerQueue {
    now_ms: u64,
    next_seq: u64,
    delayed: BTreeMap<(u64, u64), Box<dyn FnOnce()>>,
    next_frame: Vec<Box<dyn FnOnce()>>,
}

impl TimerQueue {
    /// Creates an empty queue whose current time is zero milliseconds.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty queue at the given current time.
    pub fn with_now(now_ms: u64) -> Self {
        Self {
            now_ms,
            ..Self::default()
        }
    }

    /// Current scheduling time in milliseconds.
    pub fn now_ms(&self) -> u64 {
        self.now_ms
    }

    /// Advances the scheduling time, never moving it backwards.
    ///
    /// Returns the effective current time.
    pub fn set_now(&mut self, now_ms: u64) -> u64 {
        self.now_ms = self.now_ms.max(now_ms);
        self.now_ms
    }

    /// Schedules `callback` for at least `delay_ms` after the current time.
    pub fn run_later<F>(&mut self, delay_ms: u64, callback: F)
    where
        F: FnOnce() + 'static,
    {
        let due = self.now_ms.saturating_add(delay_ms);
        let sequence = self.next_seq;
        self.next_seq = self
            .next_seq
            .checked_add(1)
            .expect("timer sequence exhausted");
        self.delayed.insert((due, sequence), Box::new(callback));
    }

    /// Alias for [`TimerQueue::run_later`].
    pub fn schedule_later<F>(&mut self, delay_ms: u64, callback: F)
    where
        F: FnOnce() + 'static,
    {
        self.run_later(delay_ms, callback);
    }

    /// Queues `callback` for the next frame boundary of the current scene.
    pub fn run_next_frame<F>(&mut self, callback: F)
    where
        F: FnOnce() + 'static,
    {
        self.next_frame.push(Box::new(callback));
    }

    /// Alias for [`TimerQueue::run_next_frame`].
    pub fn schedule_next_frame<F>(&mut self, callback: F)
    where
        F: FnOnce() + 'static,
    {
        self.run_next_frame(callback);
    }

    /// Advances the clock to `now_ms` and runs every delayed timer due by then.
    ///
    /// Timers run in `(due, sequence)` order. Timers scheduled during this call
    /// are not run until a later call, even if their deadline is already due.
    /// Returns the number of callbacks executed.
    pub fn run_due(&mut self, now_ms: u64) -> usize {
        let now = self.set_now(now_ms);
        let due: Vec<(u64, u64)> = self
            .delayed
            .range(..=(now, u64::MAX))
            .map(|(&key, _)| key)
            .collect();
        let count = due.len();
        for key in due {
            if let Some(callback) = self.delayed.remove(&key) {
                callback();
            }
        }
        count
    }

    /// Alias for [`TimerQueue::run_due`].
    pub fn drain_due(&mut self, now_ms: u64) -> usize {
        self.run_due(now_ms)
    }

    /// Runs and clears every queued next-frame callback in scheduling order.
    ///
    /// Returns the number of callbacks executed.
    pub fn drain_next_frame(&mut self) -> usize {
        let callbacks = std::mem::take(&mut self.next_frame);
        let count = callbacks.len();
        for callback in callbacks {
            callback();
        }
        count
    }

    /// Runs due delayed timers followed by all next-frame callbacks.
    ///
    /// This is the normal frame-boundary entry point. The order matches the
    /// engine's pre-update timer pass. Returns the total callbacks executed.
    pub fn drain_frame(&mut self, now_ms: u64) -> usize {
        self.run_due(now_ms) + self.drain_next_frame()
    }

    /// Executes every delayed timer immediately, regardless of its deadline,
    /// then discards queued next-frame callbacks.
    ///
    /// Delayed timers still run in `(due, sequence)` order. Returns the number
    /// of delayed callbacks executed. This is the scene-switch behavior.
    pub fn flush_all(&mut self) -> usize {
        let delayed = std::mem::take(&mut self.delayed);
        let count = delayed.len();
        for (_, callback) in delayed {
            callback();
        }
        self.next_frame.clear();
        count
    }

    /// Alias for [`TimerQueue::flush_all`].
    pub fn flush(&mut self) -> usize {
        self.flush_all()
    }

    /// Discards all delayed and next-frame callbacks without running them.
    pub fn clear(&mut self) {
        self.delayed.clear();
        self.next_frame.clear();
    }

    /// Number of delayed timers that have not run yet.
    pub fn pending_later(&self) -> usize {
        self.delayed.len()
    }

    /// Number of queued next-frame callbacks.
    pub fn pending_next_frame(&self) -> usize {
        self.next_frame.len()
    }

    /// Returns `true` if no callback is pending.
    pub fn is_empty(&self) -> bool {
        self.delayed.is_empty() && self.next_frame.is_empty()
    }
}

/// Severity of a [`LogRecord`].
///
/// The order runs from most to least severe. [`Log::set_min_level`] uses that
/// order when selecting a contiguous enabled range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LogLevel {
    /// Recoverable or fatal failures.
    Error,
    /// Unexpected conditions that do not stop execution.
    Warn,
    /// Normal high-level lifecycle information.
    Info,
    /// Developer diagnostics.
    Debug,
    /// Detailed per-frame tracing.
    Trace,
}

impl LogLevel {
    /// Every level from most to least severe.
    pub const ALL: [Self; 5] = [
        Self::Error,
        Self::Warn,
        Self::Info,
        Self::Debug,
        Self::Trace,
    ];

    /// Uppercase display name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Error => "ERROR",
            Self::Warn => "WARN",
            Self::Info => "INFO",
            Self::Debug => "DEBUG",
            Self::Trace => "TRACE",
        }
    }

    const fn index(self) -> usize {
        match self {
            Self::Error => 0,
            Self::Warn => 1,
            Self::Info => 2,
            Self::Debug => 3,
            Self::Trace => 4,
        }
    }
}

impl fmt::Display for LogLevel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// One captured log entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogRecord {
    /// Severity of the message.
    pub level: LogLevel,
    /// Category that emitted the message.
    pub category: String,
    /// Rendered message text.
    pub message: String,
    /// Global frame at which the message was emitted.
    pub frame: u64,
    /// Frame timestamp in milliseconds.
    pub now_ms: u64,
}

/// In-memory game logger with categories, level filtering, and throttling.
///
/// The logger never writes to stdout/stderr because that would corrupt the
/// terminal renderer. Records are retained for a built-in viewer or tests.
/// Callers supply frame and time values, keeping output deterministic.
pub struct Log {
    category: String,
    enabled: [bool; 5],
    records: Vec<LogRecord>,
}

impl Default for Log {
    fn default() -> Self {
        Self {
            category: String::from("default"),
            enabled: [true; 5],
            records: Vec::new(),
        }
    }
}

impl Log {
    /// Creates a logger for `category` with all levels enabled.
    pub fn new(category: impl Into<String>) -> Self {
        Self {
            category: category.into(),
            ..Self::default()
        }
    }

    /// Category used by methods that do not pass an explicit category.
    pub fn category(&self) -> &str {
        &self.category
    }

    /// Replaces the default category.
    pub fn set_category(&mut self, category: impl Into<String>) {
        self.category = category.into();
    }

    /// Enables one level.
    pub fn enable(&mut self, level: LogLevel) {
        self.enabled[level.index()] = true;
    }

    /// Disables one level.
    pub fn disable(&mut self, level: LogLevel) {
        self.enabled[level.index()] = false;
    }

    /// Enables all levels.
    pub fn enable_all(&mut self) {
        self.enabled = [true; 5];
    }

    /// Disables all levels.
    pub fn disable_all(&mut self) {
        self.enabled = [false; 5];
    }

    /// Replaces the enabled set with all levels at least as severe as `minimum`.
    ///
    /// For example, `Info` enables `Error`, `Warn`, and `Info`; `Trace` enables
    /// everything. Levels can still be adjusted individually afterwards.
    pub fn set_min_level(&mut self, minimum: LogLevel) {
        for level in LogLevel::ALL {
            self.enabled[level.index()] = level <= minimum;
        }
    }

    /// Tests whether `level` is currently enabled.
    pub fn is_enabled(&self, level: LogLevel) -> bool {
        self.enabled[level.index()]
    }

    /// Logs with the default category and returns whether a record was added.
    pub fn log(
        &mut self,
        level: LogLevel,
        message: impl Into<String>,
        frame: u64,
        now_ms: u64,
    ) -> bool {
        let category = self.category.clone();
        self.log_in(level, category, message, frame, now_ms)
    }

    /// Logs with an explicit category and returns whether a record was added.
    pub fn log_in(
        &mut self,
        level: LogLevel,
        category: impl Into<String>,
        message: impl Into<String>,
        frame: u64,
        now_ms: u64,
    ) -> bool {
        if !self.is_enabled(level) {
            return false;
        }
        self.records.push(LogRecord {
            level,
            category: category.into(),
            message: message.into(),
            frame,
            now_ms,
        });
        true
    }

    /// Alias for [`Log::log_in`].
    pub fn log_with_category(
        &mut self,
        level: LogLevel,
        category: impl Into<String>,
        message: impl Into<String>,
        frame: u64,
        now_ms: u64,
    ) -> bool {
        self.log_in(level, category, message, frame, now_ms)
    }

    /// Logs an error with the default category.
    pub fn error(&mut self, message: impl Into<String>, frame: u64, now_ms: u64) -> bool {
        self.log(LogLevel::Error, message, frame, now_ms)
    }

    /// Logs a warning with the default category.
    pub fn warn(&mut self, message: impl Into<String>, frame: u64, now_ms: u64) -> bool {
        self.log(LogLevel::Warn, message, frame, now_ms)
    }

    /// Logs information with the default category.
    pub fn info(&mut self, message: impl Into<String>, frame: u64, now_ms: u64) -> bool {
        self.log(LogLevel::Info, message, frame, now_ms)
    }

    /// Logs a debug message with the default category.
    pub fn debug(&mut self, message: impl Into<String>, frame: u64, now_ms: u64) -> bool {
        self.log(LogLevel::Debug, message, frame, now_ms)
    }

    /// Logs a trace message with the default category.
    pub fn trace(&mut self, message: impl Into<String>, frame: u64, now_ms: u64) -> bool {
        self.log(LogLevel::Trace, message, frame, now_ms)
    }

    /// Logs on every `n`th frame, using the default category.
    ///
    /// This matches CosPlay's throttle rule `frame % n == 0`, including frame
    /// zero. A throttle of zero is treated as one. Returns whether a record was
    /// added.
    pub fn log_every(
        &mut self,
        n: u64,
        level: LogLevel,
        message: impl Into<String>,
        frame: u64,
        now_ms: u64,
    ) -> bool {
        let n = n.max(1);
        if frame.is_multiple_of(n) {
            self.log(level, message, frame, now_ms)
        } else {
            false
        }
    }

    /// Logs on every `n`th frame with an explicit category.
    pub fn log_every_in(
        &mut self,
        n: u64,
        level: LogLevel,
        category: impl Into<String>,
        message: impl Into<String>,
        frame: u64,
        now_ms: u64,
    ) -> bool {
        let n = n.max(1);
        if frame.is_multiple_of(n) {
            self.log_in(level, category, message, frame, now_ms)
        } else {
            false
        }
    }

    /// Runs `operation`, then logs its wall-clock duration at debug level.
    ///
    /// The returned value is the operation's result. The log record carries the
    /// caller-supplied frame and timestamp. Duration measurement uses
    /// [`std::time::Instant`], so only the logged duration is non-deterministic.
    pub fn perf<T>(
        &mut self,
        label: &str,
        frame: u64,
        now_ms: u64,
        operation: impl FnOnce() -> T,
    ) -> T {
        let start = Instant::now();
        let result = operation();
        let elapsed = start.elapsed();
        self.debug(
            format!("{label}: {}ns", duration_nanos(elapsed)),
            frame,
            now_ms,
        );
        result
    }

    /// All records in emission order.
    pub fn records(&self) -> &[LogRecord] {
        &self.records
    }

    /// Removes and returns all records in emission order.
    pub fn drain_records(&mut self) -> Vec<LogRecord> {
        std::mem::take(&mut self.records)
    }

    /// Number of retained records.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Returns `true` if no record has been retained.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Discards all retained records.
    pub fn clear(&mut self) {
        self.records.clear();
    }
}

/// Rendering statistics for the current frame and a bounded recent history.
///
/// FPS samples are integer frames per second. The average is the truncated mean
/// of the samples in the window. The low 1% value is the truncated mean of the
/// lowest `max(1, window_len / 100)` samples, matching CosPlay's "average across
/// the lowest 1%" behavior. At most [`RenderStats::capacity`] samples are kept.
#[derive(Debug, Clone)]
pub struct RenderStats {
    /// Global frame count since game start.
    pub frame: u64,
    /// Frame count since the current scene started.
    pub scene_frame: u64,
    /// Frames per second for the most recent frame.
    pub fps: u32,
    /// Mean frames per second over the retained window.
    pub avg_fps: u32,
    /// Mean of the slowest 1% of frames in the retained window.
    pub low_1pct_fps: u32,
    /// Duration of the most recent frame in nanoseconds.
    pub frame_duration_nanos: u64,
    /// Object count reported for the most recent frame.
    pub object_count: u64,
    /// Scene count reported for the most recent frame.
    pub scene_count: u64,
    samples: VecDeque<u32>,
    capacity: usize,
}

impl Default for RenderStats {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderStats {
    /// Creates empty statistics using [`DEFAULT_FPS_WINDOW`] samples.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_FPS_WINDOW)
    }

    /// Creates empty statistics with a custom sample window.
    ///
    /// A capacity of zero is clamped to one so that a current sample always
    /// exists.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            frame: 0,
            scene_frame: 0,
            fps: 0,
            avg_fps: 0,
            low_1pct_fps: 0,
            frame_duration_nanos: 0,
            object_count: 0,
            scene_count: 0,
            samples: VecDeque::new(),
            capacity: capacity.max(1),
        }
    }

    /// Maximum number of retained FPS samples.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of retained FPS samples.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Retained FPS samples from oldest to newest.
    pub fn samples(&self) -> impl Iterator<Item = u32> + '_ {
        self.samples.iter().copied()
    }

    /// Records one frame and updates every derived statistic.
    ///
    /// Both frame counters advance by one. `delta` is the frame duration,
    /// `objects` is the object count, and `scenes` is the scene count.
    pub fn record_frame(&mut self, delta: Duration, objects: u64, scenes: u64) {
        self.frame = self.frame.saturating_add(1);
        self.scene_frame = self.scene_frame.saturating_add(1);
        self.frame_duration_nanos = duration_nanos(delta);
        self.object_count = objects;
        self.scene_count = scenes;
        self.fps = fps_from_duration(delta);

        if self.samples.len() >= self.capacity {
            self.samples.pop_front();
        }
        self.samples.push_back(self.fps);
        self.recalculate();
    }

    fn recalculate(&mut self) {
        let count = self.samples.len();
        if count == 0 {
            self.avg_fps = 0;
            self.low_1pct_fps = 0;
            return;
        }

        let sum: u64 = self.samples.iter().map(|&fps| u64::from(fps)).sum();
        self.avg_fps = (sum / count as u64) as u32;

        let mut sorted: Vec<u32> = self.samples.iter().copied().collect();
        sorted.sort_unstable();
        let low_count = (count / 100).max(1);
        let low_sum: u64 = sorted
            .iter()
            .take(low_count)
            .map(|&fps| u64::from(fps))
            .sum();
        self.low_1pct_fps = (low_sum / low_count as u64) as u32;
    }

    /// Resets the scene-local frame counter without clearing global FPS history.
    pub fn reset_scene(&mut self) {
        self.scene_frame = 0;
    }

    /// Resets all counters and clears retained FPS samples.
    ///
    /// The configured window capacity is preserved.
    pub fn reset(&mut self) {
        self.frame = 0;
        self.scene_frame = 0;
        self.fps = 0;
        self.avg_fps = 0;
        self.low_1pct_fps = 0;
        self.frame_duration_nanos = 0;
        self.object_count = 0;
        self.scene_count = 0;
        self.samples.clear();
    }
}

/// The lifetime scope of a singleton callback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SingletonScope {
    /// Run once at the start of the game, when the global frame is zero.
    Game,
    /// Run once at the start of each scene, when the scene frame is zero.
    Scene,
    /// Run once for the individual object, until it reports being touched.
    Object,
}

impl SingletonScope {
    /// Returns whether a singleton callback should run in this frame.
    ///
    /// `touched` is owned by the singleton object and becomes `true` after its
    /// first callback.
    pub const fn should_run(self, global_frame: u64, scene_frame: u64, touched: bool) -> bool {
        match self {
            Self::Game => global_frame == 0,
            Self::Scene => scene_frame == 0,
            Self::Object => !touched,
        }
    }
}

/// Free-function form of [`SingletonScope::should_run`].
pub const fn should_run(
    scope: SingletonScope,
    global_frame: u64,
    scene_frame: u64,
    touched: bool,
) -> bool {
    scope.should_run(global_frame, scene_frame, touched)
}

/// Descriptor of a game's stable metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GameInfo {
    /// Stable technical ID used for persistence and log directories.
    pub id: String,
    /// Public display name.
    pub name: String,
    /// Game version, conventionally semantic versioning.
    pub version: String,
    /// Optional initial dimensions for a built-in terminal emulator.
    pub emulator_dim: Option<(u16, u16)>,
    /// Terminal background pixel.
    pub background: Pixel,
    /// Smallest supported native terminal size.
    pub min_terminal_dim: (u16, u16),
}

impl Default for GameInfo {
    fn default() -> Self {
        Self {
            id: String::from("cosplay"),
            name: String::from("CosPlay"),
            version: String::from("1.0.0"),
            emulator_dim: None,
            background: Pixel::new(' ', Color::WHITE, Some(Color::rgb(0x11, 0x11, 0x11))),
            min_terminal_dim: (1, 1),
        }
    }
}

/// Invalid [`GameInfo`] field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameInfoError {
    /// The technical game ID is empty.
    EmptyId,
    /// The display name is empty.
    EmptyName,
    /// The version string is empty.
    EmptyVersion,
    /// The emulator width is zero.
    ZeroEmulatorWidth,
    /// The emulator height is zero.
    ZeroEmulatorHeight,
    /// The minimum terminal width is zero.
    ZeroMinTerminalWidth,
    /// The minimum terminal height is zero.
    ZeroMinTerminalHeight,
}

impl fmt::Display for GameInfoError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::EmptyId => "game ID cannot be empty",
            Self::EmptyName => "game name cannot be empty",
            Self::EmptyVersion => "game version cannot be empty",
            Self::ZeroEmulatorWidth => "emulator width cannot be zero",
            Self::ZeroEmulatorHeight => "emulator height cannot be zero",
            Self::ZeroMinTerminalWidth => "minimum terminal width cannot be zero",
            Self::ZeroMinTerminalHeight => "minimum terminal height cannot be zero",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for GameInfoError {}

impl GameInfo {
    /// Creates metadata with a display name and otherwise default fields.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    /// Validates required text fields and non-zero dimensions.
    pub fn validate(&self) -> Result<(), GameInfoError> {
        if self.id.trim().is_empty() {
            return Err(GameInfoError::EmptyId);
        }
        if self.name.trim().is_empty() {
            return Err(GameInfoError::EmptyName);
        }
        if self.version.trim().is_empty() {
            return Err(GameInfoError::EmptyVersion);
        }
        if let Some((width, height)) = self.emulator_dim {
            if width == 0 {
                return Err(GameInfoError::ZeroEmulatorWidth);
            }
            if height == 0 {
                return Err(GameInfoError::ZeroEmulatorHeight);
            }
        }
        if self.min_terminal_dim.0 == 0 {
            return Err(GameInfoError::ZeroMinTerminalWidth);
        }
        if self.min_terminal_dim.1 == 0 {
            return Err(GameInfoError::ZeroMinTerminalHeight);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn events() -> Rc<RefCell<Vec<String>>> {
        Rc::new(RefCell::new(Vec::new()))
    }

    fn push_event(
        events: &Rc<RefCell<Vec<String>>>,
        label: &'static str,
    ) -> impl FnOnce() + 'static {
        let events = Rc::clone(events);
        move || events.borrow_mut().push(String::from(label))
    }

    #[test]
    fn cache_mutations_are_delayed_and_ordered() {
        let mut cache = Cache::new();
        cache.put("hp", 10i32);
        assert_eq!(cache.get::<i32>("hp"), None);
        assert!(!cache.contains("hp"));
        assert_eq!(cache.pending_len(), 1);

        assert_eq!(cache.apply(), 1);
        assert_eq!(cache.get::<i32>("hp"), Some(&10));
        assert_eq!(cache.get_cloned::<i32>("hp"), Some(10));
        assert_eq!(cache.get::<u32>("hp"), None);
        assert!(cache.contains("hp"));

        cache.put("hp", 20i32);
        assert_eq!(cache.get_cloned::<i32>("hp"), Some(10));
        cache.apply();
        assert_eq!(cache.get_cloned::<i32>("hp"), Some(20));

        cache.put("temporary", true);
        cache.remove("temporary");
        cache.put("hp", 30i32);
        cache.remove("hp");
        cache.put("hp", 40i32);
        cache.apply();
        assert!(!cache.contains("temporary"));
        assert_eq!(cache.get_cloned::<i32>("hp"), Some(40));

        cache.clear();
        assert!(cache.contains("hp"));
        cache.apply();
        assert!(cache.is_empty());
    }

    #[test]
    fn cache_filter_and_map_values_are_deterministic() {
        let mut cache = Cache::new();
        cache.put("b", 2i32);
        cache.put("a", 1i32);
        cache.put("c", 3i32);
        cache.apply();

        let seen = events();
        let filter_seen = Rc::clone(&seen);
        cache.filter(move |key, value| {
            filter_seen.borrow_mut().push(key.to_string());
            key != "b" && value.downcast_ref::<i32>().is_some()
        });
        assert!(cache.contains("b"));
        cache.apply();
        assert_eq!(
            *seen.borrow(),
            [String::from("a"), String::from("b"), String::from("c")]
        );
        assert_eq!(cache.len(), 2);
        assert!(!cache.contains("b"));

        cache.map_values(|_, value| {
            let value = value.downcast_ref::<i32>().copied().unwrap_or_default();
            Box::new(value * 10)
        });
        cache.apply();
        assert_eq!(cache.get_cloned::<i32>("a"), Some(10));
        assert_eq!(cache.get_cloned::<i32>("c"), Some(30));
    }

    #[test]
    fn cache_clear_now_discards_values_and_pending_work() {
        let mut cache = Cache::new();
        cache.put("old", 1i32);
        cache.apply();
        cache.put("new", 2i32);
        cache.clear_now();

        assert!(cache.is_empty());
        assert_eq!(cache.pending_len(), 0);
        cache.put("late", 3i32);
        cache.apply();
        assert_eq!(cache.get_cloned::<i32>("late"), Some(3));
        assert!(!cache.contains("new"));
    }

    #[test]
    fn messages_drain_per_id_and_retain_unknown_ids() {
        let mut queue = MessageQueue::new();
        queue.send("alpha", vec![1, 2]);
        queue.send("alpha", vec![3]);
        queue.send("beta", vec![9]);
        queue.send("empty", Vec::<i32>::new());

        assert_eq!(queue.len(), 4);
        assert_eq!(queue.len_for("alpha"), 3);
        assert_eq!(queue.len_for("unknown"), 0);

        assert_eq!(queue.receive("alpha"), vec![1, 2, 3]);
        assert_eq!(queue.receive("alpha"), Vec::<i32>::new());
        assert_eq!(queue.len(), 1);
        assert_eq!(queue.receive("unknown"), Vec::<i32>::new());
        assert_eq!(queue.len(), 1);

        queue.send_one("beta", 10);
        assert_eq!(queue.len_for("beta"), 2);
        assert_eq!(queue.clear_for("beta"), 2);
        assert!(queue.is_empty());

        queue.send("gamma", vec![11i32]);
        queue.clear();
        assert!(queue.is_empty());
        assert_eq!(queue.receive("gamma"), Vec::<i32>::new());
    }

    #[test]
    fn timers_run_in_due_then_sequence_order() {
        let log = events();
        let mut timers = TimerQueue::new();
        timers.run_later(10, push_event(&log, "first-at-10"));
        timers.run_later(5, push_event(&log, "at-5"));
        timers.run_later(10, push_event(&log, "second-at-10"));
        timers.run_next_frame(push_event(&log, "next-1"));
        timers.run_next_frame(push_event(&log, "next-2"));

        assert_eq!(timers.drain_due(4), 0);
        assert_eq!(timers.pending_later(), 3);
        assert_eq!(timers.drain_due(9), 1);
        assert_eq!(*log.borrow(), ["at-5"]);
        assert_eq!(timers.drain_due(10), 2);
        assert_eq!(*log.borrow(), ["at-5", "first-at-10", "second-at-10"]);
        assert_eq!(timers.pending_later(), 0);
        assert_eq!(timers.pending_next_frame(), 2);

        assert_eq!(timers.drain_next_frame(), 2);
        assert_eq!(
            *log.borrow(),
            ["at-5", "first-at-10", "second-at-10", "next-1", "next-2"]
        );
        assert!(timers.is_empty());
    }

    #[test]
    fn timer_schedules_are_relative_and_clock_is_monotonic() {
        let log = events();
        let mut timers = TimerQueue::with_now(50);
        timers.run_later(10, push_event(&log, "at-60"));

        assert_eq!(timers.set_now(20), 50);
        assert_eq!(timers.now_ms(), 50);
        assert_eq!(timers.run_due(59), 0);
        assert_eq!(timers.run_due(60), 1);
        assert_eq!(*log.borrow(), ["at-60"]);
    }

    #[test]
    fn timer_flush_runs_later_but_discards_next_frame() {
        let log = events();
        let mut timers = TimerQueue::new();
        timers.run_later(1_000, push_event(&log, "far"));
        timers.run_later(5, push_event(&log, "near"));
        timers.run_next_frame(push_event(&log, "discarded"));

        assert_eq!(timers.flush_all(), 2);
        assert_eq!(*log.borrow(), ["near", "far"]);
        assert!(timers.is_empty());

        assert_eq!(timers.flush(), 0);
        assert_eq!(*log.borrow(), ["near", "far"]);
    }

    #[test]
    fn timer_drain_frame_runs_due_before_next_frame() {
        let log = events();
        let mut timers = TimerQueue::with_now(100);
        timers.run_later(10, push_event(&log, "due"));
        timers.run_next_frame(push_event(&log, "next"));

        assert_eq!(timers.drain_frame(109), 1);
        assert_eq!(*log.borrow(), ["next"]);
        assert_eq!(timers.drain_frame(110), 1);
        assert_eq!(*log.borrow(), ["next", "due"]);
        assert!(timers.is_empty());
    }

    #[test]
    fn log_levels_categories_and_throttling() {
        let mut log = Log::new("game");
        log.set_min_level(LogLevel::Info);

        assert!(log.error("boom", 0, 10));
        assert!(log.warn("careful", 1, 20));
        assert!(log.info("started", 2, 30));
        assert!(!log.debug("hidden", 3, 40));
        assert!(!log.trace("hidden", 4, 50));
        assert_eq!(log.len(), 3);
        assert_eq!(log.records()[0].level, LogLevel::Error);
        assert_eq!(log.records()[0].category, "game");
        assert_eq!(log.records()[0].message, "boom");
        assert_eq!(log.records()[0].frame, 0);
        assert_eq!(log.records()[0].now_ms, 10);

        log.enable(LogLevel::Debug);
        assert!(log.debug("visible", 5, 60));
        assert!(log.log_in(LogLevel::Warn, "ui", "category", 6, 70));
        assert_eq!(log.records()[4].category, "ui");

        log.clear();
        log.enable_all();
        for frame in 0..7 {
            log.log_every(
                3,
                LogLevel::Info,
                format!("frame-{frame}"),
                frame,
                frame * 10,
            );
        }
        let frames: Vec<u64> = log.records().iter().map(|record| record.frame).collect();
        assert_eq!(frames, [0, 3, 6]);

        log.clear();
        assert!(log.log_every(0, LogLevel::Info, "zero-throttle", 8, 80));
        assert_eq!(log.len(), 1);

        log.set_min_level(LogLevel::Error);
        assert!(!log.log_every(3, LogLevel::Info, "disabled", 9, 90));
        assert!(!log.log_every_in(3, LogLevel::Info, "other", "disabled", 9, 90));
    }

    #[test]
    fn log_perf_returns_result_and_records_duration() {
        let mut log = Log::new("perf");
        let result = log.perf("answer", 7, 700, || 6 * 7);
        assert_eq!(result, 42);
        assert_eq!(log.len(), 1);
        let record = &log.records()[0];
        assert_eq!(record.level, LogLevel::Debug);
        assert_eq!(record.category, "perf");
        assert_eq!(record.frame, 7);
        assert_eq!(record.now_ms, 700);
        assert!(record.message.starts_with("answer: "));
        assert!(record.message.ends_with("ns"));
    }

    #[test]
    fn render_stats_tracks_window_average_and_low_percentile() {
        let mut stats = RenderStats::with_capacity(4);
        assert_eq!(stats.capacity(), 4);
        assert_eq!(stats.sample_count(), 0);

        stats.record_frame(Duration::from_millis(100), 3, 1);
        stats.record_frame(Duration::from_millis(50), 4, 1);
        stats.record_frame(Duration::from_millis(40), 5, 2);
        stats.record_frame(Duration::from_millis(20), 6, 2);

        assert_eq!(stats.frame, 4);
        assert_eq!(stats.scene_frame, 4);
        assert_eq!(stats.fps, 50);
        assert_eq!(stats.avg_fps, 26);
        assert_eq!(stats.low_1pct_fps, 10);
        assert_eq!(stats.frame_duration_nanos, 20_000_000);
        assert_eq!(stats.object_count, 6);
        assert_eq!(stats.scene_count, 2);
        assert_eq!(stats.samples().collect::<Vec<_>>(), [10, 20, 25, 50]);

        stats.record_frame(Duration::from_millis(10), 7, 3);
        assert_eq!(stats.fps, 100);
        assert_eq!(stats.avg_fps, 48);
        assert_eq!(stats.low_1pct_fps, 20);
        assert_eq!(stats.samples().collect::<Vec<_>>(), [20, 25, 50, 100]);
        assert_eq!(stats.sample_count(), 4);

        stats.reset_scene();
        assert_eq!(stats.frame, 5);
        assert_eq!(stats.scene_frame, 0);
        stats.reset();
        assert_eq!(stats.frame, 0);
        assert_eq!(stats.sample_count(), 0);
        assert_eq!(stats.capacity(), 4);
    }

    #[test]
    fn render_stats_default_window_and_one_percent_mean() {
        let mut stats = RenderStats::new();
        assert_eq!(stats.capacity(), DEFAULT_FPS_WINDOW);

        for _ in 0..99 {
            stats.record_frame(Duration::from_millis(10), 1, 1);
        }
        stats.record_frame(Duration::from_millis(100), 2, 1);

        assert_eq!(stats.frame, 100);
        assert_eq!(stats.fps, 10);
        assert_eq!(stats.avg_fps, 99);
        assert_eq!(stats.low_1pct_fps, 10);

        stats.reset();
        for _ in 0..198 {
            stats.record_frame(Duration::from_millis(10), 1, 1);
        }
        stats.record_frame(Duration::from_millis(100), 1, 1);
        stats.record_frame(Duration::from_millis(50), 1, 1);
        assert_eq!(stats.low_1pct_fps, 15);

        stats.record_frame(Duration::ZERO, 1, 1);
        assert_eq!(stats.fps, 0);
    }

    #[test]
    fn singleton_scopes_match_upstream_predicates() {
        assert!(should_run(SingletonScope::Game, 0, 99, true));
        assert!(!should_run(SingletonScope::Game, 1, 0, false));
        assert!(SingletonScope::Scene.should_run(99, 0, true));
        assert!(!SingletonScope::Scene.should_run(0, 1, false));
        assert!(should_run(SingletonScope::Object, 99, 99, false));
        assert!(!should_run(SingletonScope::Object, 0, 0, true));
    }

    #[test]
    fn game_info_default_validates_and_rejects_bad_dimensions() {
        let info = GameInfo::default();
        assert_eq!(info.validate(), Ok(()));
        assert_eq!(info.version, "1.0.0");
        assert_eq!(info.emulator_dim, None);
        assert_eq!(info.min_terminal_dim, (1, 1));
        assert!(!info.background.is_transparent());

        let mut invalid = GameInfo::new("");
        assert_eq!(invalid.validate(), Err(GameInfoError::EmptyName));
        invalid.name = String::from("Game");
        invalid.id.clear();
        assert_eq!(invalid.validate(), Err(GameInfoError::EmptyId));
        invalid.id = String::from("game");
        invalid.version.clear();
        assert_eq!(invalid.validate(), Err(GameInfoError::EmptyVersion));

        invalid.version = String::from("2.0.0");
        invalid.emulator_dim = Some((0, 10));
        assert_eq!(invalid.validate(), Err(GameInfoError::ZeroEmulatorWidth));
        invalid.emulator_dim = Some((10, 0));
        assert_eq!(invalid.validate(), Err(GameInfoError::ZeroEmulatorHeight));

        invalid.emulator_dim = None;
        invalid.min_terminal_dim = (0, 1);
        assert_eq!(invalid.validate(), Err(GameInfoError::ZeroMinTerminalWidth));
        invalid.min_terminal_dim = (1, 0);
        assert_eq!(
            invalid.validate(),
            Err(GameInfoError::ZeroMinTerminalHeight)
        );
    }

    #[test]
    fn duration_helpers_saturate_and_compute_fps() {
        assert_eq!(duration_millis(Duration::from_millis(1_500)), 1_500);
        assert_eq!(duration_nanos(Duration::from_nanos(42)), 42);
        assert_eq!(fps_from_duration(Duration::from_millis(10)), 100);
        assert_eq!(fps_from_duration(Duration::ZERO), 0);
        assert_eq!(fps_from_duration(Duration::from_secs(2)), 0);
    }
}
