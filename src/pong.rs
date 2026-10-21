// SPDX-License-Identifier: Apache-2.0
//! Small native Pong game demonstrating the scene runtime and canvas.
//! This is a new terminal-first demo, not a feature-identical Scala Pong port.

use crate::engine::{FrameContext, FrameInfo, Scene, SceneObject};
use crate::graphics::{Canvas, Color, Pixel, Rect};
use crate::terminal::Key;
use std::time::Duration;

pub const MIN_WIDTH: u16 = 40;
pub const MIN_HEIGHT: u16 = 16;
const PADDLE_HEIGHT: f64 = 4.0;
const WIN_SCORE: u32 = 7;
const FG: Color = Color::rgb(218, 232, 247);
const ACCENT: Color = Color::rgb(72, 220, 190);
const BG: Color = Color::rgb(12, 18, 28);

pub fn scene(autoplay: bool) -> Scene {
    let mut scene = Scene::new("pong", Pixel::new(' ', FG, Some(BG)));
    scene.add_object(Pong::new(autoplay));
    scene
}

pub struct Pong {
    width: u16,
    height: u16,
    player_y: f64,
    computer_y: f64,
    ball_x: f64,
    ball_y: f64,
    vx: f64,
    vy: f64,
    player_score: u32,
    computer_score: u32,
    paused: bool,
    serve_remaining: f64,
    autoplay: bool,
}

impl Pong {
    pub fn new(autoplay: bool) -> Self {
        Self {
            width: 0,
            height: 0,
            player_y: 0.0,
            computer_y: 0.0,
            ball_x: 0.0,
            ball_y: 0.0,
            vx: 24.0,
            vy: 9.0,
            player_score: 0,
            computer_score: 0,
            paused: false,
            serve_remaining: 0.7,
            autoplay,
        }
    }

    pub fn score(&self) -> (u32, u32) {
        (self.player_score, self.computer_score)
    }
    pub fn is_paused(&self) -> bool {
        self.paused
    }
    pub fn is_finished(&self) -> bool {
        self.player_score >= WIN_SCORE || self.computer_score >= WIN_SCORE
    }

    fn fits(&self) -> bool {
        self.width >= MIN_WIDTH && self.height >= MIN_HEIGHT
    }
    fn paddle_max(&self) -> f64 {
        f64::from(self.height) - 3.0 - PADDLE_HEIGHT
    }
    fn center_paddles(&mut self) {
        self.player_y = (f64::from(self.height) - PADDLE_HEIGHT) / 2.0;
        self.computer_y = self.player_y;
    }

    fn serve(&mut self, direction: f64) {
        self.ball_x = f64::from(self.width) / 2.0;
        self.ball_y = f64::from(self.height) / 2.0;
        self.vx = 24.0 * direction;
        self.vy = if (self.player_score + self.computer_score).is_multiple_of(2) {
            9.0
        } else {
            -9.0
        };
        self.serve_remaining = 0.7;
    }

    fn reset(&mut self) {
        self.player_score = 0;
        self.computer_score = 0;
        self.paused = false;
        self.center_paddles();
        self.serve(1.0);
    }

    /// Advance using injected time and key presses. Returns true on a quit key.
    /// Long stalls are capped at 250ms, then integrated in <= 1/120s substeps.
    /// Too-small viewports suspend gameplay but continue accepting quit/restart.
    pub fn step(&mut self, delta: Duration, keys: &[Key], width: u16, height: u16) -> bool {
        if keys
            .iter()
            .any(|key| matches!(key, Key::Escape | Key::Char('q' | 'Q')))
        {
            return true;
        }
        if (width, height) != (self.width, self.height) {
            self.width = width;
            self.height = height;
            self.center_paddles();
            self.serve(if self.vx < 0.0 { -1.0 } else { 1.0 });
        }
        for key in keys {
            match key {
                Key::Char('r' | 'R') => self.reset(),
                Key::Space | Key::Char('p' | 'P') => {
                    if self.is_finished() {
                        self.reset();
                    } else {
                        self.paused = !self.paused;
                    }
                }
                Key::Up | Key::Char('w' | 'W')
                    if self.fits() && !self.paused && !self.is_finished() =>
                {
                    self.player_y -= 2.0
                }
                Key::Down | Key::Char('s' | 'S')
                    if self.fits() && !self.paused && !self.is_finished() =>
                {
                    self.player_y += 2.0
                }
                _ => {}
            }
        }
        if !self.fits() {
            return false;
        }
        self.player_y = self.player_y.clamp(3.0, self.paddle_max());
        if self.paused || self.is_finished() {
            return false;
        }
        let mut remaining = delta.as_secs_f64().min(0.25);
        while remaining > 0.0 {
            let dt = remaining.min(1.0 / 120.0);
            remaining -= dt;
            let target = self.ball_y - PADDLE_HEIGHT / 2.0;
            self.computer_y += (target - self.computer_y).clamp(-13.0 * dt, 13.0 * dt);
            self.computer_y = self.computer_y.clamp(3.0, self.paddle_max());
            if self.autoplay {
                self.player_y += (target - self.player_y).clamp(-18.0 * dt, 18.0 * dt);
                self.player_y = self.player_y.clamp(3.0, self.paddle_max());
            }
            if self.serve_remaining > 0.0 {
                self.serve_remaining = (self.serve_remaining - dt).max(0.0);
                continue;
            }
            self.move_ball(dt);
            if self.ball_x < 1.0 {
                self.computer_score += 1;
                self.serve(1.0);
                break;
            }
            if self.ball_x > f64::from(self.width) - 2.0 {
                self.player_score += 1;
                self.serve(-1.0);
                break;
            }
        }
        false
    }

    fn vertical_motion(&self, dt: f64) -> (f64, f64) {
        let mut y = self.ball_y + self.vy * dt;
        let mut vy = self.vy;
        let bottom = f64::from(self.height) - 4.0;
        // Speed and substep bounds guarantee at most one wall impact here.
        if y < 3.0 {
            y = 6.0 - y;
            vy = vy.abs();
        }
        if y > bottom {
            y = 2.0 * bottom - y;
            vy = -vy.abs();
        }
        (y, vy)
    }

    fn move_ball(&mut self, dt: f64) {
        let (plane, paddle, direction) = if self.vx < 0.0 {
            (4.0, self.player_y, 1.0)
        } else {
            (f64::from(self.width) - 5.0, self.computer_y, -1.0)
        };
        let impact_time = (plane - self.ball_x) / self.vx;
        let mut remaining = dt;
        if (0.0..=dt).contains(&impact_time) {
            let (y, vy) = self.vertical_motion(impact_time);
            // Evaluate at the swept plane crossing, not the substep's final y.
            if y >= paddle - 0.5 && y < paddle + PADDLE_HEIGHT + 0.5 {
                self.ball_x = plane;
                self.ball_y = y;
                self.vy = vy;
                self.bounce(paddle, direction);
                remaining -= impact_time;
            }
        }
        (self.ball_y, self.vy) = self.vertical_motion(remaining);
        self.ball_x += self.vx * remaining;
    }

    fn bounce(&mut self, paddle_y: f64, direction: f64) {
        self.vx = (self.vx.abs() + 1.0).min(46.0) * direction;
        let offset = (self.ball_y - paddle_y - PADDLE_HEIGHT / 2.0) / (PADDLE_HEIGHT / 2.0);
        self.vy = (offset * 17.0).clamp(-17.0, 17.0);
        if self.vy.abs() < 3.0 {
            self.vy = if self.vy < 0.0 { -3.0 } else { 3.0 };
        }
    }

    fn centered(&self, canvas: &mut Canvas<'_>, y: i32, text: &str, color: Color) {
        let x = (i32::from(self.width) - text.len() as i32) / 2;
        canvas.draw_text(x, y, 10, text, color, Some(BG));
    }
}

impl SceneObject for Pong {
    fn rect(&self) -> Rect {
        Rect::new(0, 0, self.width, self.height)
    }
    fn update(&mut self, ctx: &mut FrameContext<'_>) {
        if self.step(
            ctx.frame.delta,
            &ctx.frame.keys,
            ctx.frame.width,
            ctx.frame.height,
        ) {
            ctx.quit();
        }
    }
    fn render(&self, canvas: &mut Canvas<'_>, _: &FrameInfo) {
        if !self.fits() {
            // Start at the left edge so even a tiny viewport shows a useful prefix.
            canvas.draw_text(0, 0, 0, "Resize terminal to at least 40x16", FG, Some(BG));
            canvas.draw_text(0, 1, 0, "Q / Esc: quit", FG, Some(BG));
            return;
        }
        self.centered(canvas, 0, "COSPLAY / RUST PONG", ACCENT);
        self.centered(
            canvas,
            1,
            &format!(
                "YOU {:02}    CPU {:02}   first to {WIN_SCORE}",
                self.player_score, self.computer_score
            ),
            FG,
        );
        let border = Pixel::new('-', Color::rgb(70, 90, 115), Some(BG));
        canvas.draw_rect(Rect::new(0, 2, self.width, self.height - 4), 0, border);
        for y in (3..i32::from(self.height) - 3).step_by(2) {
            canvas.draw_pixel(
                i32::from(self.width) / 2,
                y,
                0,
                Pixel::new(':', Color::rgb(45, 60, 80), Some(BG)),
            );
        }
        for (x, y) in [
            (3, self.player_y),
            (i32::from(self.width) - 4, self.computer_y),
        ] {
            canvas.fill_rect(
                Rect::new(x, y.round() as i32, 1, PADDLE_HEIGHT as u16),
                1,
                Pixel::new('|', ACCENT, Some(ACCENT)),
            );
        }
        canvas.draw_pixel(
            self.ball_x.round() as i32,
            self.ball_y.round() as i32,
            2,
            Pixel::new('O', Color::WHITE, Some(BG)),
        );
        if self.is_finished() {
            self.centered(
                canvas,
                i32::from(self.height) / 2,
                if self.player_score >= WIN_SCORE {
                    "YOU WIN! Space: play again"
                } else {
                    "CPU WINS! Space: play again"
                },
                ACCENT,
            );
        } else if self.paused {
            self.centered(
                canvas,
                i32::from(self.height) / 2,
                "PAUSED - Space to resume",
                ACCENT,
            );
        }
        self.centered(
            canvas,
            i32::from(self.height) - 2,
            "W/S or arrows: move | Space: pause",
            FG,
        );
        self.centered(
            canvas,
            i32::from(self.height) - 1,
            "R: restart | Q/Esc: quit",
            FG,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Engine;

    fn ready() -> Pong {
        let mut p = Pong::new(false);
        p.step(Duration::ZERO, &[], 80, 24);
        p.serve_remaining = 0.0;
        p
    }
    #[test]
    fn controls_pause_restart_and_quit() {
        let mut p = ready();
        let y = p.player_y;
        p.step(Duration::ZERO, &[Key::Up], 80, 24);
        assert_eq!(p.player_y, y - 2.0);
        let ball = (p.ball_x, p.ball_y);
        p.step(Duration::from_secs(1), &[Key::Space], 80, 24);
        assert!(p.is_paused());
        assert_eq!((p.ball_x, p.ball_y), ball);
        p.player_score = 3;
        p.step(Duration::ZERO, &[Key::Char('r')], 80, 24);
        assert_eq!(p.score(), (0, 0));
        assert!(!p.is_paused());
        assert!(p.step(Duration::ZERO, &[Key::Escape], 80, 24));
    }
    #[test]
    fn paddle_and_wall_bounces() {
        let mut p = ready();
        p.ball_x = 4.1;
        p.ball_y = p.player_y + 1.0;
        p.vx = -24.0;
        p.step(Duration::from_millis(20), &[], 80, 24);
        assert!(p.vx > 0.0);
        p.ball_y = 3.01;
        p.vy = -12.0;
        p.step(Duration::from_millis(20), &[], 80, 24);
        assert!(p.vy > 0.0);
        assert!(p.ball_y >= 3.0);
    }
    #[test]
    fn paddle_tip_uses_swept_impact_not_substep_endpoint() {
        let mut p = ready();
        p.player_y = 10.0;
        p.ball_x = 4.01;
        p.ball_y = 9.55;
        p.vx = -24.0;
        p.vy = -17.0;
        p.step(Duration::from_nanos(8_333_333), &[], 80, 24);
        assert!(
            p.vx > 0.0,
            "inside at crossing, outside at endpoint: must hit"
        );

        let mut p = ready();
        p.player_y = 10.0;
        p.ball_x = 4.01;
        p.ball_y = 9.45;
        p.vx = -24.0;
        p.vy = 17.0;
        p.step(Duration::from_nanos(8_333_333), &[], 80, 24);
        assert!(
            p.vx < 0.0,
            "outside at crossing, inside at endpoint: must miss"
        );
    }
    #[test]
    fn miss_scores_once_and_serves() {
        let mut p = ready();
        p.ball_x = 1.1;
        p.ball_y = 3.0;
        p.vx = -24.0;
        p.step(Duration::from_secs(10), &[], 80, 24);
        assert_eq!(p.score(), (0, 1));
        assert_eq!(p.ball_x, 40.0);
        assert!(p.serve_remaining > 0.0);
    }
    #[test]
    fn resize_suspends_and_restarts_rally_without_losing_score() {
        let mut p = ready();
        p.player_score = 2;
        p.step(Duration::from_secs(1), &[], 10, 4);
        let ball = (p.ball_x, p.ball_y);
        p.step(Duration::from_secs(1), &[], 10, 4);
        assert_eq!((p.ball_x, p.ball_y), ball);
        p.step(Duration::ZERO, &[], 40, 16);
        assert_eq!(p.score(), (2, 0));
        assert_eq!(p.ball_x, 20.0);
    }
    #[test]
    fn winning_freezes_until_restart() {
        let mut p = ready();
        p.player_score = WIN_SCORE - 1;
        p.ball_x = 78.0;
        p.vx = 24.0;
        p.step(Duration::from_millis(20), &[], 80, 24);
        assert!(p.is_finished());
        let ball = (p.ball_x, p.ball_y);
        p.step(Duration::from_secs(1), &[], 80, 24);
        assert_eq!((p.ball_x, p.ball_y), ball);
        p.step(Duration::ZERO, &[Key::Space], 80, 24);
        assert!(!p.is_finished());
    }
    #[test]
    fn deterministic_game_runs_headlessly_and_adapts_to_resize() {
        let run = || {
            let mut e = Engine::new(60, 20);
            e.add_scene(scene(true)).unwrap();
            e.start("pong").unwrap();
            for _ in 0..600 {
                e.tick(Duration::from_nanos(33_333_333), &[]).unwrap();
            }
            let snapshot = e.screen().snapshot();
            assert!(snapshot.contains("COSPLAY / RUST PONG"));
            e.resize(10, 4);
            e.tick(Duration::ZERO, &[]).unwrap();
            assert!(e.screen().snapshot().starts_with("Resize ter"));
            snapshot
        };
        assert_eq!(run(), run());
    }
}
