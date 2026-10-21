// SPDX-License-Identifier: Apache-2.0
use std::process::{Command, Stdio};

fn pong() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_pong"));
    command.stdin(Stdio::null());
    command
}

#[test]
fn headless_snapshot_is_reproducible_and_matches_golden_checksum() {
    let run = || {
        pong()
            .args(["--headless", "--frames", "120", "--size", "60x20"])
            .output()
            .unwrap()
    };
    let a = run();
    let b = run();
    assert!(a.status.success(), "{}", String::from_utf8_lossy(&a.stderr));
    assert_eq!(a.stdout, b.stdout);
    let text = String::from_utf8(a.stdout).unwrap();
    assert_eq!(
        text.lines().next().unwrap(),
        "headless: frames=120 size=60x20 snapshot=05b6d35f0db2f3cd"
    );
    assert_eq!(text.lines().count(), 21);
    assert!(text.contains("COSPLAY / RUST PONG"));
    assert!(!text.contains('\x1b'));
}

#[test]
fn interactive_rejects_pipes_without_emitting_terminal_controls() {
    let result = pong().output().unwrap();
    assert!(!result.status.success());
    assert!(result.stdout.is_empty());
    assert!(String::from_utf8_lossy(&result.stderr).contains("TTY"));
}

#[test]
fn help_and_invalid_arguments_do_not_require_a_terminal() {
    let help = pong().arg("--help").output().unwrap();
    assert!(help.status.success());
    assert!(String::from_utf8_lossy(&help.stdout).contains("Usage:"));
    let invalid = pong()
        .args(["--headless", "--frames", "0"])
        .output()
        .unwrap();
    assert!(!invalid.status.success());
    assert!(String::from_utf8_lossy(&invalid.stderr).contains("frames must be"));
}
