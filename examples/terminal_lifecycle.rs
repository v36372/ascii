// SPDX-License-Identifier: Apache-2.0
//! Exercise explicit restoration, early-error Drop, and panic-hook cleanup.
//! Run in a TTY: cargo run --example terminal_lifecycle -- normal|error|panic

use cosplay::terminal::{install_panic_hook, Terminal};
use std::{io, process::ExitCode};

fn run(mode: &str) -> io::Result<()> {
    if !matches!(mode, "normal" | "error" | "panic") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "expected normal, error, or panic",
        ));
    }
    let mut terminal = Terminal::new()?;
    match mode {
        "normal" => terminal.restore(),
        // Return before doing any rendering: the guard must restore on error.
        "error" => Err(io::Error::other("intentional lifecycle error")),
        // The hook must restore *before* diagnostics, not just during unwinding.
        "panic" => panic!("intentional lifecycle panic"),
        _ => unreachable!(),
    }
}

fn main() -> ExitCode {
    // Demonstrate that the terminal hook chains an existing diagnostic hook.
    std::panic::set_hook(Box::new(|info| eprintln!("LIFECYCLE_PANIC: {info}")));
    install_panic_hook();
    let mode = std::env::args().nth(1).unwrap_or_else(|| "normal".into());
    match run(&mode) {
        Ok(()) => {
            println!("LIFECYCLE_NORMAL");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("LIFECYCLE_ERROR: {error}");
            ExitCode::FAILURE
        }
    }
}
