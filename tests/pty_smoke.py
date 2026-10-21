#!/usr/bin/env python3
"""Unix-only end-to-end terminal checks; Python stdlib, no interactive user needed.

Run after cargo build: python3 rust/tests/pty_smoke.py rust/target/debug/pong
Optionally build --example terminal_lifecycle and pass its executable as the
second argument to also check normal, early-error, and panic cleanup.
"""
# SPDX-License-Identifier: Apache-2.0
import errno
import fcntl
import os
import pathlib
import pty
import re
import select
import signal
import struct
import subprocess
import sys
import termios
import time

CSI = re.compile(rb"\x1b\[[0-?]*[ -/]*[@-~]")


def check(executable, quit_key, exercise_resize=False):
    master, slave = pty.openpty()
    # A session leader exiting revokes the slave on macOS. Keep a tiny wrapper
    # alive to inspect attributes after Pong exits, before terminal hangup.
    wrapper = (
        "import subprocess,sys,termios; "
        "before=termios.tcgetattr(0); "
        "result=subprocess.run([sys.argv[1]]); "
        "assert termios.tcgetattr(0)==before, 'terminal attributes not restored'; "
        "print('PTY_ATTRIBUTES_RESTORED', flush=True); "
        "sys.exit(result.returncode)"
    )
    output = bytearray()
    process = None

    def resize(width, height):
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", height, width, 0, 0))
        if process is not None:
            os.killpg(process.pid, signal.SIGWINCH)

    def controlling_terminal():
        os.setsid()
        fcntl.ioctl(0, termios.TIOCSCTTY, 0)

    def until(marker, since=0, raw=False):
        deadline = time.monotonic() + 8
        while True:
            data = bytes(output[since:])
            if marker in (data if raw else CSI.sub(b"", data)):
                return len(output)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(f"timeout waiting for {marker!r}; tail={bytes(output[-500:])!r}")
            readable, _, _ = select.select([master], [], [], remaining)
            if not readable:
                continue
            try:
                data = os.read(master, 65536)
            except OSError as error:
                if error.errno == errno.EIO:
                    data = b""
                else:
                    raise
            if not data:
                raise AssertionError(f"terminal closed before {marker!r}; output={bytes(output)!r}")
            output.extend(data)

    try:
        resize(80, 24)
        process = subprocess.Popen(
            [sys.executable, "-c", wrapper, str(executable)], stdin=slave, stdout=slave, stderr=slave,
            preexec_fn=controlling_terminal,
            env={**os.environ, "TERM": "xterm-256color", "COLORTERM": "truecolor"},
        )
        until(b"COSPLAY / RUST PONG")
        assert b"\x1b[?1049h" in output, "did not enter alternate screen"
        during = termios.tcgetattr(slave)
        assert not during[3] & termios.ICANON, "did not enter raw mode"
        if exercise_resize:
            offset = len(output)
            os.write(master, b"ws ")
            until(b"PAUSED", offset)
            offset = len(output)
            resize(30, 10)
            until(b"Resize terminal", offset)
            offset = len(output)
            resize(60, 20)
            until(b"COSPLAY / RUST PONG", offset)
            os.write(master, b"r")
        offset = len(output)
        os.write(master, quit_key)
        until(b"\x1b[?1049l", offset, raw=True)
        until(b"PTY_ATTRIBUTES_RESTORED", offset)
        assert process.wait(timeout=8) == 0, f"nonzero exit: {process.returncode}"
        for sequence in (b"\x1b[?25h", b"\x1b[?7h"):
            assert sequence in output[offset:], f"missing cleanup {sequence!r}"
        print(f"PTY smoke passed: quit={quit_key!r}, resize={exercise_resize}")
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=8)
        os.close(master)
        os.close(slave)


def check_lifecycle(executable, mode):
    master, slave = pty.openpty()
    # Keep the session leader alive until attributes have been compared. On
    # macOS, inspecting the slave after that leader exits can fail with EIO.
    # A zero-time readiness check lets the kernel settle its transient PENDIN
    # flag before each snapshot. No attributes are rewritten or masked, and no
    # input is consumed: the full termios structures must still match exactly.
    wrapper = (
        "import os,select,subprocess,sys,termios; "
        "select.select([0], [], [], 0); before=termios.tcgetattr(0); "
        "result=subprocess.run(sys.argv[1:]); "
        "select.select([0], [], [], 0); after=termios.tcgetattr(0); "
        "assert after==before, f'terminal attributes not restored: {before!r} -> {after!r}'; "
        "print('PTY_ATTRIBUTES_RESTORED', flush=True); "
        "os._exit(result.returncode)"
    )
    output = bytearray()
    process = None

    def controlling_terminal():
        os.setsid()
        fcntl.ioctl(0, termios.TIOCSCTTY, 0)

    try:
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 80, 0, 0))
        process = subprocess.Popen(
            [sys.executable, "-c", wrapper, str(executable), mode],
            stdin=slave, stdout=slave, stderr=slave,
            preexec_fn=controlling_terminal,
            env={**os.environ, "TERM": "xterm-256color", "COLORTERM": "truecolor"},
        )
        deadline = time.monotonic() + 8
        # Read through the wrapper's final marker, not until the PTY stops
        # being readable: EOF itself stays readable indefinitely on macOS.
        while b"PTY_ATTRIBUTES_RESTORED" not in output.splitlines():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(f"lifecycle {mode}: timed out; output={bytes(output)!r}")
            readable, _, _ = select.select([master], [], [], remaining)
            if not readable:
                continue
            try:
                data = os.read(master, 65536)
            except OSError as error:
                if error.errno != errno.EIO:
                    raise
                data = b""
            if not data:
                raise AssertionError(
                    f"lifecycle {mode}: closed before attribute check; output={bytes(output)!r}"
                )
            output.extend(data)

        expected_status, diagnostic = {
            "normal": (0, b"LIFECYCLE_NORMAL"),
            "error": (1, b"LIFECYCLE_ERROR: intentional lifecycle error"),
            "panic": (101, b"LIFECYCLE_PANIC:"),
        }[mode]
        status = process.wait(timeout=8)
        assert status == expected_status, (
            f"lifecycle {mode}: exit {status}, expected {expected_status}; output={bytes(output)!r}"
        )
        enter, leave = b"\x1b[?1049h", b"\x1b[?1049l"
        assert output.count(enter) == 1, f"lifecycle {mode}: missing/duplicate alternate entry"
        assert output.count(leave) == 1, f"lifecycle {mode}: missing/duplicate alternate cleanup"
        leave_offset = output.index(leave)
        for setup, cleanup in (
            (enter, leave),
            (b"\x1b[?25l", b"\x1b[?25h"),
            (b"\x1b[?7l", b"\x1b[?7h"),
        ):
            assert setup in output and cleanup in output, (
                f"lifecycle {mode}: missing mode change {setup!r}/{cleanup!r}"
            )
            assert output.index(setup) < output.index(cleanup) <= leave_offset, (
                f"lifecycle {mode}: incorrect cleanup order for {cleanup!r}"
            )
        assert output.count(b"\x1b[0m") >= 2, f"lifecycle {mode}: missing color reset"
        assert output.rindex(b"\x1b[0m") < leave_offset, f"lifecycle {mode}: late color reset"
        assert output.count(diagnostic) == 1, f"lifecycle {mode}: missing/duplicate diagnostic"
        assert leave_offset + len(leave) <= output.index(diagnostic), (
            f"lifecycle {mode}: diagnostic printed before alternate-screen restoration"
        )
        assert output.index(diagnostic) < output.index(b"PTY_ATTRIBUTES_RESTORED")
        if mode == "panic":
            assert b"intentional lifecycle panic" in output, "missing panic payload"
        print(f"PTY lifecycle passed: mode={mode}, exit={status}, exact termios restored")
    finally:
        if process is not None and process.poll() is None:
            # A session leader can disappear or become unsignalable while its
            # PTY is being revoked. Never replace the test's original failure
            # with a cleanup PermissionError/TimeoutExpired.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as error:
                print(f"lifecycle cleanup: killpg: {error}", file=sys.stderr)
                try:
                    process.kill()
                except OSError as error:
                    print(f"lifecycle cleanup: kill: {error}", file=sys.stderr)
            try:
                process.wait(timeout=8)
            except (OSError, subprocess.TimeoutExpired) as error:
                print(f"lifecycle cleanup: wait: {error}", file=sys.stderr)
        os.close(master)
        os.close(slave)


if __name__ == "__main__":
    executable = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "rust/target/debug/pong").resolve()
    check(executable, b"q", exercise_resize=True)
    check(executable, b"\x03")  # Ctrl-C is a key in raw mode, not SIGINT.
    check(executable, b"\x1b")
    if len(sys.argv) > 2:
        lifecycle = pathlib.Path(sys.argv[2]).resolve()
        for mode in ("normal", "error", "panic"):
            check_lifecycle(lifecycle, mode)
