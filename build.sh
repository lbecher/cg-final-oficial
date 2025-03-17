#!/bin/bash
cargo build --target x86_64-pc-windows-gnu --release
cargo build --target x86_64-unknown-linux-gnu --release
mkdir -p bin
cp target/x86_64-pc-windows-gnu/release/cg-final-oficial.exe bin/
