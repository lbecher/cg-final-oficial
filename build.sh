#!/bin/bash
sudo apt install -y curl build-essential gcc-mingw-w64-x86-64 binutils-mingw-w64-x86-64 pkg-config libssl-dev librust-glib-sys-dev librust-gdk-sys-dev 
curl https://sh.rustup.rs -sSf | sh
cargo build --target x86_64-pc-windows-gnu --release
cargo build --target x86_64-unknown-linux-gnu --release
mkdir -p bin
cp target/x86_64-pc-windows-gnu/release/cg-final-oficial.exe bin/
