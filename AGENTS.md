# AGENTS.md - Brush Development Guidelines

## Build/Test Commands
- **Build**: `cargo build` or `cargo build --release`
- **Test all**: `cargo test --all`
- **Single test**: `cargo test test_name` or `cargo test -p crate_name test_name`
- **Check**: `cargo check --locked --all-features --all-targets`
- **Lint**: `cargo clippy --all-targets --all-features -- -D warnings`
- **Format**: `cargo fmt --all`
- **Benchmarks**: `cargo bench`

## Code Style
- **Rust edition 2024**, toolchain 1.88.0+
- **No comments** unless specifically requested by user
- **Error handling**: Use `anyhow::Result`, `thiserror` for custom errors, `miette` for diagnostics
- **Unsafe**: Always document safety with `// SAFETY:` comments
- **Imports**: Group std, external crates, then local modules. Use explicit imports over glob imports
- **Naming**: snake_case for functions/variables, PascalCase for types, SCREAMING_SNAKE_CASE for constants
- **Types**: Explicit type annotations when helpful, prefer `DType` for tensors
- **Clippy**: Extensive lints enabled (see Cargo.toml), warnings treated as errors

## Architecture
- **Workspace** with multiple crates under `crates/`
- **GPU computing** via Burn/CubeCL, WGPU shaders
- **Cross-platform** support (desktop, web, mobile)
- **No dynamic allocations** in hot paths where possible