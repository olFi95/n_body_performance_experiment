# N-Body Simulation

N-Body sim with different implementations. Wanted to see which one's fastest.

## Implementations

- CPU single-threaded (slow)
- CPU multi-threaded (Rayon)
- SIMD single-threaded
- SIMD multi-threaded (usually fastest for CPU)
- GPU (WGPU - fast for lots of bodies)

## Build

```bash
cargo build --release
```

Needs nightly Rust for SIMD stuff.

## Run Benchmarks

```bash
cargo bench
```

Results in `target/criterion/report/index.html`

## Usage

```rust
use nbody_sim::nbody::*;

let bodies = utils::generate_random_bodies(1000, 100.0);
let mut sim = CpuMultiThreaded::new(bodies, SimulationParams::default());
sim.step(100);
```

All implementations use the same `Simulation` trait, so just swap the type.

## Tests

```bash
cargo test
```

## Notes

- GPU is slower for small body counts (initialization overhead)
- SIMD variants need alignment, handled internally
- More bodies = GPU wins
- More CPU cores = Rayon scales better

That's it.
