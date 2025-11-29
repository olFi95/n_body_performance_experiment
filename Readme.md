# N-Body Simulation - Performance-Vergleich

Dieses Projekt demonstriert verschiedene Optimierungstechniken für N-Body-Simulationen in 2D.

## Implementierungen

Das Projekt enthält 5 verschiedene Backend-Implementierungen der N-Body-Simulation:

1. **CPU Single-threaded** - Naive, single-threaded CPU-Implementierung
2. **CPU Rayon Multi-threaded** - Parallelisiert mit Rayon
3. **SIMD Single-threaded** - Verwendet SIMD-Instruktionen (f32x8)
4. **SIMD Rayon Multi-threaded** - Kombiniert SIMD und Rayon
5. **GPU WGPU** - Läuft auf der GPU mittels WGPU

## Verwendung

```rust
use monte_carlo_root::nbody::*;

// Erstelle Körper
let bodies = utils::generate_random_bodies(100, 100.0);
let params = SimulationParams::default();

// Wähle ein Backend
let mut sim = CpuSingleThreaded::new(bodies.clone(), params);
// let mut sim = CpuMultiThreaded::new(bodies.clone(), params);
// let mut sim = SimdSingleThreaded::new(bodies.clone(), params);
// let mut sim = SimdMultiThreaded::new(bodies.clone(), params);
// let mut sim = GpuSimulator::new(bodies.clone(), params).await; // async!

// Führe Simulation durch
sim.step(10);

// Hole Ergebnisse
let result = sim.get_bodies();
```

## Tests ausführen

```bash
cargo test
```

Die Tests vergleichen alle Implementierungen miteinander und stellen sicher, dass sie die gleichen Ergebnisse produzieren (innerhalb von Floating-Point-Toleranzen).

## Benchmarks ausführen

```bash
cargo bench
```

Die Benchmarks vergleichen die Performance aller Implementierungen mit verschiedenen Anzahlen von Körpern und Simulationsschritten.

## Ergebnisse

Die Benchmarks zeigen typischerweise:
- **CPU Single** ist die Baseline
- **CPU Rayon** skaliert gut mit mehreren Cores
- **SIMD Single** ist deutlich schneller für die Vektoroperationen
- **SIMD Rayon** kombiniert beide Vorteile
- **GPU** ist am effizientesten bei großen Datenmengen (>1000 Körper)

## Voraussetzungen

- Rust Nightly (für portable_simd Feature)
- WGPU-kompatible GPU für GPU-Backend

## Simulationsparameter

- `dt`: Zeitschritt (Standard: 0.016 für ~60 FPS)
- `epsilon`: Softening-Parameter zur Vermeidung von Singularitäten (Standard: 1e-6)
- `g_constant`: Gravitationskonstante (Standard: 1.0)

