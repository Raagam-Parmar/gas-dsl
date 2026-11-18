# GAS-DSL

[![CI](https://github.com/Raagam-Parmar/gas-dsl/actions/workflows/ci.yml/badge.svg)](https://github.com/Raagam-Parmar/gas-dsl/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**GAS-DSL** is a statically-typed domain-specific language embedded in OCaml that compiles Gather-Apply-Scatter (GAS) graph processing algorithms directly into standalone NVIDIA PTX (Parallel Thread Execution) assembly.

Targeting static CSR graphs, the compiler generates decoupled GPU kernels, an edge-parallel **Scatter** kernel and an optional vertex-parallel **Apply** kernel, for algorithms like Single-Source Shortest Path (SSSP), PageRank, Breadth-First Search (BFS), and Weakly Connected Components (WCC).

**Documentation:** [https://raagam-parmar.github.io/gas-dsl](https://raagam-parmar.github.io/gas-dsl/gas-dsl/index.html)

---

## Key Features

- **Type-Safe EDSL:** Phantom-typed arithmetic expressions (`'a expr`) with type-safe vertex and edge attribute readers.
- **Two-Stage IR:** Translates from high-level `GraphIR` (GAS semantics) to `LowerIR` (virtual register SSA, pointer arithmetic, hardware atomics) down to `Ptx.Ast`.
- **Target Architecture:** Generates optimized PTX targeting NVIDIA Compute Capability `sm_75` (Turing architecture) and above.
- **Hardware-Level Reductions:** Emits native integer atomic reductions (`atom.global.min/max/add`) and automated Compare-And-Swap retry loops (`atom.global.cas.b32`) for 32-bit floating-point min/max reductions.
- **Zero Host Runtime Dependency:** Generates pure PTX assembly without requiring proprietary CUDA runtime libraries at compile time.

---

## Installation via `opam pin`

You can install and use `gas-dsl` directly as an opam package without publishing it to the public opam repository:

### 1. Pin from GitHub (Recommended)

```bash
opam pin add gas-dsl git+https://github.com/Raagam-Parmar/gas-dsl.git
```

To update to the latest commit later:
```bash
opam update
opam upgrade gas-dsl
```

### 2. Pin from a Local Clone

If you are developing locally:
```bash
git clone https://github.com/Raagam-Parmar/gas-dsl.git
cd gas-dsl
opam pin add gas-dsl .
```

To unpin:
```bash
opam pin remove gas-dsl
```

---

## Quick Example: PageRank in EDSL

```ocaml
open Gas_dsl.Edsl

let pagerank =
  let rank       = Vertex.f32 "rank" in
  let rank_accum = Vertex.f32 "rank_accum" in
  let out_deg    = Vertex.f32 "out_deg" in
  algorithm ~name:"pagerank"
    ~scatter:(scatter ~target:rank_accum ~reduce:Sum (fun ~src ~dst:_ ~edge:_ ->
      src.get rank /: src.get out_deg))
    ~apply:(apply ~dest:rank (fun ~self ->
      float 0.15 +: float 0.85 *: self.get rank_accum))
    ()

(* Compile to PTX assembly *)
let ptx_string =
  pagerank
  |> Gas_dsl.GraphIR.Translate.translate
  |> Gas_dsl.LowerIR.Translate.translate ~target:"sm_75"
  |> Gas_dsl.Ptx.Printer.to_string
```

---

## Development & Testing

Building the project and running the test suite (requires OCaml $\ge$ 5.0, Dune $\ge$ 3.21, and `ptxas` for GPU assembly validation):

```bash
# Install dependencies
opam install . --deps-only --with-test

# Build everything and run test suite
dune build @all @runtest
```

---

## Authors

- **Raagam Parmar** ([@Raagam-Parmar](https://github.com/Raagam-Parmar))
- **M. Sreenath Karthick** ([@SreenathKarthick11](https://github.com/SreenathKarthick11))
- **M. Murali Karthick** ([@Murali47k](https://github.com/Murali47k))

Licensed under the [MIT License](LICENSE).
