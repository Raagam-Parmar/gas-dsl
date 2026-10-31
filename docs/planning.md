# GAS-DSL: Architecture Specification & 2-Week Implementation Plan

This document details the system design, programming model, compiler pipeline, and 2-week execution roadmap for the OCaml Gather-Apply-Scatter (GAS) GPU DSL compiler targeting NVIDIA PTX assembly.

---

## 1. System Summary

* **Objective:** An OCaml-based DSL compiler that lowers Gather-Apply-Scatter graph analytics (SSSP, PageRank) to standalone `.ptx` kernels targeting NVIDIA `sm_75` (Turing) and above.
* **Execution Paradigm:** Edge-parallel Scatter kernel (1 thread per edge with atomic propagation) + optional Vertex-parallel Apply kernel for post-reduction normalization (e.g., PageRank damping factor).
* **Data Format:** Augmented Compressed Sparse Row (CSR) including a precomputed edge source array (`src[e]`, `dst[e]`, `weights[e]`, and `row_offsets`) to eliminate branch divergence and search overhead.
* **Memory Schema:** Structure-of-Arrays (SoA) passing typed 64-bit pointers (`.param .u64`) to global memory arrays.
* **Atomic Primitives:** Native integer atomics (`atom.global.min.u32/s32`, `atom.global.add.u32`) and native float add (`atom.global.add.f32`), with floating-point `min`/`max` synthesized via PTX Compare-and-Swap loops (`atom.global.cas.b32`).
* **EDSL Frontend:** Phantom-typed smart constructors (`'a expr`) over a linear algebraic IR AST.
* **Host Verification:** Decoupled C++ test harness using the CUDA Driver API (`cuModuleLoadData`, `cuLaunchKernel`) comparing GPU outputs against CPU golden models.

---

## 2. Compiler Pipeline & IR Structure

1. **Frontend (`lib/ast.ml`, `lib/dsl.ml`):**
   * Types: `F32`, `U32`, `I32`, `B8`.
   * Typed field references: `val field : string -> 'a typ -> 'a field`.
   * Combinators: `(+:)`, `(-:)`, `(*:)`, `(/:)`, `min_expr`.
   * Kernel builders: `scatter`, `apply`.
2. **Intermediate Representation (`lib/ir.ml`):**
   * Virtual register representation (`%r<N>`, `%f<N>`, `%p<N>`).
   * Explicit parameter loading, memory indexing (wide 32-to-64 bit conversions), and arithmetic ops.
   * Atomic operations including hardware atomics and loop-based CAS blocks.
3. **Backend (`lib/codegen.ml`):**
   * Emits PTX 7.0 for `sm_75`.
   * Direct emission of virtual register SSA code validated with `ptxas`.

---

## 3. Two-Week Execution Roadmap

### Week 1: OCaml Compiler & PTX Validation
* **Days 1–2:** Core EDSL AST, phantom-typed combinators, and field schema definitions (`lib/ast.ml`, `lib/dsl.ml`).
* **Days 3–4:** Linear IR, SSA virtual register generator, and lowering passes (`lib/ir.ml`).
* **Days 5–6:** PTX code generator targeting `sm_75`, with native atomics and float CAS loops (`lib/codegen.ml`).
* **Day 7:** Unit testing & automated validation against NVIDIA `ptxas` assembler.

### Week 2: Host Driver Harness & End-to-End Benchmarks
* **Days 8–9:** Minimal C++ CUDA Driver API runner (`cuda_driver_runner.cpp`) with graph memory allocation and kernel launch wrappers.
* **Day 10:** Augmented CSR graph file reader and synthetic graph generator (e.g. grid and random graphs).
* **Days 11–12:** SSSP implementation in DSL, PTX generation, and validation against Dijkstra/Bellman-Ford CPU reference.
* **Days 13–14:** PageRank implementation in DSL, 2-kernel iteration loop, and convergence validation against CPU reference.
