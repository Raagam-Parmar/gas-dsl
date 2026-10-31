# AGENTS.md — Developer & AI Agent Guidelines for GAS-DSL

This file serves as the definitive reference for AI agents and developers working on **GAS-DSL**—an OCaml-based domain-specific language compiler that translates Gather-Apply-Scatter (GAS) graph processing algorithms directly into standalone NVIDIA PTX assembly for the NVIDIA A30 GPU cluster.

---

## 1. Project Mission & Constraints

* **Core Goal:** Translate high-level GAS algorithms (specifically **Single-Source Shortest Path (SSSP)** and **PageRank**) into standalone, high-performance `.ptx` kernels.
* **Timeline:** 2-week development window across a 3-person engineering team.
* **Target Hardware:** **NVIDIA A30 GPU** (Ampere architecture, **Compute Capability `sm_80`**, 64-bit address size, 24 GB HBM2).
* **Graph Format:** Static **Augmented Compressed Sparse Row (CSR)** format with precomputed `src_indices[e]` to avoid in-kernel binary searches.
* **Decoupled Execution:** The OCaml compiler emits standalone `.ptx` files. A lightweight C++ host harness loads and executes kernels via the CUDA Driver API (`cuModuleLoadData`, `cuLaunchKernel`). No complex host runtime engine or dynamic GPU memory management.

---

## 2. Compiler Pipeline & Module Architecture

The compiler implements a strict **2-level Intermediate Representation (IR)** pipeline:

```
OCaml EDSL (edsl/)
      │
      ▼
Graph IR (graphIR/)
      │
      ▼
Lower IR (lowerIR/)
      │
      ▼
PTX Assembly AST (ptx/)
      │
      ▼
Standalone .ptx Text File
```

Dune is configured with `(include_subdirs qualified)` in `lib/dune`. Modules are exposed under the `Gas_dsl` umbrella:

| Directory | Module Path | Purpose | Key Files |
| :--- | :--- | :--- | :--- |
| `lib/graphIR/` | `Gas_dsl.GraphIR` | High-level graph semantics (Gather expressions, Scatter reductions, Apply updates, SoA property schemas). Hardware-agnostic. | `types.ml`<br>`ast.ml` |
| `lib/lowerIR/` | `Gas_dsl.LowerIR` | Machine-level linear SSA instruction sequence. Virtual registers, explicit 64-bit address scaling, memory spaces, hardware atomics, CAS loops. | `types.ml`<br>`ast.ml` |
| `lib/ptx/` | `Gas_dsl.Ptx` | Concrete NVIDIA PTX ISA AST targeting `sm_80`. Header directives, register banks (`%r`, `%rd`, `%f`, `%p`), cache modifiers, instruction emission. | `types.ml`<br>`ast.ml` |
| `lib/edsl/` | `Gas_dsl.Edsl` | Frontend user-facing combinators and phantom-typed smart constructors (`'a expr`, `gather`, `scatter`, `apply`). | `edsl.ml` |

---

## 3. Key Technical Invariants & Rules

### 3.1 Target Hardware: NVIDIA A30 (`sm_80`)
1. **PTX Directives:** Every emitted PTX kernel must begin with:
   ```ptx
   .version 7.0
   .target sm_80
   .address_size 64
   ```
2. **64-bit Address Scaling:** Memory pointers are 64-bit (`.u64`). Because vertex and edge counts fit in 32-bit integers (`.u32`), index offsets must be scaled using wide multiplication (`mul.wide.u32`) before adding to the 64-bit base pointer (`add.u64`).

### 3.2 Memory Layout: Structure of Arrays (SoA)
* No Array-of-Structures (AoS) structs in GPU global memory.
* Every vertex/edge property (`dist`, `rank`, `out_deg`, `weights`) resides in an independent global memory buffer passed as a distinct `.param .u64` pointer parameter to the kernel.
* This layout guarantees coalesced memory accesses across consecutive GPU threads.

### 3.3 Atomic Reductions & Float CAS Loops
* **PageRank (Float Sum):** Supported via native hardware `atom.global.add.f32`.
* **SSSP Integer Mode (UInt32 Min):** Supported via native hardware `atom.global.min.u32`.
* **SSSP Float Mode (Float32 Min):** Hardware PTX does not support `atom.min.f32`. It **must** lower to a Compare-and-Swap retry loop using `atom.global.cas.b32`:
  ```ptx
  $CAS_LOOP:
  ld.global.f32 %cur_val, [%addr];
  setp.ge.f32   %p_skip, %cur_val, %new_val;
  @!%p_skip bra $CAS_END;
  atom.global.cas.b32 %old_bits, [%addr], %cur_bits, %new_bits;
  setp.eq.b32   %p_success, %old_bits, %cur_bits;
  @!%p_success bra $CAS_LOOP;
  $CAS_END:
  ```

### 3.4 Predicate Registers (`Pred`)
* Predicates (`%p<N>`) are 1-bit boolean condition flags.
* Used exclusively for:
  * Thread boundary guards: `setp.ge.u32 %p0, %tid, %num_items; @%p0 bra $EXIT;`
  * CAS loop convergence tests: `setp.eq.b32 %p1, %old_bits, %cur_bits; @!%p1 bra $CAS_LOOP;`
  * Active frontier filtering.

---

## 4. Execution Model & Graph Representation

1. **Augmented CSR Graph Format:**
   * `row_offsets` (size $|V|+1$): Standard CSR vertex pointers.
   * `src_indices` (size $|E|$): Precomputed source vertex for each edge (COO-style).
   * `dst_indices` (size $|E|$): Target vertex for each edge (standard CSR column indices).
   * `weights` (size $|E|$): Edge weights (optional).
2. **Two-Stage Kernel Iteration:**
   * **Scatter Kernel (Edge-Parallel):** 1 thread per edge. Thread ID $e \in [0, |E|-1]$ directly accesses $u = \text{src}[e]$ and $v = \text{dst}[e]$ in $O(1)$ time without divergent binary search, atomically propagating updates to $v$.
   * **Apply Kernel (Vertex-Parallel):** 1 thread per vertex $v \in [0, |V|-1]$ applying post-reduction normalization (e.g. PageRank damping $(1-d)/|V| + d \times \dots$).

---

## 5. Team Work Breakdown (3 Engineers)

* **Member 1 (Frontend):**
  * `lib/edsl/`: Phantom-typed smart constructors (`(+:)`, `(-:)`, `min_expr`).
  * Vertex/edge state schema builders (`field "dist" TyF32`).
  * Algorithm authoring in DSL (SSSP, PageRank).
  * Lowering from EDSL to `GraphIR`.
* **Member 2 (Mid-end / IR Core):**
  * `lib/lowerIR/`: Virtual register allocation and SSA tracking.
  * Lowering from `GraphIR` to `LowerIR`.
  * Grid boundary guard insertion (`tid < num_items`).
  * SoA 64-bit wide address arithmetic calculation (`MulWideU32`, `AddU64`).
  * CAS loop expansion for float min/max.
* **Member 3 (Backend & Runtime):**
  * `lib/ptx/`: Lowering from `LowerIR` to `Ptx` AST and `.ptx` string emission.
  * Automated syntax validation via `/usr/local/cuda-13.3/bin/ptxas -arch=sm_80`.
  * Standalone C++ test runner (`cuda.h`, `cuModuleLoadData`, `cuLaunchKernel`).
  * CPU golden reference implementation & A30 benchmark validation.

---

## 6. Standard Development Workflows

### Build the Compiler
```bash
dune build
```

### Run Test Suite
```bash
dune runtest
```

### Validate Emitted PTX with NVIDIA Assembler
```bash
/usr/local/cuda-13.3/bin/ptxas -arch=sm_80 -m64 output_kernel.ptx
```

### Build Documentation & Diagrams
```bash
make -C docs
```

---

## 7. Code Style & Documentation Guidelines

* **Comments:** Maintain docstrings on all exported types and functions (`(** ... *)`).
* **Constructor Comments:** Every variant constructor in `ast.ml` and `types.ml` must include an inline doc-comment (`(** ... *)`) explaining its semantic role.
* **Multiplication Syntax in Comments:** In OCaml comments, avoid writing `(*)`, as `*)` closes the comment block. Always add spaces: `( * )`.
* **No Premature Lowering:** Keep each IR layer strictly isolated to its respective domain. Never mix hardware register allocation inside `GraphIR`.

---

## 8. Documentation Reference Protocol

AI agents and developers should routinely consult the project and hardware documentation:
* **Project Specifications & Mathematics:** Refer to [`docs/planning.md`](docs/planning.md) and [`docs/abstract.md`](docs/abstract.md) for complete mathematical definitions of SSSP and PageRank, algorithm invariants, and high-level compiler pipeline details.
* **Architecture Diagrams & Visuals:** Refer to [`docs/architecture.mmd`](docs/architecture.mmd) and [`docs/abstract.typ`](docs/abstract.typ) whenever inspecting module dependencies or pipeline stages.
* **NVIDIA PTX Hardware Reference:** When authoring or validating PTX instructions for the NVIDIA A30 (`sm_80`), consult the official NVIDIA PTX ISA Version 7.0+ documentation (specifically regarding `.target sm_80`, 64-bit addressing, and hardware atomics).
* **CUDA Driver API:** Refer to CUDA Driver API documentation for module loading (`cuModuleLoadData`) and execution dispatch (`cuLaunchKernel`).

---

## 9. Updating AGENTS.md (User Permission Required)

* **Keeping the Guide Current:** As the project progresses through EDSL design, lowering passes, and host harness implementation, `AGENTS.md` should be kept in sync with new conventions, module additions, and compiler passes.
* **User Permission Rule:** AI agents **MUST NOT** modify or overwrite `AGENTS.md` unilaterally. Agents must present proposed changes, explain the rationale, and obtain **explicit permission from the user** before making any edits to this file.
