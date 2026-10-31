#set page(
  paper: "a4",
  margin: (x: 2cm, top: 2cm, bottom: 2cm),
  header: context {
    if here().page() > 1 {
      align(right, text(size: 8.5pt, fill: luma(120), [GAS-DSL: Project Abstract & Architecture Specification]))
    }
  },
  footer: context {
    align(center, text(size: 8.5pt, fill: luma(120), counter(page).display("1 of 1", both: true)))
  }
)

#set text(
  font: ("Liberation Sans", "DejaVu Sans"),
  size: 10pt,
  spacing: 115%,
  lang: "en",
  hyphenate: false
)

#set par(justify: true, leading: 0.65em)

// Document Header
#align(center)[
  #block(width: 100%)[
    #text(size: 18pt, weight: "bold", fill: rgb("#1d3557"))[GAS-DSL: Gather-Apply-Scatter EDSL for GPU Graph Analytics] \
    #v(3pt)
    #text(size: 11pt, weight: "medium", fill: rgb("#457b9d"))[Project Abstract & Architecture Specification] \
    #v(6pt)
    #line(length: 100%, stroke: 1.2pt + rgb("#1d3557"))
  ]
]

#v(4pt)

== 1. Summary

*GAS-DSL* is a statically-typed domain-specific language embedded in OCaml designed to compile Gather-Apply-Scatter (GAS) graph processing algorithms directly into standalone NVIDIA PTX (Parallel Thread Execution) assembly. Targeting static graphs stored in an augmented Compressed Sparse Row (CSR) format, the compiler translates high-level graph specifications into decoupled, high-performance GPU kernels for iterative analytics such as *Single-Source Shortest Path (SSSP)* and *PageRank*, without requiring complex host runtimes or dynamic GPU memory management.

#v(6pt)

== 2. Architecture Design

The system decouples compilation from runtime execution. The OCaml frontend takes user-defined vertex/edge schemas and GAS logic, lowering them through a platform-agnostic intermediate representation into optimized PTX text. The generated kernels are loaded and dispatched by a lightweight host test harness via the low-level CUDA Driver API.

#v(4pt)

#align(center)[
  #figure(
    image("architecture.png", height: 13.5cm),
    caption: [GAS-DSL End-to-End Compiler and Host Execution Architecture.]
  ) <fig:arch>
]

#pagebreak()

== 3. Design Decisions

#v(8pt)

#show table.cell: set text(hyphenate: false)
#show table.cell: set par(justify: false, leading: 0.6em)

#set table(
  stroke: (x, y) => if y == 0 { (bottom: 1.5pt + rgb("#1d3557")) } else { 0.5pt + rgb("#d1d5db") },
  fill: (col, row) => if row == 0 { rgb("#f1faee") } else if calc.even(row) { rgb("#f8fafc") } else { none },
  inset: (x: 10pt, y: 9pt)
)

#table(
  columns: (1.3fr, 1.4fr, 2.3fr),
  align: (col, row) => if row == 0 { left + horizon } else { left + top },
  [#text(weight: "bold", fill: rgb("#1d3557"))[Subsystem]],
  [#text(weight: "bold", fill: rgb("#1d3557"))[Decision]],
  [#text(weight: "bold", fill: rgb("#1d3557"))[Rationale]],

  [*Execution Model*],
  [*Edge-Parallel Kernel + Vertex-Parallel Apply*],
  [Decomposes execution into an edge-parallel Scatter kernel (1 thread per edge) for atomic propagation, and a vertex-parallel Apply kernel for post-reduction state updates (e.g., PageRank damping). This avoids load-imbalance on power-law graphs and eliminates in-kernel binary searches.],

  [*Graph Representation*],
  [*Augmented CSR (COO-style edge source array)*],
  [Augments standard CSR arrays (`row_offsets`, `dst_indices`, `weights`) with a precomputed `src_indices` array. Edge threads retrieve source and target vertices in $O(1)$ time without divergent control flow.],

  [*Memory Layout*],
  [*Structure-of-Arrays (SoA)*],
  [Vertex and edge properties (`dist`, `rank`, `out_deg`) reside in independent global memory arrays passed as 64-bit pointers (`.param .u64`), guaranteeing coalesced memory transactions.],

  [*EDSL Typing*],
  [*Phantom Types (`'a expr`) over IR ADT*],
  [Uses smart constructors tagged with phantom types to enforce type safety in OCaml at compile time while lowering to an untyped, easily optimizable linear IR for code generation.],

  [*Verification*],
  [*Minimal C++ CUDA Driver Harness*],
  [A lightweight standalone test harness using the CUDA Driver API (`cuda.h`, `cuModuleLoadData`, `cuLaunchKernel`) loads the emitted `.ptx` files, manages host-device data transfers, and validates results against a sequential CPU reference.]
)
