(** {1 Built-in GAS-DSL Benchmark Examples}

    Reference Gather-Apply-Scatter (GAS) algorithm definitions authored
    using {!Gas_dsl.Edsl}. *)

(** Single-Source Shortest Path with 32-bit unsigned integer distances. *)
val sssp_u32 : GraphIR.Ast.algorithm

(** Single-Source Shortest Path with 32-bit float distances (CAS reduction). *)
val sssp_f32 : GraphIR.Ast.algorithm

(** Iterative PageRank with [Sum] scatter and damped [Apply] update. *)
val pagerank : GraphIR.Ast.algorithm

(** Association list of all built-in benchmark names to their algorithms. *)
val all : (string * GraphIR.Ast.algorithm) list
