(** {1 GraphIR → LowerIR Helper Utilities}

    This module provides the supporting infrastructure consumed by
    {!GraphIR.Translate} when lowering a high-level {!GraphIR.Ast.algorithm}
    into {!LowerIR.Ast.kernel} sequences.  It contains:

    - A monotonic virtual-register allocator.
    - Mappings from {!GraphIR.Types.prim_type} to {!LowerIR.Types.data_type}.
    - Mappings from {!GraphIR.Types.reduction_op} to {!LowerIR.Types.atomic_op}.
    - Helpers for building the standard SoA kernel parameter lists.

    No hardware PTX constructs appear here; all decisions remain at the
    LowerIR abstraction boundary. *)

open Types
module LT = LowerIR.Types


(** Mutable counter used to generate fresh, monotonically increasing
    virtual-register IDs within a single kernel lowering scope. *)
type reg_state = { mutable next_id : int }

(** [make_state ()] creates a fresh register allocator seeded at zero. *)
let make_state () : reg_state = { next_id = 0 }

(** [fresh state ty] allocates a new {!LowerIR.Types.reg} with the given
    [ty] and advances the internal counter. *)
let fresh (s : reg_state) (ty : LT.data_type) : LT.reg =
  let id = s.next_id in
  s.next_id <- id + 1;
  { LT.id = id; LT.ty }


(** [lower_prim_type t] maps a {!GraphIR.Types.prim_type} to the
    corresponding {!LowerIR.Types.data_type}.

    - [TyF32] → [F32]   (32-bit float register / memory slot)
    - [TyU32] → [U32]   (32-bit unsigned integer)
    - [TyI32] → [S32]   (32-bit signed integer)
    - [TyB8]  → [U32]   (8-bit booleans are widened to 32-bit for register use;
                          hardware does not have 8-bit register banks) *)
let lower_prim_type : prim_type -> LT.data_type = function
  | TyF32 -> LT.F32
  | TyU32 -> LT.U32
  | TyI32 -> LT.S32
  | TyB8  -> LT.U32

(** [stride_of_prim t] returns the element byte stride for a
    {!GraphIR.Types.prim_type} as used in wide-multiply address scaling.
    All supported types occupy 4 bytes in global memory. *)
let stride_of_prim : prim_type -> int = function
  | TyF32 | TyU32 | TyI32 | TyB8 -> 4


(** Describes how a {!GraphIR.Types.reduction_op} maps to GPU hardware.

    - [NativeAtomic op]: the reduction can be performed with a single
      hardware atomic instruction (e.g. [atom.global.min.u32]).
    - [CasMinF32]:       float min has no native PTX atomic; it must be
      synthesised as a Compare-and-Swap retry loop (see AGENTS.md §3.3).
    - [CasMaxF32]:       analogous float max CAS loop. *)
type atomic_strategy =
  | NativeAtomic of LT.atomic_op (** Direct hardware atomic suffices *)
  | CasMinF32                    (** Float min via CAS retry loop     *)
  | CasMaxF32                    (** Float max via CAS retry loop     *)

(** [atomic_strategy_of field reduction] derives the {!atomic_strategy}
    required to implement [reduction] on the given target [field].

    Integer Min/Max use native [atom.min] / [atom.max].
    Float Sum uses native [atom.add.f32].
    Float Min and Float Max must fall back to CAS loops because PTX has
    no [atom.min.f32] or [atom.max.f32] instruction. *)
let atomic_strategy_of (f : field) (r : reduction_op) : atomic_strategy =
  match r, f.ty with
  | Min, TyU32 -> NativeAtomic LT.AtomMin
  | Min, TyI32 -> NativeAtomic LT.AtomMin
  | Min, TyF32 -> CasMinF32
  | Max, TyU32 -> NativeAtomic LT.AtomMax
  | Max, TyI32 -> NativeAtomic LT.AtomMax
  | Max, TyF32 -> CasMaxF32
  | Sum, _     -> NativeAtomic LT.AtomAdd
  | Min, TyB8  -> NativeAtomic LT.AtomMin (* treat boolean min as AND-like *)
  | Max, TyB8  -> NativeAtomic LT.AtomMax


(** [scatter_params vertex_fields edge_fields] constructs the typed
    parameter list for an edge-parallel Scatter kernel.

    Every SoA buffer is passed as a 64-bit pointer ([U64]).  In addition,
    the kernel receives two 32-bit scalars: [num_edges] and [num_vertices].

    Parameter name convention (matches host harness expectations):
    - Vertex buffers: ["{name}_ptr"]
    - Edge buffers:   ["{name}_ptr"]
    - Scalars:        ["num_edges"], ["num_vertices"] *)
let scatter_params
    (vertex_fields : field list)
    (edge_fields   : field list)
  : (string * LT.data_type) list =
  let ptr_params =
    List.map (fun f -> (f.name ^ "_ptr", LT.U64))
      (vertex_fields @ edge_fields)
  in
  (* Augmented CSR arrays: row_offsets, src_indices, dst_indices are always present *)
  let csr_params = [
    ("row_offsets_ptr", LT.U64);
    ("src_indices_ptr", LT.U64);
    ("dst_indices_ptr", LT.U64);
  ] in
  ptr_params @ csr_params @ [
    ("num_edges",    LT.U32);
    ("num_vertices", LT.U32);
  ]

(** [apply_params vertex_fields] constructs the typed parameter list for a
    vertex-parallel Apply kernel.

    Only vertex SoA buffers and the [num_vertices] scalar are required;
    edge data and CSR structure are not accessed during Apply. *)
let apply_params (vertex_fields : field list) : (string * LT.data_type) list =
  let ptr_params =
    List.map (fun f -> (f.name ^ "_ptr", LT.U64)) vertex_fields
  in
  ptr_params @ [ ("num_vertices", LT.U32) ]
