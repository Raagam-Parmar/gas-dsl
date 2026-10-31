(** {1 GraphIR Types}

    This module defines domain-level primitive types, reduction operators,
    and memory locations for static graph analytics in the Gather-Apply-Scatter (GAS)
    programming model.

    These types represent the semantic properties of graph state before any
    lowering to GPU hardware registers or memory spaces takes place. *)

(** Primitive scalar data types supported in graph state arrays. *)
type prim_type =
  | TyF32 (** 32-bit single-precision IEEE 754 floating point (e.g., PageRank rank, float weights) *)
  | TyU32 (** 32-bit unsigned integer (e.g., vertex IDs, degree counts, uint weights) *)
  | TyI32 (** 32-bit signed integer (e.g., delta values, signed counters) *)
  | TyB8  (** 8-bit boolean flag (e.g., active frontier mask, convergence dirty bit) *)

(** Reduction operators used to combine gathered neighbor data during Scatter. *)
type reduction_op =
  | Min (** Select the minimum value across neighbors (used in SSSP) *)
  | Max (** Select the maximum value across neighbors (used in Connected Components) *)
  | Sum (** Sum values across all neighbors (used in PageRank) *)

(** Location contexts indicating where a state attribute is being accessed. *)
type location =
  | Src      (** State array of the source vertex [u] of an edge [(u, v)] *)
  | Dst      (** State array of the destination vertex [v] of an edge [(u, v)] *)
  | EdgeAttr (** Edge attribute or weight array of edge [e] *)
  | Self     (** Current vertex state in the vertex-parallel Apply phase *)

(** A named, typed attribute descriptor belonging to vertex or edge state. *)
type field = {
  name : string;    (** Unique name identifier for the state buffer (e.g., "dist", "rank") *)
  ty   : prim_type; (** Primitive type of each element in the buffer *)
}
