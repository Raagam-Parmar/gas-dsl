(** {1 GraphIR AST}

    This module defines the abstract syntax tree for the high-level Graph IR.
    It models graph algorithms using Gather-Apply-Scatter (GAS) semantics:
    - Expressions evaluate vertex and edge properties.
    - Scatter stages gather edge/neighbor data and reduce into target vertices.
    - Apply stages update local vertex state after scatter reduction.

    This IR is hardware-agnostic and free of machine registers or thread indexing. *)

open Types

(** Binary operators supported in graph expressions. *)
type binop =
  | Add   (** Arithmetic addition (+) *)
  | Sub   (** Arithmetic subtraction (-) *)
  | Mul   (** Arithmetic multiplication ( * ) *)
  | Div   (** Arithmetic division (/) *)
  | MinOp (** Minimum of two scalar expressions *)
  | MaxOp (** Maximum of two scalar expressions *)

(** Expression tree representing computations over vertex and edge properties. *)
type expr =
  | ConstFloat of float                  (** Literal floating-point constant (e.g., 0.85 in PageRank) *)
  | ConstInt   of int                    (** Literal integer constant (e.g., 0 for root distance) *)
  | LoadField  of field * location       (** Load property from a vertex or edge attribute array *)
  | BinOp      of binop * expr * expr    (** Binary arithmetic or comparison expression *)

(** Specification of the edge-parallel Scatter phase.
    Each edge thread gathers data from the source vertex and edge weight,
    and atomically reduces the result into the destination vertex state. *)
type scatter_stage = {
  gather_expr  : expr;         (** Expression evaluated per edge using source/edge properties *)
  target_field : field;        (** Target vertex state array to update atomically *)
  reduction    : reduction_op; (** Reduction operation (Min, Max, or Sum) applied at target *)
}

(** Specification of the vertex-parallel Apply phase.
    Each vertex thread updates its own local state after the scatter phase completes
    (e.g., applying PageRank damping factor and normalization). *)
type apply_stage = {
  update_expr  : expr;  (** Expression computed per vertex to calculate updated state *)
  dest_field   : field; (** Destination vertex state array to store the result *)
}

(** Complete specification of an iterative graph algorithm. *)
type algorithm = {
  name         : string;           (** Descriptive name of algorithm (e.g., "sssp", "pagerank") *)
  vertex_state : field list;       (** List of vertex state buffers in Structure-of-Arrays (SoA) layout *)
  edge_state   : field list;       (** List of edge attribute buffers (e.g., weights) *)
  scatter      : scatter_stage;    (** Edge-parallel scatter stage *)
  apply        : apply_stage option; (** Optional vertex-parallel apply stage *)
}
