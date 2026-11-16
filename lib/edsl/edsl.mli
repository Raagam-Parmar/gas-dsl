(** {1 GAS-DSL Frontend EDSL}

    This module provides a Bonsai-inspired, phantom-typed OCaml EDSL for
    authoring Gather-Apply-Scatter (GAS) graph algorithms.

    It enforces compile-time separation between:
    - Floating-point ([float expr]) and integer ([int expr]) expressions.
    - Vertex attributes ([vertex]) and edge attributes ([edge]).
    - Stage-valid location readers ([src], [dst], [edge] in Scatter;
      [self] in Apply).

    Algorithms constructed with {!algorithm} lower directly to
    {!Gas_dsl.GraphIR.Ast.algorithm}. *)

(** Phantom tag for vertex-scoped attribute fields. *)
type vertex

(** Phantom tag for edge-scoped attribute fields. *)
type edge

(** Typed attribute field descriptor parameterized by value type ['a]
    and graph entity kind ['loc] ([vertex] or [edge]). *)
type ('a, 'loc) field

(** Phantom-typed EDSL expression producing a scalar of type ['a]. *)
type 'a expr

(** Rank-2 polymorphic field reader passed to stage callbacks.
    Allows reading fields of different scalar types within the same callback. *)
type 'loc reader = {
  get : 'a. ('a, 'loc) field -> 'a expr; (** Load a field at this location. *)
}

(** Reduction operators supported during the atomic Scatter phase. *)
type reduction = GraphIR.Types.reduction_op =
  | Min (** Select the minimum value across neighbors. *)
  | Max (** Select the maximum value across neighbors. *)
  | Sum (** Sum values across all neighbors. *)

(** Vertex-scoped attribute field constructors. *)
module Vertex : sig
  (** [f32 name] declares a 32-bit float vertex property. *)
  val f32 : string -> (float, vertex) field

  (** [u32 name] declares an unsigned 32-bit integer vertex property. *)
  val u32 : string -> (int, vertex) field

  (** [i32 name] declares a signed 32-bit integer vertex property. *)
  val i32 : string -> (int, vertex) field

  (** [b8 name] declares an 8-bit boolean vertex property. *)
  val b8 : string -> (bool, vertex) field
end

(** Edge-scoped attribute field constructors. *)
module Edge : sig
  (** [f32 name] declares a 32-bit float edge property. *)
  val f32 : string -> (float, edge) field

  (** [u32 name] declares an unsigned 32-bit integer edge property. *)
  val u32 : string -> (int, edge) field

  (** [i32 name] declares a signed 32-bit integer edge property. *)
  val i32 : string -> (int, edge) field

  (** [b8 name] declares an 8-bit boolean edge property. *)
  val b8 : string -> (bool, edge) field
end

(** {2 Expression Constructors & Operators} *)

(** [float x] lifts a literal OCaml float into a [float expr]. *)
val float : float -> float expr

(** [int n] lifts a literal OCaml int into an [int expr]. *)
val int : int -> int expr

(** [a +: b] adds two expressions of the same scalar type. *)
val ( +: ) : 'a expr -> 'a expr -> 'a expr

(** [a -: b] subtracts [b] from [a]. *)
val ( -: ) : 'a expr -> 'a expr -> 'a expr

(** [a *: b] multiplies two expressions of the same scalar type. *)
val ( *: ) : 'a expr -> 'a expr -> 'a expr

(** [a /: b] divides [a] by [b]. *)
val ( /: ) : 'a expr -> 'a expr -> 'a expr

(** [min a b] computes the minimum of two expressions. *)
val min : 'a expr -> 'a expr -> 'a expr

(** [max a b] computes the maximum of two expressions. *)
val max : 'a expr -> 'a expr -> 'a expr

(** {2 Stage & Algorithm Builders} *)

(** [scatter ~target ~reduce fn] builds an edge-parallel Scatter stage.
    [fn] receives polymorphic readers for [~src], [~dst], and [~edge]. *)
val scatter :
  target:('a, vertex) field ->
  reduce:reduction ->
  (src:vertex reader -> dst:vertex reader -> edge:edge reader -> 'a expr) ->
  GraphIR.Ast.scatter_stage

(** [apply ~dest fn] builds a vertex-parallel Apply stage.
    [fn] receives a polymorphic reader for [~self]. *)
val apply :
  dest:('a, vertex) field ->
  (self:vertex reader -> 'a expr) ->
  GraphIR.Ast.apply_stage

(** [algorithm ~name ~scatter ?apply ()] assembles a complete {!GraphIR.Ast.algorithm},
    automatically collecting and deduplicating [vertex_state] and [edge_state]
    from the referenced fields.

    Raises [Invalid_argument] if a field name is used with conflicting types. *)
val algorithm :
  name:string ->
  scatter:GraphIR.Ast.scatter_stage ->
  ?apply:GraphIR.Ast.apply_stage ->
  unit ->
  GraphIR.Ast.algorithm
