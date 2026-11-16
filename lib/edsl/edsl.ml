(** {1 GAS-DSL Frontend EDSL}

    Implementation of the Bonsai-inspired OCaml EDSL combinators and
    automatic schema collection lowering into {!GraphIR.Ast.algorithm}. *)

module GT = GraphIR.Types
module GA = GraphIR.Ast

type vertex = unit
type edge   = unit

type ('a, 'loc) field = GT.field
type 'a expr = GA.expr

type 'loc reader = {
  get : 'a. ('a, 'loc) field -> 'a expr;
}

type reduction = GT.reduction_op =
  | Min
  | Max
  | Sum

(* Shared smart constructor helper for typed attribute descriptors *)
let make_field (name : string) (ty : GT.prim_type) : ('a, 'loc) field =
  { GT.name; ty }

module Vertex = struct
  (** [f32 name] declares a 32-bit float vertex property. *)
  let f32 name = make_field name GT.TyF32

  (** [u32 name] declares an unsigned 32-bit integer vertex property. *)
  let u32 name = make_field name GT.TyU32

  (** [i32 name] declares a signed 32-bit integer vertex property. *)
  let i32 name = make_field name GT.TyI32

  (** [b8 name] declares an 8-bit boolean vertex property. *)
  let b8 name = make_field name GT.TyB8
end

module Edge = struct
  (** [f32 name] declares a 32-bit float edge property. *)
  let f32 name = make_field name GT.TyF32

  (** [u32 name] declares an unsigned 32-bit integer edge property. *)
  let u32 name = make_field name GT.TyU32

  (** [i32 name] declares a signed 32-bit integer edge property. *)
  let i32 name = make_field name GT.TyI32

  (** [b8 name] declares an 8-bit boolean edge property. *)
  let b8 name = make_field name GT.TyB8
end

(** [float x] lifts a literal OCaml float into a [float expr]. *)
let float x = GA.ConstFloat x

(** [int n] lifts a literal OCaml int into an [int expr]. *)
let int n = GA.ConstInt n

(** [a +: b] adds two expressions of the same scalar type. *)
let ( +: ) a b = GA.BinOp (GA.Add, a, b)

(** [a -: b] subtracts [b] from [a]. *)
let ( -: ) a b = GA.BinOp (GA.Sub, a, b)

(** [a *: b] multiplies two expressions of the same scalar type. *)
let ( *: ) a b = GA.BinOp (GA.Mul, a, b)

(** [a /: b] divides [a] by [b]. *)
let ( /: ) a b = GA.BinOp (GA.Div, a, b)

(** [min a b] computes the minimum of two expressions. *)
let min a b = GA.BinOp (GA.MinOp, a, b)

(** [max a b] computes the maximum of two expressions. *)
let max a b = GA.BinOp (GA.MaxOp, a, b)

(* Build a polymorphic reader pinned to a fixed GraphIR location *)
let make_reader (loc : GT.location) : 'loc reader =
  { get = (fun f -> GA.LoadField (f, loc)) }

(** [scatter ~target ~reduce fn] builds an edge-parallel Scatter stage. *)
let scatter
    ~(target : ('a, vertex) field)
    ~(reduce : reduction)
    (fn : src:vertex reader -> dst:vertex reader -> edge:edge reader -> 'a expr)
  : GA.scatter_stage =
  let src  = make_reader GT.Src in
  let dst  = make_reader GT.Dst in
  let edge = make_reader GT.EdgeAttr in
  let gather_expr = fn ~src ~dst ~edge in
  { GA.gather_expr; target_field = target; reduction = reduce }

(** [apply ~dest fn] builds a vertex-parallel Apply stage. *)
let apply
    ~(dest : ('a, vertex) field)
    (fn : self:vertex reader -> 'a expr)
  : GA.apply_stage =
  let self = make_reader GT.Self in
  let update_expr = fn ~self in
  { GA.update_expr; dest_field = dest }

(* Deduplicate fields by name while preserving first-seen order and checking type consistency *)
let record_field
    (seen : (string, GT.prim_type) Hashtbl.t)
    (acc  : GT.field list ref)
    (f    : GT.field)
  : unit =
  match Hashtbl.find_opt seen f.GT.name with
  | None ->
      Hashtbl.add seen f.GT.name f.GT.ty;
      acc := f :: !acc
  | Some prev_ty ->
      if prev_ty <> f.GT.ty then
        invalid_arg
          (Printf.sprintf "Edsl.algorithm: field %S used with conflicting types" f.GT.name)

(* Walk an expression tree and register vertex vs. edge fields based on access location *)
let rec collect_expr
    ~(v_seen : (string, GT.prim_type) Hashtbl.t)
    ~(v_acc  : GT.field list ref)
    ~(e_seen : (string, GT.prim_type) Hashtbl.t)
    ~(e_acc  : GT.field list ref)
    (expr    : GA.expr)
  : unit =
  match expr with
  | GA.ConstFloat _ | GA.ConstInt _ -> ()
  | GA.LoadField (f, loc) ->
      (match loc with
       | GT.Src | GT.Dst | GT.Self -> record_field v_seen v_acc f
       | GT.EdgeAttr               -> record_field e_seen e_acc f)
  | GA.BinOp (_, lhs, rhs) ->
      collect_expr ~v_seen ~v_acc ~e_seen ~e_acc lhs;
      collect_expr ~v_seen ~v_acc ~e_seen ~e_acc rhs

(** [algorithm ~name ~scatter ?apply ()] assembles a complete {!GraphIR.Ast.algorithm},
    automatically collecting and deduplicating [vertex_state] and [edge_state]. *)
let algorithm
    ~(name    : string)
    ~(scatter : GA.scatter_stage)
    ?(apply   : GA.apply_stage option)
    ()
  : GA.algorithm =
  let v_seen = Hashtbl.create 8 in
  let v_acc  = ref [] in
  let e_seen = Hashtbl.create 4 in
  let e_acc  = ref [] in
  collect_expr ~v_seen ~v_acc ~e_seen ~e_acc scatter.GA.gather_expr;
  record_field v_seen v_acc scatter.GA.target_field;
  (match apply with
   | None -> ()
   | Some a ->
       collect_expr ~v_seen ~v_acc ~e_seen ~e_acc a.GA.update_expr;
       record_field v_seen v_acc a.GA.dest_field);
  {
    GA.name;
    vertex_state = List.rev !v_acc;
    edge_state   = List.rev !e_acc;
    scatter;
    apply;
  }
