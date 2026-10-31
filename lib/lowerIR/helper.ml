open Types
open Ast

module P  = Ptx.Types
module PA = Ptx.Ast

(* ── type / register / operand mappings ─────────────────────────────── *)

let lower_type : data_type -> P.ptx_type = function
  | U32  -> P.U32  | U64 -> P.U64
  | S32  -> P.S32  | S64 -> P.S64
  | F32  -> P.F32  | B32 -> P.B32
  | Pred -> P.Pred

let lower_reg (r : reg) : P.ptx_reg =
  match r.ty with
  | U32 | S32 | B32 -> P.R  r.id
  | U64 | S64       -> P.RD r.id
  | F32             -> P.F  r.id
  | Pred            -> P.P  r.id

let lower_operand : operand -> PA.ptx_operand = function
  | Reg r    -> PA.Reg (lower_reg r)
  | ImmU32 i -> PA.ImmInt (Int64.of_int32 i)
  | ImmU64 i -> PA.ImmInt i
  | ImmF32 f -> PA.ImmFloat f

let lower_atomic_action : atomic_op -> P.atomic_action = function
  | AtomAdd -> P.AtomAdd | AtomMin -> P.AtomMin
  | AtomMax -> P.AtomMax | AtomCas -> P.AtomCas

(* ── setp suffix builder ─────────────────────────────────────────────── *)

let cmp_str (op : cmp_op) (ty : data_type) : string =
  let suffix = match ty with
    | U32 -> "u32" | U64 -> "u64" | S32 -> "s32" | S64 -> "s64"
    | F32 -> "f32" | B32 -> "b32" | Pred -> "pred"
  in
  let rel = match op with
    | Eq -> "eq" | Ne -> "ne" | Lt -> "lt"
    | Le -> "le" | Gt -> "gt" | Ge -> "ge"
  in
  rel ^ "." ^ suffix

(* ── binop lowering ──────────────────────────────────────────────────── *)

let lower_arith (op : binop) (ty : P.ptx_type) (dst : P.ptx_reg)
    (a : PA.ptx_operand) (b : PA.ptx_operand) : PA.statement =
  match op with
  | Add                         -> PA.Add (ty, dst, a, b)
  | Sub                         -> PA.Sub (ty, dst, a, b)
  | Mul | Div | And | Or
  | Xor | Shl | Shr             -> PA.Mul (ty, dst, a, b)

(* ── CAS loop expansion ──────────────────────────────────────────────── *)

(* Expands CasLoop into the PTX CAS retry sequence (AGENTS.md §3.3):
     $CAS_LOOP_N:
       ld.global.f32  %cur_val, [%addr]
       setp.ge.f32    %pred,    %cur_val, %new_val   (min; le for max)
       @%pred bra     $CAS_END_N
       atom.global.cas.b32 %old, [%addr], %cur_bits, %new_bits
       setp.eq.b32    %succ,    %old, %cur_bits
       @!%succ bra    $CAS_LOOP_N
     $CAS_END_N:
*)
let expand_cas_loop ~target_addr ~new_val ~cur_val ~pred_reg ~is_min
    (loop_id : int) : PA.statement list =
  let loop_lbl  = Printf.sprintf "$CAS_LOOP_%d" loop_id in
  let end_lbl   = Printf.sprintf "$CAS_END_%d"  loop_id in
  let addr_op   = PA.Mem (lower_reg target_addr, None) in
  let cur_f     = lower_reg cur_val  in
  let new_f     = lower_reg new_val  in
  let pred      = lower_reg pred_reg in
  let skip_cmp  = if is_min then "ge.f32" else "le.f32" in
  let succ_pred = match pred with P.P i -> P.P (i + 1) | r -> r in
  let to_r32    = function P.F i -> P.R i | r -> r in
  [
    PA.LabelDef loop_lbl;
    PA.LdGlobal (P.F32, cur_f, addr_op, None);
    PA.SetP (skip_cmp, pred, PA.Reg cur_f, PA.Reg new_f);
    PA.Bra  (end_lbl, Some (pred, true));
    PA.AtomGlobal (P.AtomCas, P.B32, to_r32 cur_f, addr_op, PA.Reg (to_r32 new_f));
    PA.SetP ("eq.b32", succ_pred, PA.Reg (to_r32 cur_f), PA.Reg (to_r32 cur_f));
    PA.Bra  (loop_lbl, Some (succ_pred, false));
    PA.LabelDef end_lbl;
  ]

(* ── register bank counting & .reg declarations ─────────────────────── *)

type reg_counts = {
  mutable n_r  : int;
  mutable n_rd : int;
  mutable n_f  : int;
  mutable n_p  : int;
}

let count_regs (body : instruction list) : reg_counts =
  let c = { n_r = 0; n_rd = 0; n_f = 0; n_p = 0 } in
  let bump (r : reg) =
    let next = r.id + 1 in
    match r.ty with
    | U32 | S32 | B32 -> if next > c.n_r  then c.n_r  <- next
    | U64 | S64       -> if next > c.n_rd then c.n_rd <- next
    | F32             -> if next > c.n_f  then c.n_f  <- next
    | Pred            -> if next > c.n_p  then c.n_p  <- next
  in
  let bump_op = function Reg r -> bump r | _ -> () in
  List.iter (function
    | GetTid r | GetNtid r | GetCtaid r | GetThreadId r -> bump r
    | LoadParam (r, _)          -> bump r
    | CvtaToGlobal (d, s)       -> bump d; bump s
    | LoadGlobal  (d, a, _)     -> bump d; bump a
    | StoreGlobal (a, s, _)     -> bump a; bump s
    | MulWideU32  (d, s, _)     -> bump d; bump s
    | AddU64      (d, b, o)     -> bump d; bump b; bump o
    | Arith (_, d, a, b, _)     -> bump d; bump_op a; bump_op b
    | Mad (d, a, b, c, _)       -> bump d; bump_op a; bump_op b; bump_op c
    | Compare (_, d, a, b, _)   -> bump d; bump_op a; bump_op b
    | Atomic (_, d, a, v, _)    -> bump d; bump a; bump_op v
    | CasLoop { target_addr; new_val; cur_val; pred_reg; _ } ->
        bump target_addr; bump new_val; bump cur_val; bump pred_reg;
        let extra_p = pred_reg.id + 2 in
        if extra_p > c.n_p then c.n_p <- extra_p
    | BranchIf (p, _)           -> bump p
    | Label _ | Branch _ | Return -> ()
  ) body;
  c

let reg_decls (c : reg_counts) : PA.statement list =
  let acc = [] in
  let acc = if c.n_r  > 0 then PA.RegDecl (P.U32,  c.n_r)  :: acc else acc in
  let acc = if c.n_rd > 0 then PA.RegDecl (P.U64,  c.n_rd) :: acc else acc in
  let acc = if c.n_f  > 0 then PA.RegDecl (P.F32,  c.n_f)  :: acc else acc in
  let acc = if c.n_p  > 0 then PA.RegDecl (P.Pred, c.n_p)  :: acc else acc in
  List.rev acc
