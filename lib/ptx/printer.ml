(** {1 PTX Pretty Printer}

    This module provides pretty-printing combinators for serializing the
    PTX abstract syntax tree into syntactically valid NVIDIA PTX assembly text
    targeting Compute Capability [sm_75] (NVIDIA Ampere / A30 architecture).

    It uses the [Fmt] library to format directives, register banks,
    kernel signatures, memory operations, hardware atomics, and control flow. *)

open Types
open Ast

(** Pretty-prints a PTX primitive data type. *)
let pp_ptx_type ppf = function
  | U32  -> Fmt.string ppf ".u32"
  | U64  -> Fmt.string ppf ".u64"
  | S32  -> Fmt.string ppf ".s32"
  | S64  -> Fmt.string ppf ".s64"
  | F32  -> Fmt.string ppf ".f32"
  | B32  -> Fmt.string ppf ".b32"
  | B64  -> Fmt.string ppf ".b64"
  | Pred -> Fmt.string ppf ".pred"

(** Pretty-prints a PTX state memory space. *)
let pp_state_space ppf = function
  | Global -> Fmt.string ppf ".global"
  | Param  -> Fmt.string ppf ".param"
  | Shared -> Fmt.string ppf ".shared"
  | Reg    -> Fmt.string ppf ".reg"

(** Pretty-prints a typed PTX virtual register identifier. *)
let pp_ptx_reg ppf = function
  | R  i -> Fmt.pf ppf "%%r%d" i
  | RD i -> Fmt.pf ppf "%%rd%d" i
  | F  i -> Fmt.pf ppf "%%f%d" i
  | P  i -> Fmt.pf ppf "%%p%d" i

(** Pretty-prints a memory cache modifier. *)
let pp_cache_modifier ppf = function
  | CA -> Fmt.string ppf ".ca"
  | CG -> Fmt.string ppf ".cg"
  | CS -> Fmt.string ppf ".cs"

(** Pretty-prints an atomic action keyword. *)
let pp_atomic_action ppf = function
  | AtomAdd -> Fmt.string ppf "add"
  | AtomMin -> Fmt.string ppf "min"
  | AtomMax -> Fmt.string ppf "max"
  | AtomCas -> Fmt.string ppf "cas"

(** Pretty-prints a PTX instruction operand. *)
let pp_operand ppf = function
  | Reg r -> pp_ptx_reg ppf r
  | SpecialReg s -> Fmt.pf ppf "%s" s
  | ImmInt i -> Fmt.int64 ppf i
  | ImmFloat f ->
      (* Format whole float constants with decimal point for PTX parser *)
      if Float.is_integer f then
        Fmt.pf ppf "%.1f" f
      else
        Fmt.pf ppf "%f" f
  | Mem (r, None) -> Fmt.pf ppf "[%a]" pp_ptx_reg r
  | Mem (r, Some off) ->
      if off >= 0 then
        Fmt.pf ppf "[%a+%d]" pp_ptx_reg r off
      else
        Fmt.pf ppf "[%a%d]" pp_ptx_reg r off
  | ParamRef name -> Fmt.pf ppf "[%s]" name

(** Pretty-prints an assembler header directive. *)
let pp_directive ppf = function
  | Version v     -> Fmt.pf ppf ".version %.1f" v
  | Target arch   -> Fmt.pf ppf ".target %s" arch
  | AddressSize b -> Fmt.pf ppf ".address_size %d" b

(* Helper to determine the register bank prefix corresponding to a ptx_type *)
let reg_bank_prefix = function
  | U32 | S32 | B32 -> "%r"
  | U64 | S64 | B64 -> "%rd"
  | F32             -> "%f"
  | Pred            -> "%p"

(** Pretty-prints a single PTX statement or instruction. *)
let pp_statement ppf = function
  | RegDecl (ty, count) ->
      let prefix = reg_bank_prefix ty in
      Fmt.pf ppf "    .reg %a %s<%d>;" pp_ptx_type ty prefix count

  | LabelDef name ->
      (* Labels are rendered flush to the left margin *)
      Fmt.pf ppf "%s:" name

  | Mov (ty, dst, src) ->
      Fmt.pf ppf "    mov%a %a, %a;" pp_ptx_type ty pp_ptx_reg dst pp_operand src

  | CvtaToGlobal (dst, src) ->
      Fmt.pf ppf "    cvta.to.global.u64 %a, %a;" pp_ptx_reg dst pp_ptx_reg src

  | LdParam (ty, dst, param_name) ->
      Fmt.pf ppf "    ld.param%a %a, [%s];" pp_ptx_type ty pp_ptx_reg dst param_name

  | LdGlobal (ty, dst, addr, mod_opt) ->
      let mod_str =
        match mod_opt with
        | None   -> ""
        | Some m -> Fmt.str "%a" pp_cache_modifier m
      in
      Fmt.pf ppf "    ld.global%s%a %a, %a;" mod_str pp_ptx_type ty pp_ptx_reg dst pp_operand addr

  | StGlobal (ty, addr, src, mod_opt) ->
      let mod_str =
        match mod_opt with
        | None   -> ""
        | Some m -> Fmt.str "%a" pp_cache_modifier m
      in
      Fmt.pf ppf "    st.global%s%a %a, %a;" mod_str pp_ptx_type ty pp_operand addr pp_ptx_reg src

  | Add (ty, dst, src1, src2) ->
      Fmt.pf ppf "    add%a %a, %a, %a;" pp_ptx_type ty pp_ptx_reg dst pp_operand src1 pp_operand src2

  | Sub (ty, dst, src1, src2) ->
      Fmt.pf ppf "    sub%a %a, %a, %a;" pp_ptx_type ty pp_ptx_reg dst pp_operand src1 pp_operand src2

  | Mul (ty, dst, src1, src2) ->
      (* Floating point multiplication uses mul.f32; integer uses mul.lo.type *)
      let op =
        match ty with
        | F32 -> "mul.f32"
        | _   -> Fmt.str "mul.lo%a" pp_ptx_type ty
      in
      Fmt.pf ppf "    %s %a, %a, %a;" op pp_ptx_reg dst pp_operand src1 pp_operand src2

  | Mad (ty, dst, src1, src2, src3) ->
      let op =
        match ty with
        | S32 -> "mad.lo.s32"
        | U32 -> "mad.lo.u32"
        | _   -> Fmt.str "mad.lo%a" pp_ptx_type ty
      in
      Fmt.pf ppf "    %s %a, %a, %a, %a;" op pp_ptx_reg dst pp_operand src1 pp_operand src2 pp_operand src3

  | MulWideU32 (dst, src1, src2) ->
      (* 32-bit to 64-bit wide multiply for pointer indexing *)
      Fmt.pf ppf "    mul.wide.u32 %a, %a, %a;" pp_ptx_reg dst pp_operand src1 pp_operand src2

  | AddU64 (dst, src1, src2) ->
      Fmt.pf ppf "    add.u64 %a, %a, %a;" pp_ptx_reg dst pp_operand src1 pp_operand src2

  | SetP (cmp_str, pred_reg, op1, op2) ->
      Fmt.pf ppf "    setp.%s %a, %a, %a;" cmp_str pp_ptx_reg pred_reg pp_operand op1 pp_operand op2

  | Bra (target, pred_opt) ->
      (match pred_opt with
       | None            -> Fmt.pf ppf "    bra %s;" target
       | Some (p, true)  -> Fmt.pf ppf "    @%a bra %s;" pp_ptx_reg p target
       | Some (p, false) -> Fmt.pf ppf "    @!%a bra %s;" pp_ptx_reg p target)

  | Ret ->
      Fmt.pf ppf "    ret;"

  | AtomGlobal (action, ty, dst, addr, val_op) ->
      Fmt.pf ppf "    atom.global.%a%a %a, %a, %a;"
        pp_atomic_action action pp_ptx_type ty pp_ptx_reg dst pp_operand addr pp_operand val_op

(* Formats a single formal parameter in a kernel signature *)
let pp_param ppf (name, ty) =
  Fmt.pf ppf "    .param %a %s" pp_ptx_type ty name

(** Pretty-prints an entry kernel function. *)
let pp_entry_function ppf fn =
  Fmt.pf ppf ".visible .entry %s(\n" fn.name;
  (match fn.params with
   | [] -> ()
   | [single] -> Fmt.pf ppf "%a\n" pp_param single
   | head :: tail ->
       Fmt.pf ppf "%a" pp_param head;
       List.iter (fun p -> Fmt.pf ppf ",\n%a" pp_param p) tail;
       Fmt.pf ppf "\n");
  Fmt.pf ppf ")\n{\n";
  List.iter (fun stmt -> Fmt.pf ppf "%a\n" pp_statement stmt) fn.body;
  Fmt.pf ppf "}\n"

(** Pretty-prints an entire PTX compilation unit including directives and entry functions. *)
let pp_program ppf prog =
  List.iter (fun d -> Fmt.pf ppf "%a\n" pp_directive d) prog.directives;
  if prog.directives <> [] then Fmt.pf ppf "\n";
  List.iter (fun fn -> Fmt.pf ppf "%a\n" pp_entry_function fn) prog.functions

(** Converts a PTX statement to string. *)
let statement_to_string stmt =
  Fmt.str "%a" pp_statement stmt

(** Converts an entry function to string. *)
let entry_function_to_string fn =
  Fmt.str "%a" pp_entry_function fn

(** Converts a complete PTX program AST into PTX assembly source text. *)
let to_string prog =
  Fmt.str "%a" pp_program prog
