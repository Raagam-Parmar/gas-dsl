(** {1 PTX AST}

    This module defines the abstract syntax tree for concrete NVIDIA PTX
    assembly code targeting Compute Capability [sm_80].

    It represents top-level assembler directives, register declarations,
    entry functions, and hardware PTX instructions prior to final string serialization. *)

open Types

(** Operands appearing in PTX instructions. *)
type ptx_operand =
  | Reg      of ptx_reg             (** Register operand (e.g., %r0, %rd1, %f2, %p0) *)
  | ImmInt   of int64               (** Integer immediate literal *)
  | ImmFloat of float               (** Floating-point immediate literal *)
  | Mem      of ptx_reg * int option (** Memory address dereference: [%reg] or [%reg + offset] *)
  | ParamRef of string              (** Kernel parameter address dereference: [param_name] *)

(** Module-level PTX assembler directives. *)
type directive =
  | Version     of float (** .version <maj.min> (e.g., .version 7.0) *)
  | Target      of string (** .target <arch> (e.g., .target sm_80) *)
  | AddressSize of int    (** .address_size <bits> (e.g., .address_size 64) *)

(** Individual statements within a PTX kernel body. *)
type statement =
  (* --- Declarations --- *)
  | RegDecl    of ptx_type * int
    (** Register declaration: [.reg .type %name<count>;] *)
  | LabelDef   of string
    (** Branch target label definition: [$label:] *)

  (* --- Data Movement & Memory --- *)
  | Mov        of ptx_type * ptx_reg * ptx_operand
    (** Register move: [mov.type dst, src] *)
  | LdParam    of ptx_type * ptx_reg * string
    (** Load kernel parameter: [ld.param.type dst, [param_name]] *)
  | LdGlobal   of ptx_type * ptx_reg * ptx_operand * cache_modifier option
    (** Load from global memory: [ld.global{.modifier}.type dst, [addr]] *)
  | StGlobal   of ptx_type * ptx_operand * ptx_reg * cache_modifier option
    (** Store to global memory: [st.global{.modifier}.type [addr], src] *)

  (* --- Arithmetic & Address Calculations --- *)
  | Add        of ptx_type * ptx_reg * ptx_operand * ptx_operand
    (** Addition: [add.type dst, src1, src2] *)
  | Sub        of ptx_type * ptx_reg * ptx_operand * ptx_operand
    (** Subtraction: [sub.type dst, src1, src2] *)
  | Mul        of ptx_type * ptx_reg * ptx_operand * ptx_operand
    (** Standard multiplication: [mul.lo.type dst, src1, src2] *)
  | MulWideU32 of ptx_reg * ptx_operand * ptx_operand
    (** Wide 32-to-64 bit multiplication for pointer strides: [mul.wide.u32 dst_u64, src1_u32, src2_u32] *)
  | AddU64     of ptx_reg * ptx_operand * ptx_operand
    (** 64-bit pointer addition: [add.u64 dst_u64, base_u64, offset_u64] *)

  (* --- Predicates & Branching --- *)
  | SetP       of string * ptx_reg * ptx_operand * ptx_operand
    (** Comparison setting predicate: [setp.cmp.type pred_reg, op1, op2] *)
  | Bra        of string * (ptx_reg * bool) option
    (** Branch to label: [bra label] or predicated [@%p bra label] / [@!%p bra label] *)
  | Ret
    (** Kernel termination: [ret;] *)

  (* --- Hardware Atomics --- *)
  | AtomGlobal of atomic_action * ptx_type * ptx_reg * ptx_operand * ptx_operand
    (** Global memory atomic instruction: [atom.global.action.type dst, [addr], val] *)

(** A GPU entry kernel function definition. *)
type entry_function = {
  name   : string;                       (** Entry kernel name (e.g., "sssp_kernel") *)
  params : (string * ptx_type) list;     (** Parameter signature list: [.param .type name] *)
  body   : statement list;               (** Sequence of declarations and instructions *)
}

(** A complete PTX assembly file containing directives and entry functions. *)
type program = {
  directives : directive list;       (** Header directives (.version, .target, .address_size) *)
  functions  : entry_function list;  (** List of GPU kernel entry functions *)
}
