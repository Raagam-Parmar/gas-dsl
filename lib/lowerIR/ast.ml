(** {1 LowerIR AST}

    This module defines the linear instruction sequence AST for Lower IR.
    Instructions operate on virtual registers in Static Single Assignment (SSA) form.
    It captures:
    - Thread index resolution and out-of-bounds boundary guards.
    - 64-bit address computation (scaling 32-bit indices to 64-bit byte offsets on A30).
    - Global memory loads and stores for Structure-of-Arrays (SoA) buffers.
    - Native hardware atomics and Compare-and-Swap (CAS) loops.
    - Predicated branches and control flow. *)

open Types

(** Binary arithmetic and bitwise operations. *)
type binop =
  | Add  (** Addition (+) *)
  | Sub  (** Subtraction (-) *)
  | Mul  (** Multiplication ( * ) *)
  | Div  (** Division (/) *)
  | And  (** Bitwise AND (&) *)
  | Or   (** Bitwise OR (|) *)
  | Xor  (** Bitwise XOR (^) *)
  | Shl  (** Shift left (<<) *)
  | Shr  (** Shift right (>>) *)

(** Comparison operators for generating boolean predicates. *)
type cmp_op =
  | Eq   (** Equal (==) *)
  | Ne   (** Not equal (!=) *)
  | Lt   (** Less than (<) *)
  | Le   (** Less than or equal (<=) *)
  | Gt   (** Greater than (>) *)
  | Ge   (** Greater than or equal (>=) *)

(** Low-level machine instruction set. *)
type instruction =
  (* --- Thread & Block Identification --- *)
  | GetTid       of reg
    (** Compute global 1D thread ID: [reg = ctaid.x * ntid.x + tid.x] *)
  | GetNtid      of reg
    (** Read thread block dimension: [reg = ntid.x] *)
  | GetCtaid     of reg
    (** Read block index: [reg = ctaid.x] *)

  (* --- Parameter & Memory Operations --- *)
  | LoadParam    of reg * string
    (** Load a kernel parameter (e.g., buffer pointer or item count) into a register *)
  | LoadGlobal   of reg * reg * data_type
    (** Load value from global memory: [dst_reg = ld.global [addr_reg]] *)
  | StoreGlobal  of reg * reg * data_type
    (** Store value to global memory: [st.global [addr_reg], src_reg] *)

  (* --- 64-bit Address Arithmetic (Ampere / A30) --- *)
  | MulWideU32   of reg * reg * int
    (** Zero-extend 32-bit index and multiply by element stride: [reg_u64 = reg_u32 * byte_stride] *)
  | AddU64       of reg * reg * reg
    (** Compute 64-bit pointer address: [addr_u64 = base_ptr_u64 + offset_u64] *)

  (* --- Arithmetic and Comparison --- *)
  | Arith        of binop * reg * operand * operand * data_type
    (** Compute binary operation on two operands: [dst = op1 (binop) op2] *)
  | Compare      of cmp_op * reg * operand * operand * data_type
    (** Compare two operands and store boolean flag into a predicate register ([Pred]) *)

  (* --- Atomics & Reduction --- *)
  | Atomic       of atomic_op * reg * reg * operand * data_type
    (** Native hardware atomic operation: [old_val = atom.space.op [addr], val] *)
  | CasLoop      of {
      target_addr : reg;   (** 64-bit global memory address of target vertex state *)
      new_val     : reg;   (** 32-bit float value to be atomically reduced *)
      cur_val     : reg;   (** Temporary register holding current memory value *)
      pred_reg    : reg;   (** Predicate register for CAS retry loop *)
      is_min      : bool;  (** True for float min reduction; false for float max *)
    }
    (** Synthesized Compare-And-Swap loop implementing float min/max via [atom.global.cas.b32] *)

  (* --- Control Flow --- *)
  | Label        of string
    (** Branch target label *)
  | Branch       of string
    (** Unconditional branch: [bra label] *)
  | BranchIf     of reg * string
    (** Predicated branch: branch to label if predicate register is true ([@%p bra label]) *)
  | Return
    (** Kernel exit instruction ([ret]) *)

(** Complete standalone GPU kernel in Lower IR. *)
type kernel = {
  name   : string;                       (** Entry point kernel name (e.g., "sssp_scatter") *)
  params : (string * data_type) list;    (** Typed kernel parameter signatures passed from host *)
  body   : instruction list;             (** Sequential linear SSA instruction stream *)
}
