(** {1 LowerIR Types}

    This module defines the type system and machine-level representations for
    the Lower IR. Lower IR operates on unbounded typed virtual registers in
    Static Single Assignment (SSA) style, modeling memory spaces, pointer strides,
    and GPU hardware atomic instructions prior to PTX string emission. *)

(** Low-level data types matching GPU hardware register and memory widths. *)
type data_type =
  | U32  (** Unsigned 32-bit integer (.u32) *)
  | U64  (** Unsigned 64-bit integer (.u64), used for memory addresses on A30 *)
  | S32  (** Signed 32-bit integer (.s32) *)
  | S64  (** Signed 64-bit integer (.s64) *)
  | F32  (** Single-precision 32-bit float (.f32) *)
  | B32  (** Untyped 32-bit bitfield (.b32), used for bitwise ops and CAS *)
  | Pred (** 1-bit boolean predicate register (.pred), used for condition flags and branching *)

(** Unique numeric identifier for a virtual register. *)
type reg_id = int

(** A typed virtual register in SSA form. *)
type reg = {
  id : reg_id;   (** Monotonically assigned register index (e.g., 0, 1, 2) *)
  ty : data_type;(** Data type and bit-width of this register *)
}

(** Instruction operands: either a virtual register or an immediate literal. *)
type operand =
  | Reg    of reg    (** Virtual register operand *)
  | ImmU32 of int32  (** 32-bit unsigned integer immediate *)
  | ImmU64 of int64  (** 64-bit unsigned integer immediate (e.g., byte strides) *)
  | ImmF32 of float  (** 32-bit floating-point immediate *)

(** GPU memory address spaces. *)
type space =
  | Global (** Global device memory (HBM2 on NVIDIA A30) *)
  | Param  (** Kernel parameter memory space passed from host *)
  | Shared (** Fast on-chip scratchpad shared memory within a thread block *)
  | Local  (** Thread-private local memory / stack space *)

(** Hardware-level atomic operations. *)
type atomic_op =
  | AtomAdd (** Hardware atomic addition (atom.add) *)
  | AtomMin (** Hardware atomic minimum on integers (atom.min) *)
  | AtomMax (** Hardware atomic maximum on integers (atom.max) *)
  | AtomCas (** Hardware Compare-And-Swap (atom.cas.b32), used for float min/max loops *)
