(** {1 PTX Types}

    This module defines concrete syntax types for the NVIDIA Parallel Thread
    Execution (PTX) ISA, specifically targeting Compute Capability [sm_75]
    (NVIDIA Turing GPU architecture).

    It models PTX state spaces, register banks (%r, %rd, %f, %p), cache
    behavior modifiers, and atomic operation types. *)

(** PTX fundamental data types. *)
type ptx_type =
  | U32  (** .u32 - Unsigned 32-bit integer *)
  | U64  (** .u64 - Unsigned 64-bit integer (pointers on 64-bit targets) *)
  | S32  (** .s32 - Signed 32-bit integer *)
  | S64  (** .s64 - Signed 64-bit integer *)
  | F32  (** .f32 - IEEE 754 single-precision 32-bit floating point *)
  | B32  (** .b32 - Untyped 32-bit bitfield *)
  | B64  (** .b64 - Untyped 64-bit bitfield *)
  | Pred (** .pred - 1-bit boolean predicate flag *)

(** PTX memory address spaces. *)
type state_space =
  | Global (** .global - Global device memory (HBM2 on A30) *)
  | Param  (** .param - Kernel parameter space *)
  | Shared (** .shared - Thread block shared memory *)
  | Reg    (** .reg - Register space *)

(** Hardware virtual register banks in PTX assembly. *)
type ptx_reg =
  | R  of int (** %r<N>  - 32-bit integer register (.u32, .s32, .b32) *)
  | RD of int (** %rd<N> - 64-bit integer/pointer register (.u64, .s64, .b64) *)
  | F  of int (** %f<N>  - 32-bit floating point register (.f32) *)
  | P  of int (** %p<N>  - 1-bit boolean predicate register (.pred) *)

(** Global memory caching modifiers for load/store instructions. *)
type cache_modifier =
  | CA (** .ca - Cache at all levels (L1 and L2) *)
  | CG (** .cg - Cache at global level (L2 only, bypass L1) *)
  | CS (** .cs - Cache streaming (evict first) *)

(** Hardware atomic actions supported on PTX global memory. *)
type atomic_action =
  | AtomAdd (** add - Atomic addition (supports .u32, .s32, and .f32) *)
  | AtomMin (** min - Atomic minimum (supports .u32, .s32) *)
  | AtomMax (** max - Atomic maximum (supports .u32, .s32) *)
  | AtomCas (** cas - Atomic Compare-And-Swap (supports .b32) *)
