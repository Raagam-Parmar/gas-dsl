open Types
open Ast
open Helper

module PA = Ptx.Ast

(* ── instruction lowering ────────────────────────────────────────────── *)

let lower_instr (cas_counter : int ref) (instr : instruction) : PA.statement list =
  match instr with
  | GetTid      dst -> [ PA.Mov (P.U32, lower_reg dst, PA.SpecialReg "%tid.x") ]
  | GetThreadId dst -> [ PA.Mov (P.U32, lower_reg dst, PA.SpecialReg "%tid.x") ]
  | GetNtid     dst -> [ PA.Mov (P.U32, lower_reg dst, PA.SpecialReg "%ntid.x") ]
  | GetCtaid    dst -> [ PA.Mov (P.U32, lower_reg dst, PA.SpecialReg "%ctaid.x") ]

  | LoadParam (dst, name) -> [ PA.LdParam (lower_type dst.ty, lower_reg dst, name) ]
  | CvtaToGlobal (dst, src) -> [ PA.CvtaToGlobal (lower_reg dst, lower_reg src) ]

  | LoadGlobal (dst, addr, _ty) -> [ PA.LdGlobal (lower_type dst.ty, lower_reg dst, PA.Mem (lower_reg addr, None), None) ]

  | StoreGlobal (addr, src, _ty) -> [ PA.StGlobal (lower_type src.ty, PA.Mem (lower_reg addr, None), lower_reg src, None) ]

  | MulWideU32 (dst, src, stride) -> [ PA.MulWideU32 (lower_reg dst, PA.Reg (lower_reg src), PA.ImmInt (Int64.of_int stride)) ]

  | AddU64 (dst, base, offset) -> [ PA.AddU64 (lower_reg dst, PA.Reg (lower_reg base), PA.Reg (lower_reg offset)) ]

  | Arith (op, dst, src1, src2, ty) -> [ lower_arith op (lower_type ty) (lower_reg dst) (lower_operand src1) (lower_operand src2) ]
  | Mad (dst, a, b, c, ty) -> [ PA.Mad (lower_type ty, lower_reg dst, lower_operand a, lower_operand b, lower_operand c) ]

  | Compare (op, dst, src1, src2, ty) -> [ PA.SetP (cmp_str op ty, lower_reg dst, lower_operand src1, lower_operand src2) ]

  | Atomic (op, dst, addr, value, ty) -> [ PA.AtomGlobal (lower_atomic_action op, lower_type ty, lower_reg dst, PA.Mem (lower_reg addr, None), lower_operand value) ]
  
  | CasLoop { target_addr; new_val; cur_val; pred_reg; is_min } ->
      let id = !cas_counter in
      incr cas_counter;
      expand_cas_loop ~target_addr ~new_val ~cur_val ~pred_reg ~is_min id

  | Label  name            -> [ PA.LabelDef name ]
  | Branch name            -> [ PA.Bra (name, None) ]
  | BranchIf (pred_r, lbl) -> [ PA.Bra (lbl, Some (lower_reg pred_r, true)) ]
  | Return                  -> [ PA.Ret ]

(* ── kernel & program translation ───────────────────────────────────── *)

let translate_kernel (k : kernel) : PA.entry_function =
  let cas_counter = ref 0 in
  let decls  = reg_decls (count_regs k.body) in
  let instrs = List.concat_map (lower_instr cas_counter) k.body in
  {
    PA.name   = k.name;
    PA.params = List.map (fun (n, ty) -> (n, lower_type ty)) k.params;
    PA.body   = decls @ instrs;
  }

let translate ?(target = "sm_75") (kernels : kernel list) : PA.program =
  {
    PA.directives = [ PA.Version 7.0; PA.Target target; PA.AddressSize 64 ];
    PA.functions  = List.map translate_kernel kernels;
  }
