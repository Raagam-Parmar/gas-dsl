(** {1 Vector Addition: LowerIR to PTX Pretty-Printer}

    Constructs a complete vector addition kernel (c[i] = a[i] + b[i]) in LowerIR,
    translates it to PTX AST, and prints the generated NVIDIA PTX assembly. *)

open Gas_dsl

(* ── tiny helpers ─────────────────────────────────────────────────────── *)

let reg id ty : LowerIR.Types.reg = { LowerIR.Types.id; ty }

let u32 id = reg id LowerIR.Types.U32
let u64 id = reg id LowerIR.Types.U64
let f32 id = reg id LowerIR.Types.F32
let pred id = reg id LowerIR.Types.Pred

(* ── vecadd LowerIR to PTX ────────────────────────────────────────────── *)

let print_vecadd_ptx () =
  let open LowerIR in
  (* c[i] = a[i] + b[i] *)
  let k : Ast.kernel = {
    Ast.name   = "vecAdd";
    Ast.params = [
      ("a", Types.U64);
      ("b", Types.U64);
      ("c", Types.U64);
      ("n", Types.U32);
    ];
    Ast.body   = [
      (* Load kernel parameters from .param space *)
      Ast.LoadParam (u64 1, "a");
      Ast.LoadParam (u64 2, "b");
      Ast.LoadParam (u64 3, "c");
      Ast.LoadParam (u32 1, "n");

      (* Global thread ID index: i = blockIdx.x * blockDim.x + threadIdx.x *)
      Ast.GetCtaid (u32 2);
      Ast.GetNtid (u32 3);
      Ast.GetThreadId (u32 4);
      Ast.Mad (u32 5, Types.Reg (u32 2), Types.Reg (u32 3), Types.Reg (u32 4), Types.S32);

      (* Boundary guard: if (i >= n) return; *)
      Ast.Compare (Ast.Ge, pred 1, Types.Reg (u32 5), Types.Reg (u32 1), Types.U32);
      Ast.BranchIf (pred 1, "DONE");

      (* Convert parameter generic pointers to global state space pointers *)
      Ast.CvtaToGlobal (u64 4, u64 1);
      Ast.CvtaToGlobal (u64 5, u64 2);
      Ast.CvtaToGlobal (u64 6, u64 3);

      (* 64-bit byte offset = i * sizeof(float32) *)
      Ast.MulWideU32 (u64 7, u32 5, 4);

      (* Base address + byte offset *)
      Ast.AddU64 (u64 8,  u64 4, u64 7);
      Ast.AddU64 (u64 9,  u64 5, u64 7);
      Ast.AddU64 (u64 10, u64 6, u64 7);

      (* Load elements: f1 = a[i], f2 = b[i] *)
      Ast.LoadGlobal (f32 1, u64 8, Types.F32);
      Ast.LoadGlobal (f32 2, u64 9, Types.F32);

      (* Add elements: f3 = f1 + f2 *)
      Ast.Arith (Ast.Add, f32 3, Types.Reg (f32 1), Types.Reg (f32 2), Types.F32);

      (* Store element: c[i] = f3 *)
      Ast.StoreGlobal (u64 10, f32 3, Types.F32);

      (* Done label & return *)
      Ast.Label "DONE";
      Ast.Return;
    ];
  } in

  let prog = Translate.translate [k] in
  let ptx  = Ptx.Printer.to_string prog in
  print_string ptx

(* ── runner ───────────────────────────────────────────────────────────── *)

let () =
  print_vecadd_ptx ()
