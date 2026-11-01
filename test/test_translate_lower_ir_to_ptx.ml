open Gas_dsl

(* tiny helpers *)

let reg id ty : LowerIR.Types.reg = { LowerIR.Types.id; ty }

let u32 id = reg id LowerIR.Types.U32
let u64 id = reg id LowerIR.Types.U64
let f32 id = reg id LowerIR.Types.F32
let pred id = reg id LowerIR.Types.Pred

(* check helpers *)

let assert_contains haystack needle =
  if not (let re = Str.regexp_string needle in
          try ignore (Str.search_forward re haystack 0); true
          with Not_found -> false) then begin
    Printf.printf "  [FAIL] expected substring not found:\n";
    Printf.printf "         expected: %s\n" needle;
    Printf.printf "         in:\n%s\n" haystack;
    exit 1
  end

(* Fall back to plain String.sub search if Str is unavailable *)
let contains_substr haystack needle =
  let hl = String.length haystack and nl = String.length needle in
  if nl = 0 then true
  else
    let rec loop i =
      if i + nl > hl then false
      else if String.sub haystack i nl = needle then true
      else loop (i + 1)
    in
    loop 0

let check_contains lbl haystack needle =
  if not (contains_substr haystack needle) then begin
    Printf.printf "  [FAIL] %s\n" lbl;
    Printf.printf "         missing: %S\n" needle;
    Printf.printf "         got:\n%s\n" haystack;
    exit 1
  end else
    Printf.printf "  [OK] %s\n" lbl

let _ = assert_contains   (* silence unused-value warning *)

(* test 1: header directives *)

let test_header_directives () =
  let open LowerIR in
  let k : Ast.kernel = {
    Ast.name   = "dummy";
    Ast.params = [];
    Ast.body   = [ Ast.Return ];
  } in
  let prog   = Translate.translate [k] in
  let ptx    = Ptx.Printer.to_string prog in
  check_contains "version directive"      ptx ".version 7.0";
  check_contains "target sm_75"           ptx ".target sm_75";
  check_contains "address_size 64"        ptx ".address_size 64"

(* test 2: kernel name and params *)

let test_kernel_name_and_params () =
  let open LowerIR in
  let k : Ast.kernel = {
    Ast.name   = "sssp_scatter";
    Ast.params = [
      ("num_edges", Types.U32);
      ("dist_ptr",  Types.U64);
    ];
    Ast.body   = [ Ast.Return ];
  } in
  let prog = Translate.translate [k] in
  let ptx  = Ptx.Printer.to_string prog in
  check_contains "kernel name in PTX"      ptx "sssp_scatter";
  check_contains "u32 param in PTX"        ptx ".u32 num_edges";
  check_contains "u64 param in PTX"        ptx ".u64 dist_ptr"

(* test 3: LoadParam + MulWideU32 + AddU64 *)

let test_address_arithmetic () =
  let open LowerIR in
  let k : Ast.kernel = {
    Ast.name   = "addr_test";
    Ast.params = [ ("ptr", Types.U64); ("n", Types.U32) ];
    Ast.body   = [
      Ast.LoadParam (u64 0, "ptr");
      Ast.LoadParam (u32 0, "n");
      Ast.MulWideU32 (u64 1, u32 0, 4);
      Ast.AddU64     (u64 2, u64 0, u64 1);
      Ast.Return;
    ];
  } in
  let prog = Translate.translate [k] in
  let ptx  = Ptx.Printer.to_string prog in
  check_contains "ld.param.u64"     ptx "ld.param.u64";
  check_contains "ld.param.u32"     ptx "ld.param.u32";
  check_contains "mul.wide.u32"     ptx "mul.wide.u32";
  check_contains "add.u64"          ptx "add.u64"

(* test 4: Atomic integer min (SSSP) *)

let test_atomic_min_u32 () =
  let open LowerIR in
  let k : Ast.kernel = {
    Ast.name   = "sssp_atomic";
    Ast.params = [ ("dist_ptr", Types.U64) ];
    Ast.body   = [
      Ast.LoadParam (u64 0, "dist_ptr");
      Ast.Atomic (Types.AtomMin, u32 1, u64 0, Types.ImmU32 999l, Types.U32);
      Ast.Return;
    ];
  } in
  let prog = Translate.translate [k] in
  let ptx  = Ptx.Printer.to_string prog in
  check_contains "atom.global.min.u32" ptx "atom.global.min.u32"

(* test 5: Compare + BranchIf (boundary guard) *)

let test_boundary_guard () =
  let open LowerIR in
  let k : Ast.kernel = {
    Ast.name   = "guard_test";
    Ast.params = [ ("n", Types.U32) ];
    Ast.body   = [
      Ast.LoadParam (u32 0, "n");
      Ast.GetTid    (u32 1);
      Ast.Compare   (Ast.Ge, pred 0, Types.Reg (u32 1), Types.Reg (u32 0), Types.U32);
      Ast.BranchIf  (pred 0, "$EXIT");
      Ast.Label     "$EXIT";
      Ast.Return;
    ];
  } in
  let prog = Translate.translate [k] in
  let ptx  = Ptx.Printer.to_string prog in
  check_contains "setp.ge.u32"    ptx "setp.ge.u32";
  check_contains "@%p0 branch"    ptx "@%p0 bra $EXIT";
  check_contains "$EXIT label"    ptx "$EXIT:"

(* test 6: CAS loop expansion (SSSP float min) *)

let test_cas_loop_expansion () =
  let open LowerIR in
  let k : Ast.kernel = {
    Ast.name   = "cas_test";
    Ast.params = [ ("dist_ptr", Types.U64) ];
    Ast.body   = [
      Ast.LoadParam (u64 0, "dist_ptr");
      Ast.CasLoop {
        target_addr = u64 0;
        new_val     = f32 1;
        cur_val     = f32 0;
        pred_reg    = pred 0;
        is_min      = true;
      };
      Ast.Return;
    ];
  } in
  let prog = Translate.translate [k] in
  let ptx  = Ptx.Printer.to_string prog in
  check_contains "CAS loop start label"  ptx "$CAS_LOOP_0:";
  check_contains "CAS end label"         ptx "$CAS_END_0:";
  check_contains "ld.global.f32 in CAS"  ptx "ld.global.f32";
  check_contains "setp skip in CAS"      ptx "setp.ge.f32";
  check_contains "atom.global.cas.b32"   ptx "atom.global.cas.b32"

(* test 7: multiple kernels in one program *)

let test_multi_kernel () =
  let open LowerIR in
  let k1 : Ast.kernel = {
    Ast.name   = "scatter_kernel";
    Ast.params = [];
    Ast.body   = [ Ast.Return ];
  } in
  let k2 : Ast.kernel = {
    Ast.name   = "apply_kernel";
    Ast.params = [];
    Ast.body   = [ Ast.Return ];
  } in
  let prog = Translate.translate [k1; k2] in
  let ptx  = Ptx.Printer.to_string prog in
  check_contains "scatter_kernel present" ptx "scatter_kernel";
  check_contains "apply_kernel present"   ptx "apply_kernel"

(* test 8: reg declarations emitted *)

let test_reg_declarations () =
  let open LowerIR in
  let k : Ast.kernel = {
    Ast.name   = "regdecl_test";
    Ast.params = [];
    Ast.body   = [
      Ast.GetTid    (u32 0);
      Ast.LoadParam (u64 0, "p");
      Ast.Compare   (Ast.Eq, pred 0, Types.Reg (u32 0), Types.ImmU32 0l, Types.U32);
      Ast.Return;
    ];
  } in
  let prog = Translate.translate [k] in
  let ptx  = Ptx.Printer.to_string prog in
  check_contains ".reg .u32"  ptx ".reg .u32";
  check_contains ".reg .u64"  ptx ".reg .u64";
  check_contains ".reg .pred" ptx ".reg .pred"

(* runner *)

let () =
  print_endline "=== translate LowerIR_to_ptx ===";
  test_header_directives    (); print_endline "  [DONE] header directives";
  test_kernel_name_and_params (); print_endline "  [DONE] kernel name and params";
  test_address_arithmetic   (); print_endline "  [DONE] address arithmetic";
  test_atomic_min_u32       (); print_endline "  [DONE] atomic min u32";
  test_boundary_guard       (); print_endline "  [DONE] boundary guard";
  test_cas_loop_expansion   (); print_endline "  [DONE] CAS loop expansion";
  test_multi_kernel         (); print_endline "  [DONE] multi-kernel program";
  test_reg_declarations     (); print_endline "  [DONE] reg declarations";
  print_endline "\n === All translate LowerIR_to_ptx tests passed! === \n"
