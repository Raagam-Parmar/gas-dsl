open Gas_dsl

(* Helper: substring search *)
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

let check_not_contains lbl haystack needle =
  if contains_substr haystack needle then begin
    Printf.printf "  [FAIL] %s\n" lbl;
    Printf.printf "         unexpected substring found: %S\n" needle;
    exit 1
  end else
    Printf.printf "  [OK] %s\n" lbl

(* Emit a GraphIR algorithm -> lower to LowerIR kernels -> print to PTX string *)
let compile algo =
  let kernels = GraphIR.Translate.translate algo in
  let prog    = LowerIR.Translate.translate kernels in
  Ptx.Printer.to_string prog

(* Shorthand field constructors *)
let field_f32 name : GraphIR.Types.field = { GraphIR.Types.name; ty = GraphIR.Types.TyF32 }
let field_u32 name : GraphIR.Types.field = { GraphIR.Types.name; ty = GraphIR.Types.TyU32 }

(* ----- test 1: SSSP scatter kernel is emitted ----- *)

(* SSSP: dist[v] = min(dist[u] + weight[e])
   Uses integer (u32) distance -> should use atom.global.min.u32.
   No apply stage. *)
let test_sssp_scatter_kernel () =
  let algo : GraphIR.Ast.algorithm = {
    GraphIR.Ast.name         = "sssp";
    GraphIR.Ast.vertex_state = [ field_u32 "dist" ];
    GraphIR.Ast.edge_state   = [ field_u32 "weight" ];
    GraphIR.Ast.scatter      = {
      GraphIR.Ast.gather_expr  =
        GraphIR.Ast.BinOp (
          GraphIR.Ast.Add,
          GraphIR.Ast.LoadField (field_u32 "dist",   GraphIR.Types.Src),
          GraphIR.Ast.LoadField (field_u32 "weight", GraphIR.Types.EdgeAttr)
        );
      GraphIR.Ast.target_field = field_u32 "dist";
      GraphIR.Ast.reduction    = GraphIR.Types.Min;
    };
    GraphIR.Ast.apply = None;
  } in
  let ptx = compile algo in
  (* kernel name *)
  check_contains "sssp_scatter kernel name" ptx "sssp_scatter";
  (* standard PTX header *)
  check_contains "PTX version directive"    ptx ".version 7.0";
  check_contains "PTX target sm_75"         ptx ".target sm_75";
  check_contains "PTX address_size 64"      ptx ".address_size 64";
  (* address arithmetic for SoA indexing *)
  check_contains "mul.wide.u32 for offset"  ptx "mul.wide.u32";
  check_contains "add.u64 for pointer"      ptx "add.u64";
  (* u32 min -> native atomic *)
  check_contains "atom.global.min.u32"      ptx "atom.global.min.u32";
  (* no apply kernel should appear *)
  check_not_contains "no sssp_apply"        ptx "sssp_apply"

(* ----- test 2: SSSP with float weights -> CAS loop ----- *)

(* SSSP with f32 distances: PTX has no atom.min.f32,
   so the lowering must synthesise a CAS retry loop. *)
let test_sssp_float_cas_loop () =
  let algo : GraphIR.Ast.algorithm = {
    GraphIR.Ast.name         = "sssp_f32";
    GraphIR.Ast.vertex_state = [ field_f32 "dist" ];
    GraphIR.Ast.edge_state   = [ field_f32 "weight" ];
    GraphIR.Ast.scatter      = {
      GraphIR.Ast.gather_expr  =
        GraphIR.Ast.BinOp (
          GraphIR.Ast.Add,
          GraphIR.Ast.LoadField (field_f32 "dist",   GraphIR.Types.Src),
          GraphIR.Ast.LoadField (field_f32 "weight", GraphIR.Types.EdgeAttr)
        );
      GraphIR.Ast.target_field = field_f32 "dist";
      GraphIR.Ast.reduction    = GraphIR.Types.Min;
    };
    GraphIR.Ast.apply = None;
  } in
  let ptx = compile algo in
  check_contains "sssp_f32_scatter kernel"   ptx "sssp_f32_scatter";
  (* float min must lower to CAS loop *)
  check_contains "CAS loop start label"      ptx "$CAS_LOOP_0:";
  check_contains "CAS end label"             ptx "$CAS_END_0:";
  check_contains "ld.global.f32 in CAS"      ptx "ld.global.f32";
  check_contains "setp.ge.f32 skip cmp"      ptx "setp.ge.f32";
  check_contains "atom.global.cas.b32"       ptx "atom.global.cas.b32";
  (* must NOT emit native min on float *)
  check_not_contains "no atom.min.f32"       ptx "atom.global.min.f32"

(* ----- test 3: PageRank scatter -> atom.add.f32 ----- *)

(* PageRank: rank_accum[v] += rank[u] / out_deg[u]
   Uses f32 Sum reduction -> native atom.global.add.f32. *)
let test_pagerank_scatter_add () =
  let algo : GraphIR.Ast.algorithm = {
    GraphIR.Ast.name         = "pagerank";
    GraphIR.Ast.vertex_state = [ field_f32 "rank"; field_f32 "rank_accum"; field_u32 "out_deg" ];
    GraphIR.Ast.edge_state   = [];
    GraphIR.Ast.scatter      = {
      GraphIR.Ast.gather_expr  =
        GraphIR.Ast.BinOp (
          GraphIR.Ast.Div,
          GraphIR.Ast.LoadField (field_f32 "rank",    GraphIR.Types.Src),
          GraphIR.Ast.LoadField (field_u32 "out_deg", GraphIR.Types.Src)
        );
      GraphIR.Ast.target_field = field_f32 "rank_accum";
      GraphIR.Ast.reduction    = GraphIR.Types.Sum;
    };
    GraphIR.Ast.apply = None;
  } in
  let ptx = compile algo in
  check_contains "pagerank_scatter kernel"  ptx "pagerank_scatter";
  (* float sum -> native atom.add.f32 *)
  check_contains "atom.global.add.f32"      ptx "atom.global.add.f32";
  check_not_contains "no CAS loop"          ptx "$CAS_LOOP_0:"

(* ----- test 4: apply kernel is emitted when stage is present ----- *)

(* PageRank apply: rank[v] = 0.15 + 0.85 * rank_accum[v]
   Checks that the apply kernel exists and emits a st.global. *)
let test_apply_kernel_emitted () =
  let algo : GraphIR.Ast.algorithm = {
    GraphIR.Ast.name         = "pagerank";
    GraphIR.Ast.vertex_state = [ field_f32 "rank"; field_f32 "rank_accum" ];
    GraphIR.Ast.edge_state   = [];
    GraphIR.Ast.scatter      = {
      GraphIR.Ast.gather_expr  = GraphIR.Ast.LoadField (field_f32 "rank", GraphIR.Types.Src);
      GraphIR.Ast.target_field = field_f32 "rank_accum";
      GraphIR.Ast.reduction    = GraphIR.Types.Sum;
    };
    GraphIR.Ast.apply = Some {
      GraphIR.Ast.update_expr =
        GraphIR.Ast.BinOp (
          GraphIR.Ast.Add,
          GraphIR.Ast.ConstFloat 0.15,
          GraphIR.Ast.BinOp (
            GraphIR.Ast.Mul,
            GraphIR.Ast.ConstFloat 0.85,
            GraphIR.Ast.LoadField (field_f32 "rank_accum", GraphIR.Types.Self)
          )
        );
      GraphIR.Ast.dest_field = field_f32 "rank";
    };
  } in
  let ptx = compile algo in
  (* both kernels must appear *)
  check_contains "pagerank_scatter present" ptx "pagerank_scatter";
  check_contains "pagerank_apply present"   ptx "pagerank_apply";
  (* apply writes back via st.global *)
  check_contains "st.global in apply"       ptx "st.global"

(* ----- test 5: boundary guard in scatter kernel ----- *)

(* Every scatter kernel must guard threads beyond num_edges.
   Check for setp + conditional branch to $SCATTER_EXIT. *)
let test_scatter_boundary_guard () =
  let algo : GraphIR.Ast.algorithm = {
    GraphIR.Ast.name         = "bguard";
    GraphIR.Ast.vertex_state = [ field_u32 "dist" ];
    GraphIR.Ast.edge_state   = [];
    GraphIR.Ast.scatter      = {
      GraphIR.Ast.gather_expr  = GraphIR.Ast.ConstInt 0;
      GraphIR.Ast.target_field = field_u32 "dist";
      GraphIR.Ast.reduction    = GraphIR.Types.Min;
    };
    GraphIR.Ast.apply = None;
  } in
  let ptx = compile algo in
  check_contains "setp.ge.u32 guard"      ptx "setp.ge.u32";
  check_contains "$SCATTER_EXIT label"     ptx "$SCATTER_EXIT:"

(* ----- test 6: boundary guard in apply kernel ----- *)

(* Apply kernel must also guard threads beyond num_vertices. *)
let test_apply_boundary_guard () =
  let algo : GraphIR.Ast.algorithm = {
    GraphIR.Ast.name         = "bguard_apply";
    GraphIR.Ast.vertex_state = [ field_u32 "dist" ];
    GraphIR.Ast.edge_state   = [];
    GraphIR.Ast.scatter      = {
      GraphIR.Ast.gather_expr  = GraphIR.Ast.ConstInt 0;
      GraphIR.Ast.target_field = field_u32 "dist";
      GraphIR.Ast.reduction    = GraphIR.Types.Min;
    };
    GraphIR.Ast.apply = Some {
      GraphIR.Ast.update_expr = GraphIR.Ast.LoadField (field_u32 "dist", GraphIR.Types.Self);
      GraphIR.Ast.dest_field  = field_u32 "dist";
    };
  } in
  let ptx = compile algo in
  check_contains "$APPLY_EXIT label" ptx "$APPLY_EXIT:"

(* ----- test 7: SoA pointer params are present in scatter kernel ----- *)

(* All field buffers plus CSR arrays must appear as .u64 params. *)
let test_scatter_soa_params () =
  let algo : GraphIR.Ast.algorithm = {
    GraphIR.Ast.name         = "soa";
    GraphIR.Ast.vertex_state = [ field_u32 "dist" ];
    GraphIR.Ast.edge_state   = [ field_u32 "weight" ];
    GraphIR.Ast.scatter      = {
      GraphIR.Ast.gather_expr  = GraphIR.Ast.ConstInt 0;
      GraphIR.Ast.target_field = field_u32 "dist";
      GraphIR.Ast.reduction    = GraphIR.Types.Min;
    };
    GraphIR.Ast.apply = None;
  } in
  let ptx = compile algo in
  check_contains "dist_ptr param"         ptx ".u64 dist_ptr";
  check_contains "weight_ptr param"       ptx ".u64 weight_ptr";
  check_contains "src_indices_ptr param"  ptx ".u64 src_indices_ptr";
  check_contains "dst_indices_ptr param"  ptx ".u64 dst_indices_ptr";
  check_contains "num_edges param"        ptx ".u32 num_edges"

(* ----- test 8: constant literal expression ----- *)

(* ConstInt 42 should produce an add instruction initialising the register.
   The immediate value 42 should appear verbatim in the PTX. *)
let test_const_int_literal () =
  let algo : GraphIR.Ast.algorithm = {
    GraphIR.Ast.name         = "const_test";
    GraphIR.Ast.vertex_state = [ field_u32 "dist" ];
    GraphIR.Ast.edge_state   = [];
    GraphIR.Ast.scatter      = {
      GraphIR.Ast.gather_expr  = GraphIR.Ast.ConstInt 42;
      GraphIR.Ast.target_field = field_u32 "dist";
      GraphIR.Ast.reduction    = GraphIR.Types.Min;
    };
    GraphIR.Ast.apply = None;
  } in
  let ptx = compile algo in
  check_contains "literal 42 in PTX" ptx "42"

(* ----- runner ----- *)

let () =
  print_endline "=== GraphIR.Translate tests ===";
  test_sssp_scatter_kernel   (); print_endline "  [DONE] SSSP scatter kernel (u32 min)";
  test_sssp_float_cas_loop   (); print_endline "  [DONE] SSSP float scatter -> CAS loop";
  test_pagerank_scatter_add  (); print_endline "  [DONE] PageRank scatter (f32 add)";
  test_apply_kernel_emitted  (); print_endline "  [DONE] apply kernel emitted";
  test_scatter_boundary_guard (); print_endline "  [DONE] scatter boundary guard";
  test_apply_boundary_guard  (); print_endline "  [DONE] apply boundary guard";
  test_scatter_soa_params    (); print_endline "  [DONE] SoA pointer params in scatter";
  test_const_int_literal     (); print_endline "  [DONE] constant int literal";
  print_endline "\n === All GraphIR.Translate tests passed! === \n"
