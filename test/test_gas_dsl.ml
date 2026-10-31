(** Unit tests verifying that AST and Type definitions in GraphIR,
    LowerIR, and PTX can be instantiated cleanly without type errors. *)

open Gas_dsl

let test_graph_ir () =
  let open GraphIR.Types in
  let open GraphIR.Ast in
  let dist_field = { name = "dist"; ty = TyF32 } in
  let weight_field = { name = "weight"; ty = TyF32 } in
  let gather = BinOp (Add, LoadField (dist_field, Src), LoadField (weight_field, EdgeAttr)) in
  let scatter = {
    gather_expr = gather;
    target_field = dist_field;
    reduction = Min;
  } in
  let algo = {
    name = "sssp";
    vertex_state = [dist_field];
    edge_state = [weight_field];
    scatter = scatter;
    apply = None;
  } in
  assert (algo.name = "sssp");
  print_endline "  [OK] GraphIR AST instantiated successfully."

let test_lower_ir () =
  let open LowerIR.Types in
  let open LowerIR.Ast in
  let tid_reg = { id = 0; ty = U32 } in
  let num_edges_reg = { id = 1; ty = U32 } in
  let pred_reg = { id = 2; ty = Pred } in
  let addr_reg = { id = 3; ty = U64 } in
  let val_reg = { id = 4; ty = F32 } in
  let instructions = [
    GetTid tid_reg;
    LoadParam (num_edges_reg, "num_edges");
    Compare (Ge, pred_reg, Reg tid_reg, Reg num_edges_reg, U32);
    BranchIf (pred_reg, "EXIT");
    CasLoop {
      target_addr = addr_reg;
      new_val = val_reg;
      cur_val = { id = 5; ty = F32 };
      pred_reg = { id = 6; ty = Pred };
      is_min = true;
    };
    Label "EXIT";
    Return;
  ] in
  let k = {
    name = "sssp_scatter";
    params = [("num_edges", U32); ("dist", U64)];
    body = instructions;
  } in
  assert (List.length k.body = 7);
  print_endline "  [OK] LowerIR AST instantiated successfully."

let test_ptx_ast () =
  let open Ptx.Types in
  let open Ptx.Ast in
  let prog = {
    directives = [
      Version 7.0;
      Target "sm_80";
      AddressSize 64;
    ];
    functions = [{
      name = "sssp_kernel";
      params = [("num_edges", U32); ("dist_ptr", U64)];
      body = [
        RegDecl (U32, 2);
        RegDecl (U64, 2);
        RegDecl (Pred, 1);
        Ret;
      ];
    }];
  } in
  assert (List.length prog.directives = 3);
  print_endline "  [OK] PTX AST instantiated successfully."

let () =
  print_endline "Running GAS-DSL IR syntax and type validation tests...";
  test_graph_ir ();
  test_lower_ir ();
  test_ptx_ast ();
  print_endline "All IR grammar and type tests passed successfully!"
