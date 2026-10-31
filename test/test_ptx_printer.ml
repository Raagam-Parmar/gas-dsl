(** {1 PTX Pretty-Printer & Assembler Tests}

    This test suite verifies that the PTX pretty printer serializes
    PTX AST programs into syntactically valid NVIDIA PTX assembly text,
    and validates the emitted text against the NVIDIA [ptxas] assembler
    targeting Compute Capability [sm_80]. *)

open Gas_dsl

(** Tests pretty-printing a representative PTX program and validates with ptxas. *)
let test_ptx_printer_and_ptxas () =
  let open Ptx.Types in
  let open Ptx.Ast in
  (* Construct a complete PTX kernel program with 64-bit wide pointer math and atomics *)
  let prog = {
    directives = [
      Version 7.0;
      Target "sm_80";
      AddressSize 64;
    ];
    functions = [{
      name = "test_reduce_min";
      params = [
        ("num_edges", U32);
        ("dist_ptr", U64);
        ("edge_val", U32);
      ];
      body = [
        RegDecl (U32, 4);
        RegDecl (U64, 3);
        RegDecl (Pred, 1);
        LdParam (U32, R 0, "num_edges");
        LdParam (U64, RD 0, "dist_ptr");
        LdParam (U32, R 1, "edge_val");
        MulWideU32 (RD 1, Reg (R 1), ImmInt 4L);
        AddU64 (RD 2, Reg (RD 0), Reg (RD 1));
        AtomGlobal (AtomMin, U32, R 2, Mem (RD 2, None), Reg (R 1));
        Ret;
      ];
    }];
  } in
  let ptx_code = Ptx.Printer.to_string prog in
  assert (String.length ptx_code > 0);
  print_endline "  [OK] PTX pretty-printed successfully. Emitted text:";
  print_endline "--------------------------------------------------";
  print_string ptx_code;
  print_endline "--------------------------------------------------";

  (* Assemble emitted text with NVIDIA ptxas if available on system *)
  let ptxas_bin = "/usr/local/cuda-13.3/bin/ptxas" in
  if Sys.file_exists ptxas_bin then begin
    let (tmp_file, tmp_out) = Filename.open_temp_file "kernel_" ".ptx" in
    output_string tmp_out ptx_code;
    close_out tmp_out;
    let cmd = Printf.sprintf "%s -arch=sm_80 -m64 %s -o /dev/null 2>&1" ptxas_bin tmp_file in
    let exit_code = Sys.command cmd in
    Sys.remove tmp_file;
    if exit_code <> 0 then
      failwith (Printf.sprintf "ptxas validation failed with exit code %d" exit_code);
    print_endline "  [OK] ptxas -arch=sm_80 assembled generated PTX with 0 errors!"
  end else
    print_endline "  [SKIP] ptxas not found at /usr/local/cuda-13.3/bin/ptxas"

let () =
  print_endline "=== Running PTX Pretty-Printer Tests ===";
  test_ptx_printer_and_ptxas ();
  print_endline "=== All PTX Pretty-Printer Tests Passed! ===\n"
