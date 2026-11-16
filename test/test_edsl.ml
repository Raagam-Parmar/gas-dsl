(** {1 EDSL Frontend & End-to-End Tests}

    Verifies the Bonsai-style OCaml EDSL combinators, automatic schema
    collection/deduplication, conflicting-type detection, and full compilation
    from EDSL -> GraphIR -> LowerIR -> PTX -> ptxas. *)

open Gas_dsl

(* Substring search helper *)
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

let assert_contains label haystack needle =
  if not (contains_substr haystack needle) then begin
    Printf.printf "  [FAIL] %s: missing %S\n%s\n" label needle haystack;
    exit 1
  end

(* Compile an EDSL-constructed GraphIR algorithm all the way to PTX text *)
let compile_to_ptx algo =
  let kernels = GraphIR.Translate.translate algo in
  let prog    = LowerIR.Translate.translate kernels in
  Ptx.Printer.to_string prog

(* Validate emitted PTX with NVIDIA ptxas if available *)
let validate_ptxas label ptx_code =
  let ptxas_bin =
    let default_cuda = "/usr/local/cuda-13.3/bin/ptxas" in
    if Sys.file_exists default_cuda then default_cuda
    else if Sys.command "which ptxas > /dev/null 2>&1" = 0 then "ptxas"
    else ""
  in
  if ptxas_bin <> "" then begin
    let (tmp_file, tmp_out) = Filename.open_temp_file "edsl_kernel_" ".ptx" in
    output_string tmp_out ptx_code;
    close_out tmp_out;
    let cmd = Printf.sprintf "%s -arch=sm_75 -m64 %s -o /dev/null 2>&1" ptxas_bin tmp_file in
    let exit_code = Sys.command cmd in
    Sys.remove tmp_file;
    if exit_code <> 0 then
      failwith (Printf.sprintf "%s: ptxas failed with exit code %d\n%s" label exit_code ptx_code);
    Printf.printf "  [OK] %s validated with ptxas (sm_75)\n" label
  end else
    Printf.printf "  [SKIP] ptxas not found for %s\n" label

(** Tests SSSP (u32) expressed in the EDSL. *)
let test_sssp_u32 () =
  let open Edsl in
  let dist   = Vertex.u32 "dist" in
  let weight = Edge.u32 "weight" in
  let algo =
    algorithm ~name:"sssp_u32"
      ~scatter:(scatter ~target:dist ~reduce:Min (fun ~src ~dst:_ ~edge ->
        src.get dist +: edge.get weight))
      ()
  in
  assert (List.length algo.vertex_state = 1);
  assert (List.length algo.edge_state = 1);
  let ptx = compile_to_ptx algo in
  assert_contains "sssp_u32 kernel" ptx "sssp_u32_scatter";
  assert_contains "sssp_u32 atomic" ptx "atom.global.min.u32";
  validate_ptxas "sssp_u32" ptx;
  print_endline "  [OK] EDSL SSSP (u32) passed."

(** Tests SSSP (f32) lowering to a PTX CAS loop. *)
let test_sssp_f32 () =
  let open Edsl in
  let dist   = Vertex.f32 "dist" in
  let weight = Edge.f32 "weight" in
  let algo =
    algorithm ~name:"sssp_f32"
      ~scatter:(scatter ~target:dist ~reduce:Min (fun ~src ~dst:_ ~edge ->
        src.get dist +: edge.get weight))
      ()
  in
  let ptx = compile_to_ptx algo in
  assert_contains "sssp_f32 CAS" ptx "atom.global.cas.b32";
  validate_ptxas "sssp_f32" ptx;
  print_endline "  [OK] EDSL SSSP (f32) passed."

(** Tests PageRank (f32) with both scatter and apply stages. *)
let test_pagerank () =
  let open Edsl in
  let rank       = Vertex.f32 "rank" in
  let rank_accum = Vertex.f32 "rank_accum" in
  let out_deg    = Vertex.f32 "out_deg" in
  let algo =
    algorithm ~name:"pagerank"
      ~scatter:(scatter ~target:rank_accum ~reduce:Sum (fun ~src ~dst:_ ~edge:_ ->
        src.get rank /: src.get out_deg))
      ~apply:(apply ~dest:rank (fun ~self ->
        float 0.15 +: float 0.85 *: self.get rank_accum))
      ()
  in
  (* rank, out_deg, and rank_accum should each appear exactly once *)
  assert (List.length algo.vertex_state = 3);
  assert (List.length algo.edge_state = 0);
  let ptx = compile_to_ptx algo in
  assert_contains "pagerank scatter" ptx "pagerank_scatter";
  assert_contains "pagerank apply"   ptx "pagerank_apply";
  assert_contains "pagerank add.f32" ptx "atom.global.add.f32";
  assert_contains "pagerank st.f32"  ptx "st.global.f32";
  validate_ptxas "pagerank" ptx;
  print_endline "  [OK] EDSL PageRank passed."

(** Verifies that reusing a field name with conflicting types raises Invalid_argument. *)
let test_conflicting_field_type () =
  let open Edsl in
  let dist_u32 = Vertex.u32 "dist" in
  let dist_f32 = Vertex.f32 "dist" in
  let raised =
    try
      let _ =
        algorithm ~name:"bad"
          ~scatter:(scatter ~target:dist_u32 ~reduce:Min (fun ~src ~dst:_ ~edge:_ ->
            src.get dist_u32))
          ~apply:(apply ~dest:dist_f32 (fun ~self -> self.get dist_f32))
          ()
      in
      false
    with Invalid_argument _ -> true
  in
  assert raised;
  print_endline "  [OK] Conflicting field type check passed."

let () =
  print_endline "=== Running EDSL Tests ===";
  test_sssp_u32 ();
  test_sssp_f32 ();
  test_pagerank ();
  test_conflicting_field_type ();
  print_endline "=== All EDSL Tests Passed! ==="
