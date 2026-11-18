open Gas_dsl

let compile_and_write name algo =
  let kernels = GraphIR.Translate.translate algo in
  let prog = LowerIR.Translate.translate ~target:"sm_75" kernels in
  let ptx = Ptx.Printer.to_string prog in
  let path = Filename.concat "pipeline/ptx" (name ^ ".ptx") in
  let oc = open_out path in
  output_string oc ptx;
  close_out oc;
  Printf.printf "Wrote %s\n" path

let () =
  List.iter (fun (name, algo) -> compile_and_write name algo) Examples.all
