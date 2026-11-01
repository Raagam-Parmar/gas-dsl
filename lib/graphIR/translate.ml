(** {1 GraphIR -> LowerIR Translation Pass}

    This module is the central lowering pass of the compiler mid-end.
    It takes a fully specified {!GraphIR.Ast.algorithm} and emits a list of
    {!LowerIR.Ast.kernel}s: one edge-parallel Scatter kernel and, when
    present, one vertex-parallel Apply kernel.

    {2 Design Invariants}

    - Every kernel follows the two-phase structure described in AGENTS.md §4:
      Scatter is edge-parallel (1 thread per edge, [tid ∈ \[0, |E|-1\]]);
      Apply is vertex-parallel (1 thread per vertex, [tid ∈ \[0, |V|-1\]]).
    - 64-bit address arithmetic uses {!LowerIR.Ast.MulWideU32} followed by
      {!LowerIR.Ast.AddU64}, as mandated by the A30 address-size rules.
    - Float Min reductions lower to {!LowerIR.Ast.CasLoop} with [is_min = true].
    - Float Max reductions lower to {!LowerIR.Ast.CasLoop} with [is_min = false].
    - All other reductions use {!LowerIR.Ast.Atomic} with a native [atomic_op]. *)

open Ast
open Helper
module LT = LowerIR.Types
module LA = LowerIR.Ast


(** [lower_expr s expr buf_map] recursively lowers a {!GraphIR.Ast.expr}
    into a sequence of LowerIR instructions.

    Returns [(instrs, result_reg)] where [result_reg] holds the computed value.

    [buf_map] maps [(field_name, location_tag)] pairs to the 64-bit base
    pointer register that was loaded from the corresponding kernel parameter.
    The [location_tag] is a string: ["src"], ["dst"], or ["edge"]. *)
let rec lower_expr
    (s        : reg_state)
    (expr     : expr)
    (buf_map  : (string * string, LT.reg) Hashtbl.t)
    (idx_map  : (string, LT.reg) Hashtbl.t)
  : LA.instruction list * LT.reg =
  match expr with

  | ConstFloat f ->
    let dst = fresh s LT.F32 in
    ([ LA.Arith (LA.Add, dst, LT.ImmF32 f, LT.ImmF32 0.0, LT.F32) ], dst)

  | ConstInt i ->
    let dst = fresh s LT.U32 in
    ([ LA.Arith (LA.Add, dst, LT.ImmU32 (Int32.of_int i), LT.ImmU32 0l, LT.U32) ], dst)

  | LoadField (field, loc) ->
    let loc_tag = match loc with
      | Types.Src      -> "src"
      | Types.Dst      -> "dst"
      | Types.EdgeAttr -> "edge"
      | Types.Self     -> "self"
    in
    (* vertex index to use when addressing this field *)
    let idx_reg =
      try Hashtbl.find idx_map loc_tag
      with Not_found ->
        failwith (Printf.sprintf
          "GraphIR.Translate: no index register for location '%s'" loc_tag)
    in
    let base_reg =
      try Hashtbl.find buf_map (field.name, loc_tag)
      with Not_found ->
        failwith (Printf.sprintf
          "GraphIR.Translate: unknown field '%s' at location '%s'"
          field.name loc_tag)
    in
    let ty      = lower_prim_type field.ty in
    let stride  = stride_of_prim  field.ty in
    let off_reg = fresh s LT.U64 in
    let adr_reg = fresh s LT.U64 in
    let dst_reg = fresh s ty in
    let instrs = [
      LA.MulWideU32 (off_reg, idx_reg, stride);
      LA.AddU64     (adr_reg, base_reg, off_reg);
      LA.LoadGlobal (dst_reg, adr_reg, ty);
    ] in
    (instrs, dst_reg)

  | BinOp (op, lhs, rhs) ->
    let (li, lr) = lower_expr s lhs buf_map idx_map in
    let (ri, rr) = lower_expr s rhs buf_map idx_map in
    let (lo_op, result_ty) = lower_binop op lr rr in
    let dst = fresh s result_ty in
    let instr = LA.Arith (lo_op, dst, LT.Reg lr, LT.Reg rr, result_ty) in
    (li @ ri @ [ instr ], dst)

(** [lower_binop op l r] maps a {!GraphIR.Ast.binop} to a
    {!LowerIR.Ast.binop} and the result type, inferring the type from the
    left operand register. *)
and lower_binop (op : binop) (l : LT.reg) (_r : LT.reg)
  : LA.binop * LT.data_type =
  let ty = l.LT.ty in
  let lo = match op with
    | Add   -> LA.Add
    | Sub   -> LA.Sub
    | Mul   -> LA.Mul
    | Div   -> LA.Div
    | MinOp -> LA.Add (* placeholder: min is handled via CAS / native atomic *)
    | MaxOp -> LA.Add (* placeholder: max is handled via CAS / native atomic *)
  in
  (lo, ty)


(** [emit_thread_id s] emits the standard 3-instruction sequence that
    computes the flat linear thread index:

    {v
      tid   = tid.x                  (u32)
      ntid  = ntid.x                 (u32)
      ctaid = ctaid.x                (u32)
      flat  = mad.lo.u32 ctaid, ntid, tid
    v}

    Returns [(instrs, flat_tid_reg)]. *)
let emit_thread_id (s : reg_state) : LA.instruction list * LT.reg =
  let tid   = fresh s LT.U32 in
  let ntid  = fresh s LT.U32 in
  let ctaid = fresh s LT.U32 in
  let flat  = fresh s LT.U32 in
  let instrs = [
    LA.GetTid    tid;
    LA.GetNtid   ntid;
    LA.GetCtaid  ctaid;
    LA.Mad (flat, LT.Reg ctaid, LT.Reg ntid, LT.Reg tid, LT.U32);
  ] in
  (instrs, flat)

(** [emit_bounds_guard s flat_reg bound_param exit_lbl] emits a boundary
    guard that causes out-of-range threads to jump directly to [exit_lbl].

    {v
      p_oob = (flat_reg >= bound_reg)
      @p_oob bra exit_lbl
    v} *)
let emit_bounds_guard(s: reg_state)(flat_reg: LT.reg)(bound_reg: LT.reg)(exit_lbl: string) : LA.instruction list =
  let p_oob = fresh s LT.Pred in
  [
    LA.Compare (LA.Ge, p_oob, LT.Reg flat_reg, LT.Reg bound_reg, LT.U32);
    LA.BranchIf (p_oob, exit_lbl);
  ]


(** [lower_scatter_kernel name vertex_fields edge_fields stage] lowers the
    edge-parallel Scatter stage into a {!LowerIR.Ast.kernel}.

    Thread layout: one thread per edge ([tid ∈ \[0, num_edges)]).

    Steps per thread:
    {ol
      {li Compute flat edge thread ID.}
      {li Boundary guard against [num_edges].}
      {li Load CSR [src_indices] and [dst_indices] to resolve [u] and [v].}
      {li Load kernel parameter pointers for all SoA buffers.}
      {li Evaluate [gather_expr] over the edge [e] / source [u].}
      {li Compute the 64-bit address of [v]'s target field slot.}
      {li Emit native atomic or CAS loop as determined by the reduction type.}
    } *)
let lower_scatter_kernel
    (name          : string)
    (vertex_fields : Types.field list)
    (edge_fields   : Types.field list)
    (stage         : scatter_stage)
  : LA.kernel =
  let s       = make_state () in
  let params  = scatter_params vertex_fields edge_fields in
  let exit_lbl = "$SCATTER_EXIT" in
  let instrs  = ref [] in
  let emit i  = instrs := !instrs @ i in

  (* 1. flat thread-ID (= edge index e) *)
  let (tid_instrs, e_reg) = emit_thread_id s in
  emit tid_instrs;

  (* 2. load num_edges scalar and guard *)
  let ne_reg = fresh s LT.U32 in
  emit [ LA.LoadParam (ne_reg, "num_edges") ];
  emit (emit_bounds_guard s e_reg ne_reg exit_lbl);

  (* 3. load CSR pointer parameters and cvta them to global *)
  let load_ptr name_str =
    let p_raw = fresh s LT.U64 in
    let p_gbl = fresh s LT.U64 in
    emit [
      LA.LoadParam    (p_raw, name_str);
      LA.CvtaToGlobal (p_gbl, p_raw);
    ];
    p_gbl
  in
  let src_base = load_ptr "src_indices_ptr" in
  let dst_base = load_ptr "dst_indices_ptr" in

  (* 4. load u = src[e], v = dst[e] *)
  let load_indexed base idx =
    let off = fresh s LT.U64 in
    let adr = fresh s LT.U64 in
    let val_ = fresh s LT.U32 in
    emit [
      LA.MulWideU32 (off, idx, 4);
      LA.AddU64     (adr, base, off);
      LA.LoadGlobal (val_, adr, LT.U32);
    ];
    val_
  in
  let u_reg = load_indexed src_base e_reg in
  let v_reg = load_indexed dst_base e_reg in

  (* 5. load all vertex and edge SoA buffer pointers *)
  let buf_map : (string * string, LT.reg) Hashtbl.t = Hashtbl.create 16 in
  List.iter (fun (f : Types.field) ->
    let p = load_ptr (f.name ^ "_ptr") in
    Hashtbl.replace buf_map (f.name, "src")  p;
    Hashtbl.replace buf_map (f.name, "dst")  p;
    Hashtbl.replace buf_map (f.name, "self") p;
  ) vertex_fields;
  List.iter (fun (f : Types.field) ->
    let p = load_ptr (f.name ^ "_ptr") in
    Hashtbl.replace buf_map (f.name, "edge") p;
  ) edge_fields;

  (* idx_map: maps location tag to the thread-local index register *)
  let idx_map : (string, LT.reg) Hashtbl.t = Hashtbl.create 4 in
  Hashtbl.replace idx_map "src"  u_reg;
  Hashtbl.replace idx_map "dst"  v_reg;
  Hashtbl.replace idx_map "edge" e_reg;
  Hashtbl.replace idx_map "self" v_reg;

  (* 6. evaluate gather expression *)
  let (expr_instrs, val_reg) =
    lower_expr s stage.gather_expr buf_map idx_map
  in
  emit expr_instrs;

  (* 7. compute address of target_field[v] *)
  let tgt_base =
    try Hashtbl.find buf_map (stage.target_field.name, "dst")
    with Not_found ->
      failwith (Printf.sprintf
        "GraphIR.Translate: scatter target field '%s' not found in vertex_fields"
        stage.target_field.name)
  in
  let tgt_off = fresh s LT.U64 in
  let tgt_adr = fresh s LT.U64 in
  let tgt_stride = stride_of_prim stage.target_field.ty in
  emit [
    LA.MulWideU32 (tgt_off, v_reg, tgt_stride);
    LA.AddU64     (tgt_adr, tgt_base, tgt_off);
  ];

  (* 8. atomic reduction *)
  let strategy = atomic_strategy_of stage.target_field stage.reduction in
  (match strategy with
  | NativeAtomic atom_op ->
    let result_ty = lower_prim_type stage.target_field.ty in
    let old_reg   = fresh s result_ty in
    emit [ LA.Atomic (atom_op, old_reg, tgt_adr, LT.Reg val_reg, result_ty) ]
  | CasMinF32 ->
    let cur_val  = fresh s LT.F32 in
    let pred_reg = fresh s LT.Pred in
    emit [ LA.CasLoop { target_addr = tgt_adr; new_val = val_reg;
                        cur_val; pred_reg; is_min = true } ]
  | CasMaxF32 ->
    let cur_val  = fresh s LT.F32 in
    let pred_reg = fresh s LT.Pred in
    emit [ LA.CasLoop { target_addr = tgt_adr; new_val = val_reg;
                        cur_val; pred_reg; is_min = false } ]);

  (* 9. exit label and return *)
  emit [ LA.Label exit_lbl; LA.Return ];

  { LA.name = name ^ "_scatter"; LA.params; LA.body = !instrs }


(** [lower_apply_kernel name vertex_fields stage] lowers the optional
    vertex-parallel Apply stage into a {!LowerIR.Ast.kernel}.

    Thread layout: one thread per vertex ([tid ∈ \[0, num_vertices)]).

    Steps per thread:
    {ol
      {li Compute flat vertex thread ID.}
      {li Boundary guard against [num_vertices].}
      {li Load all vertex SoA buffer pointers.}
      {li Evaluate [update_expr] for vertex [v].}
      {li Store result into [dest_field\[v\]].}
    } *)
let lower_apply_kernel
    (name          : string)
    (vertex_fields : Types.field list)
    (stage         : apply_stage)
  : LA.kernel =
  let s        = make_state () in
  let params   = apply_params vertex_fields in
  let exit_lbl = "$APPLY_EXIT" in
  let instrs   = ref [] in
  let emit i   = instrs := !instrs @ i in

  (* 1. flat thread-ID (= vertex index v) *)
  let (tid_instrs, v_reg) = emit_thread_id s in
  emit tid_instrs;

  (* 2. load num_vertices scalar and guard *)
  let nv_reg = fresh s LT.U32 in
  emit [ LA.LoadParam (nv_reg, "num_vertices") ];
  emit (emit_bounds_guard s v_reg nv_reg exit_lbl);

  (* 3. load all vertex SoA buffer pointers *)
  let buf_map : (string * string, LT.reg) Hashtbl.t = Hashtbl.create 16 in
  List.iter (fun (f : Types.field) ->
    let p_raw = fresh s LT.U64 in
    let p_gbl = fresh s LT.U64 in
    emit [
      LA.LoadParam    (p_raw, f.name ^ "_ptr");
      LA.CvtaToGlobal (p_gbl, p_raw);
    ];
    Hashtbl.replace buf_map (f.name, "self") p_gbl;
    Hashtbl.replace buf_map (f.name, "src")  p_gbl;
    Hashtbl.replace buf_map (f.name, "dst")  p_gbl;
  ) vertex_fields;

  (* idx_map: v is used for all vertex accesses *)
  let idx_map : (string, LT.reg) Hashtbl.t = Hashtbl.create 4 in
  Hashtbl.replace idx_map "self" v_reg;
  Hashtbl.replace idx_map "src"  v_reg;
  Hashtbl.replace idx_map "dst"  v_reg;

  (* 4. evaluate update expression *)
  let (expr_instrs, result_reg) =
    lower_expr s stage.update_expr buf_map idx_map
  in
  emit expr_instrs;

  (* 5. compute address of dest_field[v] and store *)
  let dst_base =
    try Hashtbl.find buf_map (stage.dest_field.name, "self")
    with Not_found ->
      failwith (Printf.sprintf
        "GraphIR.Translate: apply dest field '%s' not found in vertex_fields"
        stage.dest_field.name)
  in
  let dst_off = fresh s LT.U64 in
  let dst_adr = fresh s LT.U64 in
  let dst_stride = stride_of_prim stage.dest_field.ty in
  emit [
    LA.MulWideU32 (dst_off, v_reg, dst_stride);
    LA.AddU64     (dst_adr, dst_base, dst_off);
    LA.StoreGlobal (dst_adr, result_reg, lower_prim_type stage.dest_field.ty);
  ];

  (* 6. exit label and return *)
  emit [ LA.Label exit_lbl; LA.Return ];

  { LA.name = name ^ "_apply"; LA.params; LA.body = !instrs }


(** [translate algo] lowers a complete {!GraphIR.Ast.algorithm} into a list
    of {!LowerIR.Ast.kernel}s.

    The returned list always contains the Scatter kernel as its first element.
    If [algo.apply] is [Some stage], the Apply kernel is appended as the
    second element. *)
let translate (algo : algorithm) : LA.kernel list =
  let scatter_k =
    lower_scatter_kernel
      algo.name
      algo.vertex_state
      algo.edge_state
      algo.scatter
  in
  let apply_kernels =
    match algo.apply with
    | None       -> []
    | Some stage ->
      [ lower_apply_kernel algo.name algo.vertex_state stage ]
  in
  scatter_k :: apply_kernels
