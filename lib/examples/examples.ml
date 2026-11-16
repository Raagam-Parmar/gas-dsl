let sssp_u32 =
  let open Edsl in
  let dist   = Vertex.u32 "dist" in
  let weight = Edge.u32 "weight" in
  algorithm ~name:"sssp_u32"
    ~scatter:(scatter ~target:dist ~reduce:Min (fun ~src ~dst:_ ~edge ->
      src.get dist +: edge.get weight))
    ()

let sssp_f32 =
  let open Edsl in
  let dist   = Vertex.f32 "dist" in
  let weight = Edge.f32 "weight" in
  algorithm ~name:"sssp_f32"
    ~scatter:(scatter ~target:dist ~reduce:Min (fun ~src ~dst:_ ~edge ->
      src.get dist +: edge.get weight))
    ()

let pagerank =
  let open Edsl in
  let rank       = Vertex.f32 "rank" in
  let rank_accum = Vertex.f32 "rank_accum" in
  let out_deg    = Vertex.f32 "out_deg" in
  algorithm ~name:"pagerank"
    ~scatter:(scatter ~target:rank_accum ~reduce:Sum (fun ~src ~dst:_ ~edge:_ ->
      src.get rank /: src.get out_deg))
    ~apply:(apply ~dest:rank (fun ~self ->
      float 0.15 +: float 0.85 *: self.get rank_accum))
    ()

let bfs =
  let open Edsl in
  let depth = Vertex.u32 "depth" in
  algorithm ~name:"bfs"
    ~scatter:(scatter ~target:depth ~reduce:Min (fun ~src ~dst:_ ~edge:_ ->
      src.get depth +: int 1))
    ()

let wcc =
  let open Edsl in
  let comp = Vertex.u32 "comp" in
  algorithm ~name:"wcc"
    ~scatter:(scatter ~target:comp ~reduce:Min (fun ~src ~dst:_ ~edge:_ ->
      src.get comp))
    ()

let in_degree =
  let open Edsl in
  let deg = Vertex.u32 "deg" in
  algorithm ~name:"in_degree"
    ~scatter:(scatter ~target:deg ~reduce:Sum (fun ~src:_ ~dst:_ ~edge:_ ->
      int 1))
    ()

let max_influence =
  let open Edsl in
  let score  = Vertex.f32 "score" in
  let weight = Edge.f32 "weight" in
  algorithm ~name:"max_influence"
    ~scatter:(scatter ~target:score ~reduce:Max (fun ~src ~dst:_ ~edge ->
      src.get score *: edge.get weight))
    ()

let all = [
  ("sssp_u32", sssp_u32);
  ("sssp_f32", sssp_f32);
  ("pagerank", pagerank);
  ("bfs", bfs);
  ("wcc", wcc);
  ("in_degree", in_degree);
  ("max_influence", max_influence);
]
