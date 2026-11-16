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

let all = [
  ("sssp_u32", sssp_u32);
  ("sssp_f32", sssp_f32);
  ("pagerank", pagerank);
]
