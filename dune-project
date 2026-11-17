(lang dune 3.21)

(name gas-dsl)

(version 0.1.0)

(generate_opam_files true)

(source
 (github Raagam-Parmar/gas-dsl))

(license MIT)

(authors
 "Raagam Parmar <raagamparmar@gmail.com>"
 "M. Sreenath Karthick <msreenathkarthick2005@gmail.com>"
 "M. Murali Karthick <murali47karthick@gmail.com>")

(maintainers
 "Raagam Parmar <raagamparmar@gmail.com>"
 "M. Sreenath Karthick <msreenathkarthick2005@gmail.com>"
 "M. Murali Karthick <murali47karthick@gmail.com>")

(documentation https://raagam-parmar.github.io/gas-dsl)

(package
 (name gas-dsl)
 (synopsis "GAS-DSL: Gather-Apply-Scatter EDSL and PTX compiler for GPU graph analytics")
 (description
  "An OCaml embedded DSL and compiler that translates Gather-Apply-Scatter (GAS) graph algorithms into NVIDIA PTX assembly for CUDA execution.")
 (depends
  (ocaml (>= 5.0.0))
  (fmt (>= 0.9.0))
  (odoc :with-doc))
 (tags
  ("gpu" "cuda" "ptx" "dsl" "graph")))
