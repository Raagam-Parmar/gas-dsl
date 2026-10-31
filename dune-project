(lang dune 3.21)

(name gas-dsl)

(generate_opam_files true)

(source
 (github username/reponame))

(authors "Author Name <author@example.com>")

(maintainers "Maintainer Name <maintainer@example.com>")

(license LICENSE)

(documentation https://url/to/documentation)

(package
 (name gas-dsl)
 (synopsis "GAS-DSL: Gather-Apply-Scatter EDSL for GPU Graph Analytics")
 (description "An OCaml-based DSL compiler that translates GAS graph algorithms to NVIDIA PTX assembly for the A30 GPU cluster.")
 (depends ocaml fmt)
 (tags
  ("gpu" "cuda" "ptx" "dsl" "graph")))
