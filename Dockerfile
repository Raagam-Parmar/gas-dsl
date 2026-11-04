FROM ocaml/opam:ubuntu-24.04-ocaml-5.2
WORKDIR /home/opam/src
COPY --chown=opam:opam *.opam dune-project ./
RUN opam install . --deps-only --with-test --with-doc --yes
