# References and source provenance

## Primary model and methods

- Abramson J, Adler J, Dunger J, et al. Accurate structure prediction of
  biomolecular interactions with AlphaFold 3. *Nature*. 2024;630:493-500.
  doi:10.1038/s41586-024-07487-w.
- Dauparas J, Anishchenko I, Bennett N, et al. Robust deep learning-based
  protein sequence design using ProteinMPNN. *Science*. 2022;378(6615):49-56.
  doi:10.1126/science.add2187.

## Audited source snapshots

- AlphaFold 3: `b2f3d45fbfcacc5183bd5345d15df93571b8437f`
- BindCraft: `b971db42ba6e091afab63ccb30ae02215150a990`
- ColabDesign: `e31a56fe1d9b4de25c8697f3a28b75892941cc72`
- ProteinMPNN: `8907e6671bfbfc92303b5f79c4b5e6ce47cdef57`

Stage scheduling is a semantic reproduction of BindCraft's four-stage flow and
ColabDesign's `design()` ramps. The AF3 soft-query and fixed-geometry
Consistency adapters are project-specific implementations validated against
the stated source snapshots.
