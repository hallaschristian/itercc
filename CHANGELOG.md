# Changelog

## Unreleased

- `SelectedCI.polarizability`: static and dynamic dipole polarizabilities
  of the selected-CI ground state by linear response in a selected space.
  Each stage grows the space with the determinants coupled to the ground
  state by a dipole component and those coupled through H to the static
  response vectors (normalized), then solves alpha_ij(w) = <0|mu_i
  (H - E0 - w)^-1 Q mu_j|0> + (w -> -w) by GMRES. The full-space limit
  equals the full-CI sum over states (static and dynamic, 1e-8 au); on
  water cc-pVDZ 0.87 million determinants give 3.146, 7.030, 5.267 au
  (CCSDT 3.150, 7.037, 5.272). `SelectedCI.field_solver_factory` runs
  `finite_field_polarizability` in a fixed selected space (the same tensor
  to 2e-6 au).

- Selected CI keeps the Hamiltonian in chunks of 512 rows with their own
  arrays: a build no longer copies the rows into one array afterwards, and
  an extension releases each old chunk once its rows are copied, so a stage
  peaks at about the size of its Hamiltonian instead of two to three times
  it (CuF stages of 16 million determinants and 7 billion nonzeros had run
  out of 480 GB growing to the next stage).

- `SelectedCI` (`itercc.selected_ci`): near-full CI by heat-bath selected CI
  with Epstein-Nesbet PT2, for corrections beyond CCSDTQ, on new compiled
  kernels (`core/src/cpu/selected_ci.cpp`, core ABI 43). Determinants are
  alpha and beta bit strings (up to 256 orbitals); each epsilon grows the
  space from magnitude-sorted double-excitation lists and bounded singles;
  the space's Hamiltonian is kept as its upper triangle and applied in 64
  fixed row stripes; PT2 runs in hash-partitioned passes over sharded hash
  maps (`pt2_batches`), or semistochastically (`pt2_deterministic`: exact
  above a threshold plus a seeded sampled remainder with its standard error).
  `spin_parity` keeps singlets or triplets of M_S = 0 spaces, orbital irreps
  keep one irrep; `extrapolate` and `extrapolate_gap` fit the variational
  energies (gaps) linearly in PT2; density matrices give natural orbitals
  (`natural_orbitals`, `rotate_integrals`). PT2 numerators, density matrices
  and <S^2> accumulate in 128-bit fixed point, so results do not depend on
  the thread count. The full-space limit equals full CI (determinant oracle;
  PySCF on water/6-31G to 1e-13 Eh for singlets and triplets) and PT2 equals
  brute force to 1e-16. Cu+ (scalar X2C, cc-pVDZ-DK, 18 electrons in 38
  orbitals, 12 desktop cores): 1.9 million determinants per stage with PT2 in
  under three minutes; the 3d -> 4s 1D excitation extrapolates to about
  23,600 cm-1, 1,300 cm-1 above EOM-CCSDT. CuF in the QUEST model
  (aug-cc-pVDZ, 18 electrons in 72 orbitals) in state-averaged natural
  orbitals: up to 1.1 million determinants on the desktop give
  variational-plus-PT2 B and C excitation energies of 2.574 and 2.767 eV,
  descending toward the published full-CI 2.561 and 2.751 eV and already
  0.07 and 0.11 eV above CCSDTQ (2.508, 2.660). Each selection cycle extends
  the previous cycle's Hamiltonian (only elements with a new determinant are
  evaluated), Zobrist hashes let a PT2 batch skip other batches' determinants
  before forming them, and a 1-million-determinant CuF stage with two roots
  takes 47 s on 12 cores (Hamiltonian 14 s, Davidson 16 s, PT2 18 s; 67 s
  before these changes). PT2 inserts are staged per thread and shard (64 at a
  time under one lock, prefetched), each shard is one array of entries on
  huge pages at most half full, and the space test runs once per external
  determinant: PT2 over 107 million external determinants of a
  420,000-determinant CuF space in 10.2 s instead of 16.5 s. Stages report the
  first-order norm for renormalized PT2 (`renormalized_pt2`, `renormalized=`
  in the extrapolations), and `checkpoint=` / `initial=` resume a series
  with the same results.

- `SpinEOMCCSDT` (`itercc.eom_spin_ccsdt`): closed-shell EOM-CCSDT for
  singlets and triplets (M_S = 0 vectors of one irrep and spin parity), with
  left eigenvectors: `solve(left=True)` and `sigma_left` act with the exact
  transpose of the Jacobian, as spin-orbit and transition matrix elements
  need, and `lambda_source()` gives dE/dT for the ground-state left vector.
  The T1-dressed spin-orbital CCSDT residual (pdaggerq) is differentiated
  along the direction (the T1 legs of the dressed integrals give the r1
  terms) and integrated over spin onto the alpha-dominant blocks a, aa, ab,
  aaa, aab: 469 terms with one direction factor each
  (`itercc._spin_ccsdt_terms`). Every term runs as a binary contraction tree
  whose ground-state subtrees are formed once; the left action runs the
  same trees backwards, at the same cost. Products whose outputs are
  antisymmetric in letters of one operand write only their ordered entries,
  and sums over pairs antisymmetric in both operands run over ordered
  tuples. All products run on the symmetry-blocked executor, and the packed
  vectors are gathered and scattered by compiled kernels. Singlet roots equal
  `EOMSinglet`'s; the terms match finite differences of the dense
  spin-orbital residual for both parities. On CuF in aug-cc-pVDZ-DK (13
  occupied, 63 virtual orbitals, B2 triplet) the right and left actions equal
  those of the CCpy spin-integrated equations to 1.7e-15 and 1.3e-15 (on the
  same packed vector). One 56-core Sapphire Rapids socket takes about 24 s per
  action, right or left (the first one 51 s while plans build; 40 s of setup,
  38 GiB of cached intermediates, 106 GB peak); the CCpy equations took 585 s
  (right) and 775 s (left) on 32 cores. Ordered outputs move the right action
  from 21.7 to 23.5 s and the left one from 27.6 to 24.5 s. `share=` lets
  engines of other irreps and spins reuse the ground blocks and intermediates.

- `finite_field_polarizability` gives the static polarizability tensor of any
  ground-state method, CCSDTQ included (analytic response stops at CCSDT), as
  the second derivative of its energy in uniform fields with fixed orbitals.
  A solver factory builds each point's solver from the perturbed
  one-electron matrix and the labels the field conserves. Fields of 1e-3 and
  2e-3 au are extrapolated to zero field in F^2. Off-diagonal elements come
  from the lines (e_i + e_j)/sqrt(2). Symmetry skips the -F points of
  components that are not totally symmetric and the elements between
  different irreps (water: 9 solves instead of 25). Each point starts from
  the amplitudes predicted by the solved points of its line
  (`itercc.continuation`; 37% fewer iterations). Water CCSDT matches
  `DipoleResponse` to 1e-6 au (cc-pVDZ) and 8e-8 au (aug-cc-pVDZ) with the
  default tolerances (energy 1e-13, residual 1e-11); water CCSDTQ cc-pVDZ
  takes 23 s for the whole tensor on 12 cores. Tests compare against
  analytic CCSDT response and four-electron FCI (sum over states).

- `fold_closed_shell(h, eri, constant, count)` integrates the `count` lowest
  closed-shell orbitals into the one-electron matrix and the constant, and
  `select_active_orbitals(h, eri, constant, labels, nocc, occupied, virtual)`
  keeps chosen occupied and virtual orbitals (indices within each block) in
  solver order with their irreps, rejecting repeated or cross-block indices;
  both serve comparisons with selected CI or FCI on the same active
  Hamiltonian (`itercc.active_space`, exported from `itercc`).

- `itercc.x2c_somf.two_electron_x2c_somf(mol, gamma_ll, gamma_ls, gamma_ss)`
  contracts the two-electron X2C spin-orbit mean-field integrals of a PySCF
  molecule with large-large, large-small and small-small AO densities (block2's
  `get_x2c_hso2e` convention), in shell blocks of about `block_mib` MiB of
  integrals (default 64) instead of the whole n^4 x 9 tensor.

- CPU EOM-CCSDT and CCSDT response actions of a nonzero irrep G run both
  halves of the triples tangent on symmetry-blocked T3. The half with the
  derivative operators (irrep G) on the ground T3 ran on dense-slot T3, every
  slot read and written; it now gathers the ground T3's planes and ladder
  rows from its blocked copy in the orbital labels (the one the feedback
  reads) and multiplies them by the operators' off-diagonal sector blocks
  into the blocked R3 of the labels shifted by G
  (`_blocked_residual.triples_shifted`; seed tables with the sources' irrep,
  rectangular blocks on the GEMM executor, no new kernels). An operator that
  is not exactly of irrep G falls back to the dense-slot half, before
  anything is written; `ITERCC_SHIFTED_TANGENT=0` keeps it. Both halves of
  any irrep are now purified once, on their sum (rounding-level changes).
  CuF 13o/29v (C2v, 12 desktop threads): B1, B2 and A2 actions 1.10-1.16 ->
  0.68-0.74 s, as A1, within 3.4e-16; water B1 roots (three) cc-pVDZ 1.54 ->
  1.27 s, aug-cc-pVDZ 5.99 -> 4.83 s, the same roots and action counts.
  `benchmarks.eom_ccsdt_actions` times actions by target irrep and path.

- Benchmarks: `benchmarks.singlet_eom_ccsdtq` times singlet EOM-CCSDTQ
  actions by variant (no labels with the central difference, one residual
  per action, sector labels, sector-only outputs on and off, a
  `precision="single"` solver) on a synthetic C2v model or a saved ground
  state and checks that they agree (`--phases` adds seconds per residual
  phase);
  `benchmarks/_memory.py` traces resident memory by phase (and glibc's heap
  in use, free and mapped), used by it and by the triplet CuF driver's
  `--memory-trace`. `docs/benchmarks.md` adds CuF aug-cc-pVDZ-DK CCSDTQ and
  singlet EOM-CCSDTQ and CuF triplet EOM-CCSDTQ actions and memory. Its
  vectors are projected into the residual's range
  (`FiniteDifferenceEOM.in_range`: T4 with symmetric occupied ties,
  spin-adapted, like the ground amplitudes and every Davidson vector).

- The CPU pass of T4 into ranks two and three reads T4 once per occupied
  pair instead of three times, with fewer and contiguous products (results
  change in rounding only, for amplitudes in the residual's range):
  - The r2 term, (1/4) <mn|ef> P4_442(T)[mnij, efab], is needed only up to
    the r2 + r2^T(ij, ab) that follows, and up to it equals (1/2) <mn|ef>
    (S[mnij, efab] - S[mnij, eafb]) with S the P4_201 rows gathered for r3
    (T4's pair symmetry and <mn|ef> = <nm|fe>): no P4_442_r2 gathers (two
    per pair), and both orders of a pair in three products into r2[p, q].
  - The r3 and W_vvvvoo terms of the order (q, p) are those of (p, q) with
    b and c exchanged, which the projection of r3 and the symmetrization of
    W_vvvvoo add anyway: one product each per pair, with twice the weight.
  - Its chunks of m hold 2 GiB of packed rows rather than of dense ones (at
    13o/63v two m per chunk with B1-subgroup labels and five with C2v,
    instead of one), so the products add to their outputs fewer times.
  - W_vvoooo and W_vvvvoo are stored with their occupied axes outermost on
    the CPU from the start (`_empty_stored`), the order the R4 tiles read:
    the pass adds to contiguous slabs instead of one cache line per entry,
    and the 21 GB relayout of W_vvvvoo before R4 (13o/63v) is gone.
    `add_transpose_` and `relayout` work through such transposed storage.
  A synthetic 6o/40v model on 12 desktop threads: the C2v residual 12.0 ->
  9.6 s (the pass 2.44 -> 1.21 s), the B1 one-sided singlet action 25.3 ->
  21.0 s (the pass 5.60 -> 2.86 s). CuF aug-cc-pVDZ-DK (13o/63v) on a
  56-core Sapphire Rapids socket: the C2v CCSDTQ residual 293 -> 260 s (the
  pass 68 -> 33 s, the same residual norms), the B1 singlet action 12.6-12.9
  -> 10.4-10.9 min (the pass 3.8-4.0 -> 1.6-1.7 min), peak 410 GiB.

- One-sided singlet EOM-CCSDTQ actions form only the target sector's R4
  entries in the tile programs and the ring terms (`_cpu_packed.OutputSector`
  on the layout during an action; `ITERCC_EOM_SECTOR_OUTPUTS=0` forms every
  allowed entry), each from the same products in the same order: the same
  action bit for bit. CuF aug-cc-pVDZ-DK B1 on one Sapphire Rapids socket,
  in one process: action 902-933 -> 757-775 s.
  - R4 tile programs (`TileProgram._sector`): an entry's tile offset is its
    rows' plus its columns' (disjoint base-nv digits), so each fused block
    splits into the row and column class pairs that reach the sector, half
    its multiply-adds (counted on a 5o/36v C2v model, B1). CuF B1: tile
    programs 430-447 -> 291-320 s.
  - Ring terms: `_cpu_packed.build_plan` (`sector`) splits each block of a
    packed plan the same way. CuF B1: 137-139 -> 116-121 s.
  - Split the same way, the oooo GEMMs ran slower (54 -> 56-57 s) and the r2
    and r3 terms of T4 no faster (230 s either way; their GEMMs stream T4),
    so they form every entry. A synthetic 6o/40v action on 12 desktop
    threads gains nothing from the ring split either (memory-bound GEMMs).
- One-sided singlet EOM-CCSDTQ actions run on `precision="single"` solvers
  (float32 copies of T3 and of the spin-summed T4 rows, every product formed
  in double). The sector entries of Res(T + eps R) are O(eps) term by term,
  so the copies move the action by their rounding relative to those terms;
  the central difference (the totally symmetric sector) still refuses them,
  since there they would round O(1) entries before the division by eps. CuF
  aug-cc-pVDZ-DK B1 on one socket: 757-775 -> 737-758 s per action (tile
  programs 291-320 -> 274-285 s), the action within 1.6e-9 relative and its
  Rayleigh quotient within 2e-13 Eh; a synthetic 6o/40v action on 12 desktop
  threads: 26.6 -> 23.8 s, within 4.3e-9. `benchmarks.singlet_eom_ccsdtq`
  adds the `sector-single` variant.
- The oooo products of the CPU CCSDTQ residual (one GEMM per row class and
  spectator pair) run as the blocks of one call of the indirect GEMM
  (`itercc_fused_contract`) instead of a dense product per class, whose
  route (split sums, plain loops for small blocks) depends on block sizes.
  Results change in rounding only. CuF aug-cc-pVDZ-DK on one socket: oooo
  terms 57 -> 54 s in the B1 action, 23 -> 24 s in the C2v residual.
- The R4 tile programs' fused GEMMs cut each product's sums into the fewest
  equal chunks of at most 1024 (`ITERCC_TILE_KC`; `0` keeps chunks of 256
  like the other GEMMs), so the deep pair products read and write their
  block of the R4 tile about a quarter as often. Results change in rounding
  only. CuF aug-cc-pVDZ-DK C2v CCSDTQ residual on one Sapphire Rapids
  socket: tile programs 150.8 -> 140.6 s, the residual 302 -> 293 s; a
  synthetic 4o/44v model on 12 desktop threads: 1.69 -> 1.55 s.
- The packed T4 kernels combine boxes tile by tile (16^4 entries, every
  term of a tile before the next tile): the projection, its occupied ties,
  the row gathers, the permuted gathers and row permutations of the ring and
  lower-rank passes, and the row adds. A box of a two-irrep layout (the
  B1-subgroup labels of a singlet EOM-CCSDTQ solver) holds about 8 MB at 63
  virtuals, and a term written over a whole box with strides took a cache
  line per entry: the CuF aug-cc-pVDZ-DK B1 action spent 41 s projecting R4
  against about 6 s for a C2v residual. The same terms in the same order per
  entry: bitwise the same results. A synthetic 3o/62v B1 action on 12
  desktop threads: 18.7-19.4 -> 17.1-17.7 s (projection 0.68 -> 0.52 s).
- One-sided singlet EOM-CCSDTQ actions shift the engine's ground amplitudes
  in place (the sector's entries are eps R during the residual and zero
  after it, `QuadruplesSector.clear`, `itercc_sector_rows` mode 3; the
  totally symmetric entries are never written) instead of a fresh copy of
  T1-T4 per action: the same action bit for bit, with one T4 fewer (107 GB
  for CuF aug-cc-pVDZ-DK with B1-subgroup labels). Read-only or
  memory-mapped ground amplitudes are copied once, when the engine is made.
  With the tiled T4 kernels and 1024-sum tile chunks above, the CuF B1
  action on one Sapphire Rapids socket: 1004-1040 -> 765 s, peak 519 -> 413
  GiB (the socket holds 503 GiB; the old peak spilled to the other socket's
  memory). By phase (against the 1004 s action): tile programs 384 -> 294 s,
  T4 into ranks two and three 318 -> 230 s, ring terms 163 -> 138 s, oooo
  terms 72 -> 57 s, R4 projection 41 -> 26 s.
- `combine_states` re-lays saved T4 on all threads on the CPU
  (`CompiledQuadruplesLayout.accumulate_from`: parallel reads of the
  memory-mapped source, `itercc_permute_virtuals4` for the virtual order,
  parallel axpy; core ABI 42): CuF aug-cc-pVDZ-DK, 57 GB archived T4, 412 ->
  167 s into the C2v layout (read from a network file system) and 438 -> 34 s
  into the B1-subgroup layout (from the page cache).

- Singlet EOM-CCSDTQ (`FiniteDifferenceEOM`) runs on symmetry-blocked
  solvers: with `orbital_irreps=spatial_symmetry.sector_labels(labels,
  irrep)` (the full group for irrep 0, the subgroup of half its order in
  which the target irrep is totally symmetric otherwise) the residuals use
  compressed T4 and blocked GEMMs instead of C1 on dense T4, and
  `combine_states` re-lays archived ground states into that layout. For a
  nonzero target irrep one residual per action replaces the central
  difference (`one_sided`, the CPU default): the sector part of
  Res(T + eps R) holds only odd powers of eps, so over eps it equals the
  central difference term for term (bit for bit on the dense layout). The
  EOM vector is packed and shifted through per-class tables of slot
  positions in parallel (`itercc_sector_rows`, core ABI 41) instead of
  single-threaded NumPy indexing with an int64 index per entry (58 GB for
  CuF in aug-cc-pVDZ, now formed only if `indices` is read); its order is
  unchanged, so earlier Ritz vectors restart the new engine. C2v model, 4
  occupied and 20 virtual orbitals, 12 threads: 1.07 s per action -> 0.21 s
  (B1, A2) and 0.25 s (A1), the same actions to 3e-16 (7e-13 for A1). CuF in
  aug-cc-pVDZ-DK (13 occupied, 63 virtual, 7.3e9 EOM parameters) on one
  Sapphire Rapids socket, 56 threads: B1 18.2 min per action (subgroup T4
  107 GB, peak 519 GB; the central difference on the same solver 35.8 min
  and the same action bit for bit), A1 11.8 min (C2v T4 54 GB, peak 420
  GB); archived ground state re-laid in 7 min, engine 1 min, ground energy
  reproduced exactly. Dense C1 actions on 48 cores of an AMD Genoa node had
  not finished after 83 minutes, at a peak of 1.0 TB.

- Less resident memory in triplet EOM-CCSDTQ solves. The slot program's
  freed intermediate cache (blocks of a few MB, kept by glibc) goes back to
  the kernel when the program ends (`_core.release_free_memory`,
  `ITERCC_RELEASE_CACHE=0` keeps it), so the direct terms and the solver's
  vectors are no longer mapped on top of it; the opened direct terms' buffers
  (up to five rows of R4, 14 GB at CuF size) are mapped per call instead of
  kept on the engine; an action writes each rank into one image instead of
  concatenating them; Davidson's preconditioner and Ritz residuals update in
  place chunk by chunk (no full-length temporaries; the same values) and sort
  the diagonal only when unit vectors are needed. CuF on one Sapphire Rapids
  socket, three Davidson iterations from the converged B1 root: peak 137.4 GB
  (834ad7d, double, 64 GiB cache) -> 120.0 GB, between actions
  110-124 -> 43-54 GB, 488 -> 460 s; with single-precision intermediates and
  a 32 GiB cache 87.9 GB, 436 s.

- Opt-in single-precision intermediates for the triplet slot program:
  `TripletEOMCCSDTQ(..., intermediate_precision="single")` (the CuF driver's
  `--intermediate-precision single`; flags bit 1 of `itercc_triplet_slots`,
  core ABI 40). Each intermediate is formed in double and rounded once to
  float32 when its product is done; the cache holds the float values (half
  the bytes, so twice the intermediates for a budget), the fused tile GEMMs
  widen them as they pack them, and products that run in place read a
  double copy made once per tile. Operands, tiles, gathers and the action
  stay float64, and every reader sees the same rounded values, so the action
  does not depend on the thread count, shared pieces or the cache size
  (checked bit for bit). The default stays "double". 6o/24v synthetic model,
  12 threads, warm rank four: with a 1 GiB cache 13.4 -> 8.75 s (misses
  254k -> 159k, intermediates 63 -> 38 thread-s, tile products 38 -> 30),
  with 8 GiB 5.55 -> 4.84 s (tile products 33 -> 27 thread-s, cache peak
  5.3 -> 2.6 GB); the action moves by 4.7e-11 relative (norm). CuF on one
  Sapphire Rapids socket, on the converged B1 root (0.0785964226 Eh): the
  action moves by 5.6e-8 relative (rank four only), the Rayleigh quotient by
  -4.4e-12 Eh and the residual norm from 8.12653e-7 to 8.12689e-7. Warm
  actions: double with a 64 GiB cache 94.6 s, single 85.6 s (every distinct
  intermediate fits: 37 GB cached), single with a 32 GiB cache 87.7 s at a
  peak of 80 GB (double, 64 GiB: 115 GB).

- With fused products (the default), the triplet slot program merges a slot's
  tiles of one kind whose gathers agree up to a sign
  (`SlotProgram._merge_tiles`, `itercc_slot_program.merge_order`, core ABI
  39): their products add into one tile, scaled by that sign, which is
  cleared once and gathered once, so about half as many tiles are cleared and
  gathered (219,600 -> 106,870 per action on the 6o/24v synthetic model). The
  action changes at rounding (1.5e-16 relative); unfused products, waves and
  `ITERCC_MERGE_TILES=0` keep every tile on its own. 6o/24v, 12 threads, warm
  rank four 6.50 -> 6.02 s (gathers 15.2 -> 7.9 thread-s, clearing 1.4 ->
  0.9, slot sums 2.2 -> 1.2, tile products 31.3 -> 33.5). CuF on one
  Sapphire Rapids socket, one process, second actions: 106.7 s with every
  tile on its own, 97.0 s merged (gathers 1139 -> 558 thread-s, clearing 184
  -> 109, slot sums 192 -> 110, tile products 2658 -> 2750), the actions
  4.3e-17 apart (relative norm). One warm CuF action on one socket has gone
  161.5 -> 97.0 s since 83ae0ff, and its peak memory 151 -> 121 GB.

- The direct packed-T4 terms by opened letters (`itercc_spin_quad_opened`
  and `_lower`) look up each summed tuple's slot and particle column once per
  class instead of once per row of the other particles, and prefetch the
  amplitudes they gather (scattered over up to hundreds of slots) sixteen
  tuples ahead; the same values. 6o/24v synthetic model: the lower ranks'
  gathers 4.3 -> 3.4 thread-s; CuF on one Sapphire Rapids socket, ranks two
  and three's opened terms 8.1 -> 6.8 s (rank four's unchanged, 9.2 s), the
  warm action 107.4 s (profiled). Diagnostic phase timers:
  `-DITERCC_PROFILE_OPENED` and `itercc_opened_profile`.
- The triplet engine's rank-two and rank-three products keep their occupied
  letters inside the GEMMs (`TripletEOMCCSDTQ._lower_engine`, a blocked
  einsum without enumerated letters; `ITERCC_LOWER_WHOLE_HOLES=0` keeps one
  GEMM per occupied tuple, as rank four does). Per occupied tuple the blocks
  were tiny (lmde,cdeijk->cijklm: 3 x 7 x 210 at CuF size, the plain-loop
  route); entries whose occupied letters are out of order between the two
  operands are now formed too and never read. 13o/29v replay, 12 threads: six
  of the largest products 10.8 -> 1.8 s; 6o/24v synthetic model, warm ranks
  one to three 1.71 -> 1.12 s, the action moving at rounding (1.4e-14 of 58).
  CuF, one Sapphire Rapids socket: these products 6.9 -> 1.2 s, rank three
  14.5 -> 9.1 s, the warm action 114.7 -> 108.5 s (profiled). With slots one
  by one the 64 GiB cache still pays: 32 GiB 151.1 s (2.2M misses, peak
  96 GB), 48 GiB 118.3 s (1.16M, 108 GB).
  A warm CuF action on one Sapphire Rapids socket before this change (cProfile,
  114.7 s): slot program 86.2 s, rank-four opened terms 9.2 s, ranks two and
  three's opened terms 8.1 s and blocked products 6.9 s, R3 unpacking 1.8 s.

- The triplet slot program runs each slot's tiles on one thread, in order,
  adding each tile's gathered sums into the slot's rows right away
  (`itercc_triplet_slots`; `ITERCC_SLOT_WAVES=1` keeps the waves). The waves
  wrote about 400 partial sums per row to memory and read them back (17 GiB
  per action on the 6o/24v synthetic model, roughly a terabyte at CuF size),
  and their threads waited at each wave's end; each row now gets the same
  additions in the same order, so the action is the same bit for bit, and no
  partial-sum buffer is allocated (8 GiB at CuF by default). 6o/24v, 12
  threads, warm: rank four 11.06 -> 6.62 s (gathers 29.1 -> 15.5 thread-s,
  slot sums 4.6 -> 2.1, intermediates 14.4 -> 10.1). CuF on one Sapphire
  Rapids socket, same node, second action: 132.0 s with waves, 113.6 s slot
  by slot (rank four 95.5 s; clearing 540 -> 185 thread-s, gathers 1309 ->
  1135, slot sums 304 -> 193, cache waits 75 -> 4), peak 119.8 -> 112.4 GB,
  bit for bit.
- The triplet engine's rank-two and rank-three tiles (one per projection
  group and spin pattern, 135 per action, 429 MB each at CuF size) are one
  reused array stored with its occupied axes outermost
  (`TripletEOMCCSDTQ._lower_tile`), not fresh arrays in letter order. A fresh
  tile paid its page faults inside the blocked products that first wrote it
  (13o/29v replay, 12 threads: 150-420 ms per product instead of 10-60 ms),
  and in letter order a block of the products (one occupied tuple) wrote
  entries 17 KB apart; holes first they are one 195 KB run. Warm products of
  the replay 10.3 -> 7.0 s; 6o/24v synthetic model: ranks two and three
  2.04 -> 1.69 s, the same action bit for bit. `ITERCC_LOWER_TILE_REUSE=0`
  restores fresh tiles. CuF on one Sapphire Rapids socket (second action):
  rank three 49.9 -> 14.5 s (its blocked products 35.9 -> 6.5 s), the action
  161.5 -> 131.7 s, bit for bit.
- The triplet slot program's waves hold at most 8 GiB of partial sums by
  default (was 32 GiB). CuF on one Sapphire Rapids socket, 64 GiB cache,
  second action: 32 GiB 161.5 s and 151 GB peak, 8 GiB 163.5 s and 125 GB,
  2 GiB 173.7 s (the wave barrier grows), the same action bit for bit. The
  same run measured this release's earlier changes on the full action: 161.5 s
  against 219 s at eb539a9; intermediates in letter order 179.0 s (tile
  products 3584 -> 2760 thread-s with chains innermost); T3 and R3 dense
  168.6 s, 190 GB peak and 158 s of setup (spin blocks: 82 s).
- The triplet engine keeps T3, and each action's R3, as their nonzero spin
  blocks (`_spin_quadruples_tiles.SpinBlocks`: 20 and 15 of 64 signatures,
  each a spatial array with its occupied axes outermost; R3 straight from the
  packed vector, `_tensor_scatter.unpack_spin_blocks` on
  `itercc_spin_unpack_blocks`, core ABI 38). The Python tile loops take them
  through `spin_blocks`, the slot program as one dense operand per block,
  rank one block by block. CuF: T3 27.4 -> 8.6 GB and R3 27.4 -> 6.4 GB per
  process, and the setup no longer holds a dense T3 beside its holes-first
  copy. 6o/24v synthetic model: peak memory of an action 12.9 -> 10.7 GiB
  (after setup 3.4 -> 1.5), action 22.8 -> 21.7 s, the action moving at
  rounding (1.8e-15; rank one sums R3 block by block).
  `ITERCC_SPIN_BLOCKED_T3=0` keeps both dense.
- Compact triplet slot-program intermediates store their chains innermost too
  (`ITERCC_INTER_CHAINS_INNER`): a tile product that takes a chain of a cached
  intermediate as its GEMM columns (the skinny products' 2998 x 7 blocks of
  abcfij, say) packs it as consecutive runs instead of one value per cache
  line, and an intermediate is formed once but read about 60 times. Only
  offsets change: the same action bit for bit. 6o/24v synthetic model: tile
  products 55 -> 43 thread-s, rank four 13.6 -> 12.7 s.
- Audit of the triplet EOM-CCSDTQ slot program and engine (no wrong results
  found; these were crashes, hazards and estimates):
  * A complex vector (Davidson certifying a root with a tiny imaginary part)
    crashed the compiled slot program, which needs contiguous rows: actions
    now take contiguous copies of strided vectors.
  * Switches read while a slot program is recorded
    (`SlotProgram.RECORDING_SWITCHES`) key the engine's cached programs; a
    changed switch no longer reuses the stale program.
  * The planner's caches keyed positioned operands by object identity, which
    a freed offset group could hand to another one: positions and
    `OffsetGroup`s are now keyed by content (a digest of the rank table). The
    CUDA engine's call key now includes the positions too.
  * Out of memory inside the compiled slot program terminated the process:
    the tile abandons the intermediates it was producing (their waiters
    produce them), the rest of the wave is skipped and Python raises
    `MemoryError`. A recorded kind whose instructions could read an
    intermediate skipped by another one's cache hit is refused (none is).
  * `Partners` removes its exchange files and workers if a worker fails to
    start, workers stop when the solver dies, split engines refuse probes
    (they would cover one share) and CUDA engines.
  * Memory: the triplet engine's working set is now calibrated (CuF on one
    socket: 71 GB after rank three, 173 GB at the end of an action with a
    64 GiB cache and 32 GiB of partial sums, against an estimate of 417 GiB
    before), apart from the slot program's cache and partial sums; a default
    cache takes half of what the budget leaves beside the working set and
    a solve's Davidson vectors, and a solve's memory budget bounds it.
- One full CuF action (13o/29v, B1 triplet, 3.6e8 parameters) on one Sapphire
  Rapids socket now takes 216 s (rank three 54 s, 33 s once planned; rank
  four 158 s), where the production solve at 6bc25cc took about 900 s per
  action on both sockets. A 64 GiB intermediate cache is as fast as 192 GiB
  (32 GiB: 276 s), and the action is the same bit for bit with any of them.
- `itercc_permuted_gather_add` (the ranks two and three projections, the
  packed-row gathers of the Python paths) runs blocks of 2048 rows term by
  term with each term's strides set up once per block, as the slot program's
  gathers now do; every row gets the same additions in the same order, so
  results are the same bit for bit. 6o/24v synthetic model: 0.22 -> 0.15 s
  per action.
- Compact triplet slot-program tiles store their chains innermost and clear
  only their symmetry-allowed entries (`itercc_slot_tile_axes`, core ABI 36):
  a chain keeps every irrep, so clearing a compact tile whole wrote four
  times the entries its products and gathers touch (C2v), and with a plain
  virtual axis innermost an irrep's run was a few doubles, shorter than a
  cache line; a chain's irrep is a run of hundreds of ranks. On one Sapphire
  Rapids socket the clearing took the same wall time on 28 and 56 threads.
  6o/24v synthetic model: clearing 6.5 -> 3.8 thread-s, gathers 37.9 -> 34.8,
  rank four 14.7 -> 13.9 s, the same action bit for bit. Sapphire Rapids (one
  socket, 56 threads, CuF, first 1500 slots, one node): 20.8 s with tiles
  cleared whole, 20.5 s in letter order, 18.6 s (clearing 81 -> 49
  thread-s, gathers 168 -> 141, tile products 390 -> 343).
  `ITERCC_TILE_CHAINS_INNER=0` keeps letter order, `ITERCC_TILE_RUNS=0`
  clears tiles whole. Dense and compact tiles share the clearing routine
  (`clear_allowed`; a dense tile is four virtual axes), so the program no
  longer carries separate dense-tile runs (core ABI 37).
- Triplet slot-program gathers (tile -> packed rows) run term by term over
  every row of a part, with each term's steps, chain members and rank tables
  hoisted and the common chain shapes (one or two chains of two or three
  letters) compiled separately. Each row gets the same additions in the same
  order, so the action is the same bit for bit. At CuF size the gathers read
  2.8e10 tile entries for the first 1500 slots (2.4 per row) and took 278
  thread-s on one Sapphire Rapids socket; 6o/24v synthetic model: gathers
  70.7 -> 43.8 thread-s. Sapphire Rapids (one socket, 56 threads, CuF, first
  1500 slots): gathers 287 -> 168 thread-s, the slots 22.3 -> 20.6 s.
- Triplet tile products sum an exchangeable pair once (`_spin_quadruples_tiles.
  exchange_pairs`, `paired_routes`). Two summed letters that every operand
  holding one holds both of, in one antisymmetric half (mn of g(oovv) T4,
  ef of g(vovv) T3, ...), leave the summand unchanged when they swap: the
  spin route with (beta, alpha) on them equals the (alpha, beta) one and is
  dropped, and a route with equal spins sums over x < y only (the planner
  now applies ordering constraints between summed letters too:
  `symmetry_blocks.summed_pairs`); every route kept counts twice per pair.
  The g T4 intermediates were most of the slot program's intermediate work
  and nearly all of its packed-T4 reads; at CuF size (first 1500 slots)
  their multiply-adds fall 5.4e11 -> 3.0e11 and packed-T4 entries read
  7.6e10 -> 3.7e10, tile products 1.43e12 -> 1.24e12. 6o/24v synthetic
  model: rank four 16.6 -> 13.3 s, the action moving at rounding (2.8e-14).
  Sapphire Rapids (one socket, 56 threads, CuF, first 1500 slots, both on one
  node): 29.4 -> 22.3 s (intermediates 452 -> 254 thread-s, tile products
  543 -> 369), the action moving by 2.3e-16 relative.
  The blocked engines (CPU and CUDA) take it in ranks two to four; a product
  that falls back to a plain einsum is scaled by 1/2 per pair.
  `ITERCC_PAIR_SUMS=0` sums both orders.
- Triplet dense rank-three tensors (ground T3 and each action's R3) are stored
  with the occupied axes outermost and viewed with the usual axes
  (`eom_spin_quadruples.holes_first_layout`; R3 is unpacked that way at no
  extra cost). A tile fixes two occupied indices; with them innermost and
  spin-interleaved, its slice of a 27 GB CuF T3 lay scattered over the whole
  array, and packing it for the tile GEMMs was 65% of their time on Sapphire
  Rapids. 6o/24v synthetic model: rank four 19.6 -> 16.8 s; the action moves
  at rounding (the planner orders composite letters by stride).
  `ITERCC_HOLES_FIRST=0` keeps the usual layout. On Sapphire Rapids at CuF
  size (first 1500 slots) the slots took 29.2 s, as before (29.3-30.1 s):
  there the tile GEMMs' B packing reads mostly cached intermediates.
- Triplet slot-program tiles store only the entries their kind's products
  form and its gathers read: each chain of ordered particle letters (from one
  antisymmetric subset of the projection group, so always within one
  composite of every product) is one stored axis of ranks, and the gathers
  read a chain through its rank table. At CuF size a tile keeps 28-30% of
  the dense nv^4 on average (0.8-1.5 MB for 70% of the tile work, inside a
  Sapphire Rapids core's L2) and is cleared that much faster. 6o/24v
  synthetic model: rank four 21.0 -> 19.2 s (tile clearing 28 -> 7 thread-s,
  tile products 97 -> 84, gathers 32 -> 50), the same action bit for bit.
  `ITERCC_COMPACT_TILES=0` keeps dense tiles. Core ABI 35. On one Sapphire
  Rapids socket (CuF, first 1500 slots, 192 GiB cache), compact
  intermediates took the slots from 63.8 to 42.7 s (allocation 560 -> 19
  thread-s, gathers 522 -> 199, cache waits 207 -> 41; cache peak 192 -> 68
  GB).
- Triplet slot-program intermediates store only the entries they form.
  The ordering constraints an intermediate keeps link its letters in chains
  (b < c < d for the ring g T4 products); a chain comes from one half of one
  operand and its readers take its letters free, so every product that
  writes or reads the intermediate holds a chain in one composite. Each chain
  is now one stored axis, the rank of each kept tuple in the composites'
  order (`symmetry_blocks.OffsetGroup`, recorded by
  `SlotProgram._compact_intermediates`); only offsets change. The ring
  intermediates take a sixth of their bytes, so the cache holds about six
  times as many and their buffers are cleared six times faster. 6o/24v
  synthetic model, 12 threads, 4 GiB cache: rank four 41.6 -> 22.8 s (cache
  misses 250k -> 109k; intermediates 149 -> 45 thread-s, their allocation
  45 -> 5), the same action bit for bit. `ITERCC_COMPACT_INTERMEDIATES=0`
  stores them whole. On one Sapphire Rapids socket (CuF, first 1500 slots,
  192 GiB cache) ordered intermediates took the slots from 117 to 66 s and
  irrep-ordered program arrays to 64 s, before this change.
- Triplet slot-program intermediates and tiles keep their virtual axes in
  irrep order (`symmetry_blocks.PositionedView`: an operand whose axes store
  index i at positions[i]; the planner uses the positions only in offsets,
  so composites and sums are unchanged). With orbitals in energy order (CuF:
  virtual irreps interleaved) a symmetry class read every fourth value of an
  intermediate; it is now a contiguous run. Tiles are cleared only where
  symmetry allows entries (one run of the last letter per first three)
  instead of whole. 6o/24v synthetic model with interleaved irreps, 12
  threads: rank four 42.1 -> 35.3 s (tile products 161 -> 121 thread-s,
  gathers 44 -> 32), bit for bit the same action. The packed T4 rows that
  products read go through the program's own copies of their row maps,
  indexed by the same positions (rank four about 19% faster than the plain
  layout in all). `ITERCC_SLOT_POSITIONS=0` keeps the orbital order. Core
  ABI 34.
- Triplet slot-program intermediates are formed only where they are read.
  A tile product forms only the entries its projection reads, so it reads an
  operand only where the constraints among the operand's letters that are
  also output letters hold; an intermediate (one template, shared by tiles
  of every kind through the cache) now keeps the constraints all its readers
  share, found from the tile products down through nested intermediates
  (`SlotProgram._order_intermediates`). The g T4 products of the rings
  (`mnef,bcdfklmn->bcdekl`, read only at b < c < d) shrink six-fold. 6o/24v
  synthetic model, 12 threads: rank four 77.7 -> 47.8 s (intermediates 565
  -> 168 thread-s), action within 1.5e-16; on the CuF layout the modelled
  intermediates of half an action fall to 1.6e12 multiply-adds at a 192 GiB
  cache. `ITERCC_ORDERED_INTERMEDIATES=0` forms them whole.
- Davidson (`itercc.krylov.davidson`) pads its first space with unit vectors
  of the lowest diagonal only when the guesses do not cover every root. A
  restart from saved Ritz vectors then starts from them alone: orthogonalized
  against a dense guess, each unit vector became dense, and the CuF triplet
  restart spent three extra full actions (870-906 s each) on them.
- Triplet direct packed T4 terms by opened letters (`itercc_spin_quad_opened`,
  `itercc_spin_quad_opened_lower`, core ABI 33). A family's coefficient takes
  external particles and holes x and summed ones s; its amplitude holds the
  row's other particles B and holes J with the summed letters. For each hole
  tuple J the amplitude block Y[B, s] = T(B + s_p, J + s_h) is gathered once
  per class of B (through precomputed particle-row tables and the slots
  J + s_h), multiplied with the coefficient block in one GEMM, Z[x, B], and
  every output row with holes J + x_h adds its terms' entries of Z. Before,
  every output slot gathered the same blocks again (a ring context was
  shared by about 20 slots) through two address-map lookups per entry. With
  no summed particle (Fock-oo, oooo ladders) B is the row itself and the GEMM
  reads the amplitude slots in place. A slot's contributions with one set of
  external hole positions come from one hole tuple, so they go to that
  set's buffer and are added in order: no output has two writers, and the
  action does not depend on the thread count or on how slots are split
  between processes. Ranks two and three use the same blocks with their
  particle-major rows. Direct terms of an action: water cc-pVDZ 0.239 ->
  0.062 s (rank four), 6o/24v synthetic 5.93 -> 0.65 s (rank four) and
  3.14 -> 0.44 s (rank three), within 1.2e-15 of the contexts and lookups.
  `ITERCC_DIRECT_OPENED=0` restores them.
- Triplet EOM-CCSDTQ ranks two and three by projection groups: each group's
  raw tiles hold only its diagrams and are formed only where the letters of
  each antisymmetric subset, particles and holes alike (their occupied axes
  are whole), are in increasing spin-orbital order; the projection takes the
  group's coset permutations times their multiplicity, as rank four does.
  Water cc-pVDZ: ranks one to three 0.62 -> 0.50 s, within 1.1e-15 of the
  single dense group. Split actions also deal these tiles out to the
  workers (`TripletEOMCCSDTQ.lower_part`; part 0 keeps rank one and the
  direct packed terms), so the workers no longer wait while the solver's
  process forms ranks one to three (142 s of a 1,620 s CuF action).
- Triplet direct packed T4 terms (`itercc_spin_quad_dense`): the contexts of
  one symmetry class share their coefficient block, so they go into one GEMM
  (`indirect_gemm`, coefficient rows of the items' external values times
  the gathered T4 columns) when most of its entries are wanted, instead of
  a coefficient row streamed per item. Results move at round-off (3.9e-16 on
  water, 4.6e-16 on the first 400 CuF slots) and do not depend on the thread
  count. Water cc-pVDZ actions take 2.07 s instead of 2.61 s; at CuF size
  the kernel is bound by its T4 lookups and stays at about 4.8 s per 400
  slots.
- Triplet EOM-CCSDTQ tile products form only the tile entries their
  projection reads. A projection group takes only the permutations that keep
  each antisymmetric subset of external letters increasing, so a raw tile is
  read only where those letters are in increasing spin-orbital order (x < y,
  or x <= y when x's spin bit is the smaller): half of an antisymmetric
  pair, five sixths of a triple. The symmetry-block planner now takes
  ordering constraints on free letters (`_composite(..., ordered)`) and
  builds GEMMs over the kept tuples only; the slot program and the Python
  loop pass the constraints of each projection group and raw spin pattern
  (`ordered_letters`). The water cc-pVDZ action is bit for bit the one
  before; on the CuF layout the tile products of an action fall from
  6.7e13 to 1.7e13 multiply-adds (offline model), and the first 400 CuF
  slots take 62.7 s instead of 99.0 s (slot program 50.0 s instead of
  87.0 s; output within 3.4e-16 of the earlier one).
- `TripletEOMCCSDTQ(partial_sums_gib=...)`: the slot program's waves hold
  that many bytes of partial sums (default: as many as the cache, at most
  32 GiB) instead of always as many as the intermediate cache, so a cache
  can hold every distinct intermediate of an action (262 GiB on CuF, where
  each is then computed once: the modelled action falls from 1.2e14 to
  2.0e13 multiply-adds) without waves of the same size.
- Triplet EOM-CCSDTQ actions split over processes (`itercc/_slot_split.py`):
  every rank-four slot writes only its own rows with values that do not
  depend on the other slots of the process, so `Partners` deals the slots
  out in blocks of 16 to workers (one per NUMA node, bound with numactl)
  that build the same engine, and `sigma` adds their rows: the action of
  one process bit for bit. The slot program runs any table of slots
  (`SlotProgram.run(slots=...)`, no core change); the CuF triplet driver
  takes `--sockets N`.
- One runtime layer for every method (`itercc/_runtime.py`, documented in
  docs/design.md "Canonical machinery"). Threads: `RestrictedCCSDTQ`,
  `EOMSinglet`, `DipoleResponse` and `TripletEOMCCSDTQ` take a `cpu_threads`
  option like `RestrictedCC` (excited-state engines inherit the ground-state
  solver's), and every solve, action and residual call runs in one thread
  scope (`_runtime.threaded`); before, CCSDTQ, EOM and response kernels
  re-read the environment on every call and the EOM's internal solver
  dropped the option. Switches: every ITERCC_* variable is registered with
  its default and when it is read; the table in docs/guide.md is generated
  from the registry (nine switches were undocumented), and a test checks
  that the package and the core read nothing else. `ITERCC_BARE_LADDER`
  means the same in ground states and actions (`0` off, `1` on);
  `ITERCC_CPU_TANGENT` is read once per engine, so the ground-residual
  audit and the actions take one path. The `_scan_pool` and the
  ERI-leakage check follow the thread count.
- Machinery gaps closed: the triplet actions' T1-dressing derivative runs
  the singlet actions' one-pass kernel (`itercc_dress_derivative`) instead
  of four NumPy tensordots; two-operand products into new arrays in
  `bounded_einsum` (triplet coefficient tables, tiles without a symmetry
  engine) run on the core's GEMM executor; the package has one dense
  product engine (`_cpu_quadruples.dense_products`) instead of one per
  thread count plus a CCSDTQ copy; CCSDTQ forms the spin-summed T3 once for
  its T1-T3 equations and T3 intermediates instead of four times; on CUDA,
  fused and direct symmetry-block plans run without cuBLAS 12.5 (only
  grouped-GEMM plans then take the dense fallback) instead of the whole
  engine falling back.
- `restricted.cuda_options()` is the production CUDA CCSDT configuration
  (every fused path; `labelled=False` without orbital labels), used by the
  guide, llms.txt and benchmarks/ccsdt_backends.py instead of hand-written
  flag sets (the guide's set lacked the column paths).
- Measured on water (12 threads, same desktop, before -> after): CCSDTQ
  aug-cc-pVDZ 2.07 -> 1.75 s per iteration, EOM-CCSDT aug-cc-pVDZ 6.45 ->
  5.76 s, CCSD response cc-pVTZ 4.00 -> 3.42 s, CCSDT response cc-pVDZ 5.28
  -> 4.81 s, CCSD aug-cc-pVTZ and CCSDT cc-pVTZ iterations 3-6% faster; a
  triplet EOM-CCSDTQ action (cc-pVDZ) 3.10 -> 2.98 s. Energies agree to
  1e-13 with the same iteration counts.
- docs/benchmarks.md, its charts, README and llms.txt rerun for IterCC on one
  Xeon 8480CL socket (whole node held, 56 threads) after the review: most
  iterations 2-8% faster (CCSD aug-cc-pVQZ 59 -> 57 ms, CCSDT aug-cc-pVTZ
  0.139 -> 0.130 s, CCSDTQ aug-cc-pVDZ 0.496 -> 0.484 s), EOM-CCSDT
  aug-cc-pVDZ 5.2 -> 4.8 s and cc-pVTZ 13.3 -> 12.4 s; the other programs'
  records are unchanged.

- Triplet slot program, propagated: progress callbacks (`slot_finished`, as
  the CuF drivers print them) arrive after each wave of slots again instead
  of once at the end (`itercc_triplet_slots` takes a progress function, core
  ABI 32); the engine's memory estimate counts the program's per-thread
  tiles, dense direct-term buffers and partial sums; docs/guide.md and
  llms.txt describe the compiled path, its switches and why `occupied_slab`
  should stay at its default with packed T4.
- Triplet EOM-CCSDTQ rank-four slots run as one compiled program on the CPU
  (`itercc_triplet_slots`, core ABI 28): water cc-pVDZ on 12 threads takes
  4.0-4.3 s per action instead of 26.4 s (6-31G: 0.47 s instead of 8.3 s).
  Tile kinds and slot classes are recorded once (the first action takes
  5.8 s); every tile then runs on one thread with the single-threaded
  contraction routes, intermediates are shared between tiles through a cache
  with one producer per key, the products into a tile with the same output
  blocks run as fused GEMMs, and each slot adds its tiles' partial sums in the
  loop's order. The action does not depend on the thread count; it agrees
  with the Python slot loop to round-off, and bit for bit with
  `ITERCC_TRIPLET_FUSED=0`. `ITERCC_TRIPLET_PROGRAM=0` runs the loop. Tile
  contraction trees now count an intermediate's work once per tiles that
  share it: goovv t2 t4 contracts g with T4 first (one intermediate per T4
  hole pair instead of a packed-T4 product per tile), which changes the
  action at round-off. Without `intermediate_cache_gib`, the program's cache
  takes half of what the memory budget leaves free (at least 256 MiB).
  On a 16-slot CuF sample (56 cores, 8 GiB cache) the fused program takes
  20.8 s against 43.3 s for the loop; the loop itself is slower there than
  with the old trees (35.5 s), as the g-T4 intermediates crowd its cache.
  Packed T4 read in place keeps its summed occupied letters whole (at CuF
  size the tile byte budget had cut them into 28 slabs, each its own g-T4
  intermediate and product per tile); `occupied_slab` still sets slabs. The
  program runs a wave's tiles sorted by occupied tuple, T4 hole pair first,
  sizes waves by the intermediate cache budget and no longer clears the
  partial-sum buffer.
  On the 16-slot CuF sample this takes the fused program from 20.8 s to
  10.7 s and the loop from 43.3 s to 25.9 s.
- Direct packed spin-orbital terms (`PackedBinaryAction.add`) run all their
  terms in one pass over the rows (`itercc_spin_quad_contract_terms`, core
  ABI 29) instead of one parallel pass per diagram, derivative and
  permutation (144 at rank four), with the same results bit for bit: water
  cc-pVDZ triplet actions take 3.9 s instead of 4.3 s.
- Large intermediates of the triplet slot program (at least 2^22
  multiply-adds, such as the g-T4 products at CuF size) are computed in the
  pieces of the threaded executor, and the tiles waiting for one run its
  pieces with its producer instead of sleeping (core ABI 30). The pieces
  write disjoint entries with the single-threaded arithmetic, so the action
  is unchanged bit for bit; `ITERCC_SLOT_SHARE_WORK` sets the threshold.
  On the first 400 CuF slots (56 cores) this takes 145.5 s instead of
  177.6 s with an 8 GiB cache and 128.5 s instead of 134.3 s with 32 GiB;
  the 16-slot sample was slower there (12.6 s instead of 10.7 s), but so
  were the phases that sharing does not touch, by about the same 16%, so
  most of that difference was the allocation. A producer whose last pieces
  are with helpers now runs pieces of other intermediates being produced
  instead of waiting, and so does a tile whose intermediate has no pieces
  left. In one allocation, sharing is neutral on the 16-slot sample
  (10.3 s either way) and takes the first 400 slots from 151.8 s to 136.4 s.
- Direct packed T4 terms that read the same T4 entries in the same pattern
  share one coefficient table (`merged_diagrams`: at rank four 10 diagrams
  in 5 tables, such as gvovo T4 with (goovv t2) T4 and gvvvv T4 with
  (goovv t2) T4), so each entry is looked up once instead of once per
  diagram, and the ground-state tables are kept across actions. The action
  changes at round-off (1.9e-15 relative on water cc-pVDZ, which takes 3.7 s
  instead of 3.95 s). On the first 400 CuF slots the direct terms take 25.4 s
  instead of 41.6 s (121.2 s instead of 135.1 s in all).
- Rank-four direct packed T4 terms run by contexts
  (`itercc_spin_quad_dense`, core ABI 31): the T4 indices of a row and
  permutation that are not summed form a context; each context gathers its
  T4 values once, only over the summed tuples of the one symmetry class that
  can complete it to a stored amplitude, and every row and permutation using
  it takes a dot product with a dense block of the coefficient. The sparse
  per-row lookups mostly read amplitudes that cannot exist. The same nonzero
  products are added in the same order, so the action is the sparse path's
  bit for bit; `ITERCC_DIRECT_DENSE=0` keeps the sparse path. The slot
  program passes the direct terms 2^22 rows at a time (the kernel runs a
  call's slots in parallel). On the first 400 CuF slots (56 cores) they take
  5.4 s instead of 24.2 s, and the whole probe 99.0 s instead of 121.2 s.
- Benchmarks: IterCC re-timed on one 56-core cluster socket (node held
  whole, 56 threads at most). Three cc-pVTZ EOM-CCSDT roots take 13.3 s
  instead of 15.1 s, CCSD polarizabilities in cc-pVTZ 7.3 s instead of
  8.3 s, and aug-cc-pVQZ CCSD from RHF 2.6 s and 15 GiB instead of 4.3 s
  and 21 GiB (faster AO-to-MO transform); times per iteration are unchanged
  within noise. docs/benchmarks.md no longer quotes 112-thread runs.
- Faster triplet EOM-CCSDTQ actions on the CPU (packed spin, occupied
  tiles): water cc-pVDZ on 12 threads takes 26.5 s per action instead of
  47 s, with the same action bit for bit. A 16-slot CuF quadruples sample
  on 56 cores takes 35.5 s instead of 63.2 s; see docs/triplet-cpu-audit.md.
  Automatic occupied slab width is now the fastest setting on both systems.
  The blocked GEMMs now read packed T4 in place. A T4 operand is passed as
  `SignedRows` (the signed rows of `signed_gather_rows`, described by the
  shape and strides of the dense tile), and `itercc_sym_contract_packed`
  gathers each piece of B a route needs into its own scratch: GEMM panels of
  about 512 KB, skinny chunks, and the small and shallow blocks. The dense
  tile is never written; before, one action wrote and read back about
  112 GiB of it, half of it zero rows, from 15 GiB of distinct rows.
  Addresses split into row and entry once per sum and once per column, so no
  entry needs a division. Core ABI 27.
  A summed occupied axis of a T4 slab is now as wide as a quarter of the tile
  cache allows, not two slots, so one deeper product replaces several that
  each read and write the whole output tile. Dense expansion (GPU, fallbacks)
  works in fixed column chunks, and the rows of a repeated hole are a plain
  zero fill. The blocked engine keeps each plan's kernel addresses and keys
  per-tile labels by identity. An action also resolves its thread count once
  instead of on every kernel call. Packed-row descriptors come from hole
  lookup tables in a few array operations and are reused within an action
  (3.8 s of Python in a 35 s CuF sample). The direct packed terms of
  consecutive slots run in one call, and each slot reuses one raw tile
  buffer. Results are unchanged bit for bit.
  What remains is about 16 s of small GEMMs, most of them shallow ones
  (summing over one occupied index) that are limited by writing their output
  tile, plus about 10 s of Python scheduling the tiles.
- A calculation can now run without PySCF: `itercc.molecule.Molecule`
  (geometry; NWChem-format basis sets from `ITERCC_BASIS_PATH`, the bundled
  `itercc/basis_sets` (STO-3G, Pople 3-21G to 6-311G**, cc-pVXZ and
  aug-cc-pVXZ up to 5Z, def2-SVP to QZVPP; about 2 MB, from PySCF's Basis Set
  Exchange files, see THIRD_PARTY.md) or an installed PySCF's data files;
  elements that need an effective core potential are refused; Abelian point
  group and symmetry-adapted basis from `itercc.symmetry`), the package's own Gaussian integrals
  (`itercc.integrals`: overlap, kinetic, nuclear attraction, dipole and
  8-fold packed ERIs by McMurchie-Davidson in the core, libcint's
  conventions, equal to libcint to 1e-13 through g functions), native
  closed-shell RHF (`itercc.scf.rhf` and `run`: J/K from the core, DIIS, one
  eigenproblem per irrep) and `from_scf` for the solvers. Water RHF agrees
  with PySCF to 1e-12 Eh and CCSD/CCSDT to their convergence. The integrals
  are still 4-5x slower than libcint (cc-pVQZ: 0.39 instead of 0.09 s).
- IterCC transforms the integrals itself: `itercc.integrals.mo_eri` turns
  AO two-electron integrals (8-fold or 4-fold packed, or full) into the
  (pq|rs) the solvers take, as two half transformations of large GEMMs on
  the core's executor (`itercc_unpack_s8_s4`, `itercc_unpack_pairs`,
  `itercc_pack_pairs`, `itercc_scatter_pairs`), written once and with the
  symmetry-forbidden entries exact zeros. `from_rhf` uses it for in-core AO
  integrals instead of PySCF's ao2mo and the unpacking (water aug-cc-pVQZ on
  56 cores: 0.84 instead of 2.0 s; `ITERCC_NATIVE_TRANSFORM=0` restores
  ao2mo). Integrals without a nonzero forbidden entry pass the solvers'
  symmetry projection without a copy, and the solvers check the integrals
  for nonfinite entries in one compiled pass (NumPy's single-threaded scan
  was 0.6 s of the 0.83 s setup for water aug-cc-pVQZ). Core ABI 24.
- Symmetry-blocked contractions below 64k multiply-adds run on one thread,
  and none is cut into tasks of fewer than 8k: EOM and response actions make
  thousands of small ones, which a 56-thread team made slower than 14 threads
  (each task repacks its operands). Water on 56 threads: EOM-CCSDT
  aug-cc-pVDZ 4.7-5.0 instead of 5.8-7.0 s, CCSD polarizability cc-pVTZ
  2.6-2.7 instead of 4.3-4.8 s. One thread or the whole team, never a smaller
  team, which would make libgomp end and restart threads
  (`ITERCC_SYM_PARALLEL_WORK`, `ITERCC_SYM_MIN_SHARE` override the limits).
  Same results.
- The open CCSDT dressing no longer reads g(vvvv) at all: the products of its
  planes with T1 that the occupied rows and columns take come from the bare
  ladder sector blocks (`LadderSectors.contract_t1`,
  `itercc_sectors_contract_t1`). Core ABI 23.
- Cluster benchmark scripts request at most 56 cores (one socket of the
  nodes used) and no longer run on whole nodes; the 112-thread points are
  gone from the thread-scaling records.
- CCSDT no longer dresses the all-virtual block of the integrals. On the
  symmetry-blocked path g~(vvvv) enters only through the sector blocks of the
  ladders, which now come from those of the bare g(vvvv), built once, and
  the rank-o T1 terms (`LadderSectors.dressed`, `itercc_ladder_sectors_t1`);
  the dressing (`itercc_dress_integrals_open`) writes every other entry with
  the same values as before and reads g(vvvv) once. The T3 ladder matrix
  takes the sector blocks over instead of copying them. Water on one 56-core
  socket: aug-cc-pVQZ 0.99 instead of 1.25 s per iteration (dressing 89
  instead of 245 ms, sectors 23 instead of 45 ms; 24x from 56 threads instead
  of 22x), cc-pVQZ 0.24 instead of 0.29 s. `ITERCC_OPEN_DRESSING=0` restores
  the full dressing. Core ABI 19.
- Iterations no longer fault their memory in anew (Linux, glibc). glibc
  maps large blocks on their own and unmaps them when freed, and returns
  freed heap to the kernel at thresholds it moves at run time, so every
  iteration's temporaries came back zero-filled through page faults, in
  some processes and not others. IterCC now fixes the thresholds (blocks up
  to 32 MiB from the heap, up to 1 GiB of free heap kept), takes every block
  from the heap while a CCSDT solve runs and gives the free pages back when
  it ends (`malloc_trim`), and CCSDT keeps one array of dressed integrals
  across iterations (6.8 GB per iteration in water aug-cc-pVQZ). Water CCSDT
  on one 56-core socket: cc-pVQZ 0.29 instead of 0.36 s per iteration,
  aug-cc-pVTZ 0.15 instead of 0.17 s; without symmetry in aug-cc-pVDZ, 23 ms
  in every run instead of 24 or 35 ms. Same peak memory with symmetry. CCSD,
  EOM and CCSDTQ solves keep glibc's mapping of large blocks (no gain, and
  slower or larger with the heap alone). `ITERCC_MALLOC=0`, or glibc's
  `MALLOC_MMAP_THRESHOLD_`, `MALLOC_TRIM_THRESHOLD_` or `MALLOC_MMAP_MAX_`,
  leaves the allocator alone.
- A process that finds the core sources and compiler unchanged loads the
  runtime build without hashing the sources and running the compiler
  (about 80 ms off every small calculation).
- The CCSD dressing runs plane by plane on more than three threads per
  occupied orbital: slab by slab gave only about two per occupied orbital
  work (water aug-cc-pVQZ on one 56-core socket: 19 instead of 35 ms for the
  dressing, 58 instead of 78 ms per iteration). Same operations in the same
  order, so the same results; `ITERCC_DRESS_SLABS=1` or `0` forces either.
- CCSD polarizabilities about 2x faster and EOM-CCSDT and CCSDT response
  actions about 30% faster (water, 12 cores: CCSD cc-pVTZ static and 532 nm
  4.3 instead of 9.4 s; EOM-CCSDT aug-cc-pVDZ, three roots, 7.4 instead of
  10.5 s). Jacobian actions take the particle-particle ladder from the
  undressed g(vvvv) through its derivative (`BareLadder.tangent`, exact for
  directions of any irrep) instead of three dense products with the dressed
  g(vvvv) and its derivative; for CCSD the derivative of the dressed
  integrals skips the all-virtual planes (`itercc_dress_derivative_ccsd`,
  into a buffer kept between actions). From 30 virtual orbitals: below, the
  dense ladder is faster (`ITERCC_BARE_LADDER=0` or `1` forces either). The
  sector scan of the dense-slot triples no longer starts a thread pool per
  call. Core ABI 18.
- The pair-sector ladders of CCSD and of EOM and response actions multiply
  large blocks by one column per occupied pair; with up to 16 columns they
  now stream through `itercc_gemm_narrow` (fixed summation order) at the
  memory bandwidth instead of a fifth of it (water cc-pVQZ CCSD, 12 cores:
  62 instead of 94 ms per iteration).
- CCSDTQ plans its symmetry-blocked T4 products 1.8x faster (water 6-31G*:
  2.3 instead of 3.2 s for the whole calculation; the plans are unchanged).
- CCSD up to 5x faster per iteration (water aug-cc-pVTZ, 12 cores: 33 ms
  instead of 0.16 s; cc-pVTZ 9 instead of 24 ms). The T1-dressed integrals
  are no longer formed in their all-virtual block: the CCSD dressing
  (`itercc_dress_integrals_ccsd`) never
  reads g(vvvv) and writes only the entries the residual uses, into one
  array reused across iterations, and the particle-particle ladder takes the
  undressed g(vvvv) through pair sectors built once, with the T1 terms as
  o^3 v^3 corrections (`BareLadder`). Energies are unchanged to 1e-12 Eh.
- EOM and response actions form the T1 derivative of the dressed integrals
  in one pass without a zero fill (`itercc_dress_derivative`; 2x faster for
  that step; a water cc-pVTZ CCSD polarizability, static and 532 nm, 10.4
  instead of 12.6 s).
- `from_rhf` no longer imports all of PySCF: its density-fitting check read a
  missing attribute through PySCF's lazy `__getattr__` (0.25 s, most of a
  small molecule's first calculation).
- Setup 8-10x faster: the symmetry projection of the integrals runs in one
  pass on all threads (`itercc_project_symmetric`; NumPy's masked copies moved
  the n^4 tensor several times on one thread), and `from_rhf` unpacks PySCF's
  packed MO integrals on all threads (`itercc_unpack_s4`). Water aug-cc-pVTZ:
  `RestrictedCC(...)` 0.10 instead of 0.8-1.0 s. Core ABI 17.
- CCSDT a third to three quarters faster per iteration on many cores (water,
  12 cores: 0.71 instead of 1.17 s in aug-cc-pVTZ, 0.12 instead of 0.18 s in
  cc-pVTZ, 5 instead of 19 ms in 6-31G, where symmetry no longer costs). The
  doubles and triples ladders build their symmetry-sector blocks straight from
  the dressed integrals in one compiled pass (`itercc_ladder_sectors`), with
  the T2 term as small per-sector GEMMs: no copies of g(vvvv), no dense pair
  matrices. The T1 dressing takes two passes over the integrals instead of
  five (`itercc_dress_integrals`). Core ABI 16.
- CCSDTQ on the CPU puts T4 into DIIS by default (`diis_quadruples="auto"`):
  diffuse basis sets converge without settings (water aug-cc-pVDZ: 15
  iterations; before, damped Jacobi on T4 had not converged after 200).
  `diis_quadruples=False` restores the previous behaviour.
- The host T4 DIIS, the Davidson and the GMRES vector products use the core's
  fixed-order sums instead of NumPy's BLAS: OpenBLAS threads left spinning
  slowed the following kernels (small CCSDTQ iterations 2.6x on 12 threads,
  a response sweep about 15%).
- Benchmarks: a one-core comparison that adds ORCA 6.1.1 (AUTOCI CCSDT and
  CCSDTQ; `compare.py --code orca`, `benchmarks/compare_single_core.sh`).
- Benchmarks rerun on one 56-core cluster CPU socket per calculation, up to
  aug-cc-pVQZ, with CCSD (adding Psi4 1.11), CCSD and CCSDT polarizabilities,
  NWChem CCSDTQ and thread scaling to 112 threads
  (`benchmarks/cluster_campaign.sh` with `cluster_runs.txt`;
  `build_nwchem.sh` builds NWChem 7.3.1 with CCSDTQ and CCSDT response;
  `NWCHEM_MPIRUN` sets NWChem's launcher). The desktop records stay in
  `benchmarks/results/`.

## 0.8.0

- `from_rhf` returns the Abelian `orbital_irreps` of the correlated orbitals
  (linear molecules: their D2h or C2v subgroup), and
  `pyscf_interface.dipole_integrals(model, mol)` gives the dipole matrices and
  their irreps, so a PySCF calculation feeds every solver without helper code.
- Documentation rewritten: a short README, one user guide, a design page, and
  `llms.txt`, a complete usage guide for AI assistants.
- `benchmarks/compare.py`: timings against PySCF, ccpy and NWChem on the same
  machine, with the figures and tables of `docs/benchmarks.md`.

## 0.7.0

First standalone release of the IterCC package.

- Compiled CPU core (`core/`, C++17/OpenMP, own GEMM kernels) behind a
  versioned C ABI: built on first use from the sources an installation
  carries, or ahead of time per instruction-set level (`core/build_levels.sh`);
  a native build is skipped on CPUs lacking an extension it was compiled for.
- CCSD and CCSDT on symmetry-blocked T3, including the T3 feedback products
  (CCSDT with 131 virtuals: 11.8 -> 9.2 s per evaluation on one Sapphire
  Rapids socket).
- CCSDTQ on the CPU: packed T4, compiled tile programs reading pair-blocked
  T3 by default, concurrent thread teams in the T4 passes, compact float32
  host T4 DIIS history, symmetry-blocked T3-level products.
- EOM-CCSDT and response actions on compiled product-rule tangents with
  symmetry-blocked feedback; Davidson iterations form each Ritz vector and
  residual once.
- CUDA: saved CCSDTQ states carry the autotuned GEMM routes (bit-exact
  resumes in a new process); host-staged R4 through page-locked buffers;
  the T3 denominator is formed on the fly.
- DIIS falls back to its newest entries when the history is nearly
  dependent; reductions use fixed blocks, so CPU results do not depend on the
  thread count.
