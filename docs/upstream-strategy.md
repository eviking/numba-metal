# Upstream strategy assessment

This document assesses four possible relationships between numba-metal
and the upstream Numba project, and recommends one. It is an
assessment, not an announcement -- nothing here commits to any of these
paths, and formal Numba target registration is explicitly **not**
implemented as part of this project (see "Recommendation" below).

## The four options

### 1. Stay external (status quo)

numba-metal remains exactly what it is: an out-of-tree package that
imports Numba as a library dependency, reuses its frontend/type
inference via a small isolated adapter (`compiler/frontend.py`), and
never registers a `"metal"` target via `numba.core.target_extension`.

**Maintenance cost:** Low-to-moderate. The dependency surface is
explicitly enumerated (`docs/architecture.md`, "Relevant Numba
extension points and private APIs used") and gated at runtime
(`numba_metal.compat.check_numba_compatible`), so a Numba upgrade that
breaks something fails loudly and specifically rather than silently.
The cost is bounded to: (a) validating each new Numba release against
this project's own test suite before widening the supported-version
range, and (b) reacting if Numba changes the shape of
`compiler_machinery`/`typed_passes`/`ir.Expr.phi` internals this project
reads.

**Governance cost:** None. No Numba maintainer review, no CI gating on
Numba's release schedule, no obligation to keep this working across
Numba's own backward-compatibility promises (which do not cover the
internal APIs this project uses anyway).

**Compatibility cost:** This project narrows its own supported-Numba
range instead of Numba widening anything for it. Currently pinned to
exactly `0.67.x` (see `docs/numba-rfc.md`, "Supported versions").

**CI cost:** Fully self-contained; `.github/workflows/ci.yml` matrices
across the Python versions this project supports, with no dependency on
Numba's own CI infrastructure.

### 2. Register a formal `"metal"` target via `numba.core.target_extension`

Adopt Numba's actual target-registration machinery (`TargetDescriptor`,
a `metal` typing/target context, `@lower`-based lowering registered the
way `numba.cuda` does it), so `@metal.jit` becomes a first-class Numba
target dispatch rather than a bespoke wrapper.

**Why this was already rejected for the MVP** (see
`docs/architecture.md`, "Alternatives considered"): that machinery's
lowering half assumes the eventual output is **LLVM IR**, which then
flows into `numba.core.codegen`. There is no publicly documented path
from LLVM IR to Apple's AIR/`air64` GPU IR (confirmed independently, and
corroborated by five years of discussion on numba/numba#5706 -- see
`docs/numba-rfc.md`). Registering as a formal target while still
producing MSL text (not LLVM IR) would mean adopting
`target_extension`'s typing-context half only, which:

- Adds real exposure to APIs Numba's own docs describe as
  "in-development and may change at any time without deprecation
  notices" (`developer/target_extension.html`).
- Gains nothing functionally: the actual "turn typed IR into GPU code"
  work is still the from-scratch MSL backend this project already has,
  regardless of how the dispatcher is registered.
- Would make `@metal.jit` behave differently from `@metal.jit` as
  currently used (e.g. interacting with `@overload`/`@generated_jit`
  registries meant for LLVM-lowerable targets) in ways that would need
  their own extensive testing to characterize.

**Maintenance/governance/compatibility/CI cost:** All higher than
option 1, for a registration mechanism whose main documented benefit
(sharing Numba's generic lowering registries) does not apply here,
since this project's lowering is not LLVM-based.

### 3. Contribute generic hooks upstream (frontend-only extension point)

Propose that Numba expose a narrow, documented (even if
unstable/provisional) function that runs exactly the untyped-IR +
type-inference stages and returns typed `FunctionIR` + `Signature` +
`typemap`, without requiring a downstream consumer to assemble a
`CompilerBase` subclass or reach into `typed_passes` internals directly.
This is the specific ask floated as a question in `docs/numba-rfc.md`.

**Maintenance cost:** Would reduce numba-metal's own internal-API
surface if accepted (replacing `compiler/frontend.py`'s current
`CompilerBase`/`register_pass` usage with a stable call), but adds a
dependency on Numba's own release/review process and timeline, which is
outside this project's control.

**Governance cost:** Requires actual Numba maintainer buy-in and design
review; per the #5706 thread, the core team is small and has explicitly
said new-target work "is quite a lot of work" without a contributor
driving it. A "just expose the frontend" ask is much smaller than a
"add a Metal target" ask, but is still a real ask of limited maintainer
time.

**Compatibility cost:** Neutral-to-positive if accepted -- a documented
frontend-only API would be more stable than the current internal-API
reliance, for both this project and any similar non-LLVM target.

**CI cost:** None until/unless upstream actually adopts something.

### 4. Upstream the entire backend into Numba as a maintained target

Contribute numba-metal itself (or a rewritten version) as
`numba.metal`, maintained inside the Numba project.

**Why this is not recommended, at any point:** Per the #5706 thread,
Numba maintainers have repeatedly and consistently said (2020 and 2024)
that adding a new target is "quite a lot of work" with "continued
maintenance workload," and that they don't expect this without someone
joining the project to drive and sustain it. This project is
single-contributor, alpha-quality, and has not been vetted by anyone
with Numba-core context. Proposing full upstreaming at this stage would
be asking the Numba team to adopt maintenance risk for code they had no
hand in designing, on a codebase this document elsewhere describes with
an explicit alpha warning. This option is only worth revisiting if (a)
the project matures well past MVP status with sustained multi-
contributor maintenance, and (b) a Numba maintainer expresses actual
interest, neither of which is true today.

**Maintenance/governance/compatibility/CI cost:** Highest of all four
options, and not justified by this project's current maturity.

## Recommendation

**Stay external (option 1), and separately raise option 3 as a
low-cost, low-commitment question** in the numba-metal RFC comment
(`docs/numba-rfc.md`) posted to numba/numba#5706 -- not as a formal
proposal, just a question to gauge interest. Formal target registration
(option 2) is not implemented and not recommended: it would trade a
small, well-isolated, already-tested internal-API surface for a larger
one, in exchange for a lowering-sharing benefit that does not apply to
a non-LLVM backend. Full upstreaming (option 4) is out of proportion to
this project's current single-contributor, alpha-quality state and
would not be a responsible ask of a small maintainer team.

This recommendation should be revisited if: Numba's own architecture
changes to support non-LLVM lowering targets more directly, this
project gains additional maintainers and matures past alpha, or a
Numba maintainer responds to the RFC comment with specific interest
that changes the cost/benefit calculation above.
