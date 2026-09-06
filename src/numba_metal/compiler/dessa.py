"""De-SSA: eliminate Numba's SSA phi nodes by inserting explicit,
edge-correct parallel-copy assignments -- never by aliasing multiple SSA
variables to one shared storage location.

# Why the previous approach (union-find aliasing) was unsound

The MVP eliminated phi nodes by unioning a phi's target with all of its
incoming SSA variables into one canonical MSL identifier (see git history
for `_unify_phis`), on the theory that "the phi target and its sources
represent the same logical value, so they can share one variable." That
reasoning is incorrect in general: an incoming SSA variable can remain
*live* after the merge point (read again through a different name or a
different phi chain), and forcing it to share storage with the phi target
means a later write to the phi target's shared identifier silently
clobbers a value that must remain unchanged. This is the classic
"lost copy" / parallel-copy hazard from SSA-destruction literature (see
Briggs et al., "Practical Improvements to the Construction and Destruction
of Static Single Assignment Form", 1998, and Boissinot et al. 2009 for the
modern parallel-copy-sequentialization algorithm this module is based on).

Extensive differential testing against this specific codebase's actual
compiled output did not surface a live miscompilation from the old
approach itself (see docs/architecture.md for the investigation record),
because Numba's own bytecode-to-IR frontend happens to insert temporaries
at exactly the points where naive coalescing would otherwise be unsafe
(e.g. tuple-swap `x, y = y, x` lowers through a temporary, not a direct
cross-assignment). That is a property of the *inputs this frontend
happens to produce today*, not a guarantee the union-find scheme itself
provides -- it is not something the compiler enforces or checks, so a
future Numba version, a different bytecode pattern, or a hand-constructed
IR could silently break it. This module replaces that scheme with a
provably correct one regardless of input shape.

Building the differential test suite for this replacement (see
tests/integration/test_phi_dessa.py) did find one real, previously
unverified bug in the surrounding for-range-loop machinery: nested
`for` loops could have the inner loop's iterator-protocol-name
suppression scan (a separate mechanism from phi elimination, used to hide
Python's getiter/iternext/pair_first/pair_second bookkeeping from MSL
output) walk through the *outer* loop's already-suppressed getiter result
and incorrectly suppress the outer loop's own induction variable,
producing MSL that referenced an undeclared identifier. That bug (fixed
alongside this rewrite, in `_suppress_loop_protocol_names`) was in the
loop-protocol suppression logic, not in the union-find phi-aliasing logic
this module replaces -- but it is a concrete demonstration of why every
correctness claim in this codebase needs an executable differential test,
not just an argument.

# The replacement algorithm

1. **No aliasing.** Every distinct SSA variable name gets its own,
   independent MSL storage identifier. Nothing is ever unioned.
2. **Per-edge resolution.** For every phi node `t = phi(incoming_values,
   incoming_blocks)` in a merge block M, and for every `(value, pred)`
   pair in `zip(incoming_values, incoming_blocks, strict=True)` (using
   Numba's explicit parallel arrays -- never dict iteration order, never
   positional guessing), an assignment `t := value` must execute
   if-and-only-if control reached M via the edge `pred -> M`.
3. **Structural edge placement.** Because numba-metal's backend already
   reconstructs *structured* control flow (see `structuring.py`) before
   this pass runs, "control reached M via pred" has an exact structural
   answer: `pred` is a basic block that appears in exactly one place in
   the structured tree, on exactly one path to M. This pass locates that
   `BasicBlockNode` (by CFG block label) and appends the copy there --
   i.e. at the true end of the unique control path that constitutes edge
   `pred -> M`. No CFG edge-splitting data structure is needed because
   the structurer already guarantees each reconstructed arm corresponds
   to exactly one predecessor edge (this is what "structured" means: no
   irreducible/duplicated regions are ever produced -- anything of that
   shape already raises UnsupportedFeatureError during structuring).
4. **Parallel-copy semantics.** When multiple phis fire on the same edge
   (e.g. a loop-carried swap `a, b = b, a`), naively emitting the copies
   in phi-declaration order can read an already-overwritten value in the
   next copy, corrupting the result. This pass builds the copy set for
   each edge as a proper simultaneous-assignment problem and
   sequentializes it as a DAG of dependencies, introducing a temporary to
   break cycles (Boissinot's algorithm, simplified for the small copy
   sets phi resolution produces -- typically 1-4 simultaneous copies per
   edge in practice).
5. **Loop headers get two edges.** A loop header's phi has one incoming
   edge from outside the loop (the initial value, resolved once before
   the loop) and one from the loop's own back-edge (the loop-carried
   value, resolved at the end of the loop body, before the implicit
   continuation / explicit `continue`). Both are handled by the same
   per-edge mechanism: the "loop entry" predecessor block and the
   "back-edge" predecessor block are just two more predecessor labels
   with their own unique structural location.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from numba.core import ir

from numba_metal.compiler.structuring import (
    BasicBlockNode,
    BreakNode,
    ContinueNode,
    IfNode,
    LoopNode,
    Node,
    ReturnNode,
    RotatedWhileNode,
    Seq,
)
from numba_metal.errors import UnsupportedFeatureError


@dataclass
class Copy:
    """One resolved phi assignment: `target := source`, both plain SSA
    variable names (never a canonicalized/aliased name)."""

    target: str
    source: str


@dataclass
class PhiInfo:
    """All information needed to declare and initialize one phi target,
    kept separate from the Copy edge-assignments so declaration (which
    happens once, up front) and assignment (which happens per-edge) can
    be handled independently."""

    target: str
    # list of (value_name, pred_block)
    incoming: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class DeSSAResult:
    """Output of running de-SSA over one kernel's typed IR.

    `edge_copies[block_label]` is the ordered list of Copy assignments to
    emit at the *end* of the structured region corresponding to CFG block
    `block_label`, after that block's own statements and before its
    terminator's structural equivalent (branch/jump/return/break/continue).
    `phi_targets` is every phi target name that must be declared as an
    ordinary local variable (exactly like any other SSA var -- no special
    casing needed at declaration time beyond "it's a phi target, not a
    materializeable expression result").
    """

    edge_copies: dict[int, list[Copy]] = field(default_factory=dict)
    phi_targets: set[str] = field(default_factory=set)
    # Names that appear as a `t = phi(...)` target; these Assign statements
    # themselves must not be emitted as a value-producing MSL statement
    # (there is no MSL expression for "phi"), only their declaration and
    # per-edge Copy assignments matter.
    phi_statement_names: set[str] = field(default_factory=set)


def _collect_phis(blocks: dict[int, ir.Block]) -> list[tuple[int, PhiInfo]]:
    """Find every phi statement, returning (merge_block_label, PhiInfo),
    using Numba's explicit `incoming_values`/`incoming_blocks` parallel
    arrays -- never dict iteration order or positional inference from
    anything else.
    """
    phis: list[tuple[int, PhiInfo]] = []
    for label, block in blocks.items():
        for stmt in block.body:
            if not (isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr)):
                continue
            if stmt.value.op != "phi":
                continue
            expr = stmt.value
            if len(expr.incoming_values) != len(expr.incoming_blocks):
                # Defensive: this would mean Numba's own IR invariant (one
                # incoming value per incoming block) was violated by a
                # future Numba version's phi construction. Fail loudly
                # rather than silently mispairing values with blocks.
                raise UnsupportedFeatureError(
                    f"Internal error: phi node for {stmt.target.name!r} in "
                    f"block {label} has {len(expr.incoming_values)} incoming "
                    f"values but {len(expr.incoming_blocks)} incoming blocks "
                    "(Numba IR invariant violated)."
                )
            info = PhiInfo(target=stmt.target.name)
            for value, pred in zip(
                expr.incoming_values, expr.incoming_blocks, strict=True
            ):
                info.incoming.append((value.name, pred))
            phis.append((label, info))
    return phis


def _sequentialize_parallel_copies(copies: list[Copy]) -> list[Copy]:
    """Order a set of copies that must execute *simultaneously* (all RHS
    values are read using the pre-copy state) into a valid sequential
    order, inserting a temporary to break any cycle.

    Standard algorithm (Boissinot et al. 2009, section 3): build the
    copy graph (edge target <- source for each copy), repeatedly emit any
    copy whose target is not used as a source by any *remaining* copy
    (safe: nothing still-pending needs its old value), and if only cycles
    remain, break one edge with a temporary that captures the pre-copy
    value before that copy's target is overwritten.
    """
    remaining = list(copies)
    result: list[Copy] = []
    temp_counter = 0

    # Multiple phis can target the same variable only if the IR is
    # malformed (SSA guarantees a unique static target per phi); guard
    # against silently dropping a duplicate rather than assuming it away.
    targets_seen: dict[str, int] = {}
    for c in copies:
        targets_seen[c.target] = targets_seen.get(c.target, 0) + 1
    dup = [t for t, n in targets_seen.items() if n > 1]
    if dup:
        raise UnsupportedFeatureError(
            f"Internal error: multiple phi copies target the same variable "
            f"{dup!r} on one edge (SSA invariant violated)."
        )

    while remaining:
        sources = {c.source for c in remaining}
        ready = [c for c in remaining if c.target not in sources]
        if ready:
            for c in ready:
                result.append(c)
                remaining.remove(c)
            continue
        # Every remaining copy's target is some other remaining copy's
        # source: a pure cycle (or several). Break it by copying one
        # target's current value into a fresh temporary, redirecting any
        # copy that was waiting to read that target to read the temporary
        # instead, then let the loop proceed (the original copy is now
        # "ready" since nothing reads its target anymore).
        victim = remaining[0]
        temp_name = f"__dessa_tmp{temp_counter}__{victim.target}"
        temp_counter += 1
        result.append(Copy(target=temp_name, source=victim.target))
        for c in remaining:
            if c.source == victim.target:
                c.source = temp_name
    return result


class DeSSAPass:
    """Computes edge-correct phi resolution for one kernel's structured IR.

    Usage: `DeSSAPass(func_ir.blocks).run(structured_tree)` returns a
    DeSSAResult; the MSL backend then (a) declares every name in
    `phi_targets` as an ordinary local (identical to any other SSA var --
    genuinely no aliasing), (b) skips emitting the `phi` assignment
    statement itself (there is no MSL expression for it), and (c) emits
    `edge_copies[label]` immediately after block `label`'s own statements,
    wherever that block is encountered during structured emission.
    """

    def __init__(self, blocks: dict[int, ir.Block]):
        self.blocks = blocks
        self._phis = _collect_phis(blocks)

    def run(self, structured_tree: Node | None = None) -> DeSSAResult:
        """Compute edge copies for every phi. If `structured_tree` is
        given, validate that every phi's incoming block actually appears
        somewhere in it -- if structuring ever produced a tree missing a
        predecessor (e.g. a future change to structuring.py introduces a
        gap), silently emitting that edge's copy nowhere would be a
        correctness bug indistinguishable from "it happened to work"; this
        check turns it into an immediate, specific compile-time error
        instead.
        """
        result = DeSSAResult()
        # Group all phi incoming edges by predecessor block, so that
        # multiple phis firing on the SAME edge are resolved together as
        # one parallel-copy problem (required for correctness: resolving
        # them independently could reorder reads/writes across phis that
        # share variables, e.g. a loop-carried swap `a, b = b, a`).
        per_edge: dict[int, list[Copy]] = {}
        for _merge_label, info in self._phis:
            result.phi_targets.add(info.target)
            result.phi_statement_names.add(info.target)
            for value_name, pred_label in info.incoming:
                per_edge.setdefault(pred_label, []).append(
                    Copy(target=info.target, source=value_name)
                )

        if structured_tree is not None:
            reachable = find_block_labels_in_arm(structured_tree)
            missing = set(per_edge) - reachable
            if missing:
                raise UnsupportedFeatureError(
                    "Internal error: de-SSA requires an edge-copy "
                    f"insertion point for block(s) {sorted(missing)}, but "
                    "the structured control-flow tree does not contain "
                    "them. This means structuring.py produced a tree that "
                    "drops a predecessor edge for a phi node -- refusing "
                    "to silently produce incorrect MSL."
                )

        for pred_label, copies in per_edge.items():
            result.edge_copies[pred_label] = _sequentialize_parallel_copies(copies)

        return result


# --- Structured-tree helpers for locating edge-copy insertion points ------


def find_block_labels_in_arm(node: Node) -> set[int]:
    """Return every CFG block label that structurally terminates (i.e. is
    the *last* block executed on some path through) the given structured
    node, before control leaves it via fallthrough, break, continue, or
    return. Used to sanity-check that every phi's incoming block is
    actually reachable somewhere in the structured tree (defensive: if
    structuring ever produced a tree missing a predecessor, inserting its
    edge-copy silently nowhere would be a silent correctness bug -- this
    lets the pass raise instead).
    """
    labels: set[int] = set()
    _collect_all_block_labels(node, labels)
    return labels


def _collect_all_block_labels(node: Node, out: set[int]) -> None:
    if isinstance(node, Seq):
        for item in node.items:
            _collect_all_block_labels(item, out)
    elif isinstance(node, BasicBlockNode):
        out.add(node.label)
    elif isinstance(node, IfNode):
        _collect_all_block_labels(node.then_branch, out)
        if node.else_branch is not None:
            _collect_all_block_labels(node.else_branch, out)
    elif isinstance(node, LoopNode):
        out.add(node.header_label)
        _collect_all_block_labels(node.body, out)
    elif isinstance(node, RotatedWhileNode):
        # Unlike LoopNode, the header block here is NOT re-emitted
        # separately from `node.body` (no pre_test/do-while special
        # -casing -- see RotatedWhileNode's and `_emit_rotated_while`'s
        # docstrings): it is already an ordinary BasicBlockNode inside
        # `node.body`, structured by the same _structure_branch/
        # structure_from machinery an if/else or for-loop body uses.
        # Recursing into `node.body` alone (matching how IfNode recurses
        # into its branches, not how LoopNode separately adds its own
        # header_label) finds it there.
        _collect_all_block_labels(node.body, out)
    elif isinstance(node, ReturnNode | BreakNode | ContinueNode):
        pass
    else:  # pragma: no cover - defensive
        raise UnsupportedFeatureError(
            f"Internal error: unknown structured node {type(node)!r} while "
            "locating de-SSA edge-copy insertion points."
        )
