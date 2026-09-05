"""Reconstruct structured control flow (if/else, for/while-as-loops with
break/continue) from Numba's typed IR control-flow graph.

Numba's IR is a graph of basic blocks connected by `Branch`/`Jump`
terminators (SSA form, with `phi` nodes at merge points) -- there is no
structured "if" or "for" node once the frontend is done. Metal Shading
Language, like C, has no `goto` (verified empirically against the actual
Metal compiler -- see docs/architecture.md), so numba-metal cannot emit a
CFG-preserving goto-soup and must reconstruct real structured control flow.

This module implements a small, deliberately restricted region-based
structurer:

- A run of blocks with no incoming branches other than fallthrough is a
  linear sequence.
- A block ending in `Branch` whose two successors both reach a common
  immediate post-dominator (or one/both `return`) is emitted as
  `if (...) { ... } else { ... }`.
- A block that is the target of a back-edge (a predecessor that is
  dominated by it) is a loop header, emitted as `while (true) { ... }`;
  the edge that exits the loop becomes `if (!cond) break;` and back-edges
  become the end of the loop body (with `continue` for early back-edges).

Only the control-flow shapes that Numba actually produces for `if`,
`if/else`, and `for x in range(...)` (optionally containing `break`/
`continue`) are recognized -- this matches the task's mandated language
subset. Anything else (e.g. `while` with a non-loop-shaped CFG, complex
generator-derived control flow) raises UnsupportedFeatureError with the
offending block labels rather than emitting incorrect MSL.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from numba.core import ir

from numba_metal.errors import UnsupportedFeatureError

# --- Structured node types ------------------------------------------------


@dataclass
class Seq:
    """A linear sequence of structured nodes."""

    items: list[Node] = field(default_factory=list)


@dataclass
class BasicBlockNode:
    """A single Numba IR basic block's non-terminator statements."""

    label: int
    body: list  # list of ir.Stmt (Assign, SetItem, etc.), terminator excluded


@dataclass
class IfNode:
    """Structured `if (cond) { then_branch } else { else_branch }`."""

    cond: ir.Var
    then_branch: Node
    else_branch: Node | None


@dataclass
class LoopNode:
    """A structured loop (emitted as a native MSL `for` when it matches the
    `for x in range(...)` shape, otherwise `while (true) { if (exit) break; ...}`)."""

    header_label: int
    # statements in the header block *before* the loop-exit test (e.g. phi
    # resolution is handled separately by the backend via phi_sources)
    pre_test: list
    exit_cond: ir.Var  # loop continues while this is falsy is handled by caller
    exit_cond_negated: bool  # True if we should `if (cond) break;`
    body: Node


@dataclass
class ReturnNode:
    """A structured `return;` (kernels never return a value)."""

    value: ir.Var | None


@dataclass
class BreakNode:
    """A structured `break;`."""


@dataclass
class ContinueNode:
    """A structured `continue;`."""


Node = Seq | BasicBlockNode | IfNode | LoopNode | ReturnNode | BreakNode | ContinueNode


# --- CFG helpers -----------------------------------------------------------


def _terminator(block):
    return block.body[-1]


def _successors(blocks: dict[int, ir.Block], label: int) -> list[int]:
    term = _terminator(blocks[label])
    if isinstance(term, ir.Branch):
        return [term.truebr, term.falsebr]
    if isinstance(term, ir.Jump):
        return [term.target]
    return []


def _compute_dominators(blocks: dict[int, ir.Block], entry: int) -> dict[int, set[int]]:
    labels = list(blocks.keys())
    dom = {label: set(labels) for label in labels}
    dom[entry] = {entry}
    preds: dict[int, list[int]] = {label: [] for label in labels}
    for label in labels:
        for succ in _successors(blocks, label):
            preds[succ].append(label)

    changed = True
    while changed:
        changed = False
        for label in labels:
            if label == entry:
                continue
            if not preds[label]:
                continue
            new_dom = set(labels)
            for p in preds[label]:
                new_dom &= dom[p]
            new_dom.add(label)
            if new_dom != dom[label]:
                dom[label] = new_dom
                changed = True
    return dom


@dataclass
class CFGInfo:
    """Dominator-tree and predecessor information for one kernel's CFG,
    used throughout structuring to distinguish loop headers, break/continue
    edges, and if/else merge points."""

    blocks: dict[int, ir.Block]
    entry: int
    dominators: dict[int, set[int]]
    preds: dict[int, list[int]]

    def dominates(self, a: int, b: int) -> bool:
        """True if every path from the entry block to `b` passes through `a`."""
        return a in self.dominators[b]

    def is_back_edge(self, src: int, dst: int) -> bool:
        """True if the edge src->dst is a loop back-edge (dst dominates src)."""
        return self.dominates(dst, src)


def build_cfg_info(blocks: dict[int, ir.Block], entry: int) -> CFGInfo:
    """Compute dominator and predecessor information for a kernel's CFG."""
    dom = _compute_dominators(blocks, entry)
    preds: dict[int, list[int]] = {label: [] for label in blocks}
    for label in blocks:
        for succ in _successors(blocks, label):
            preds[succ].append(label)
    return CFGInfo(blocks=blocks, entry=entry, dominators=dom, preds=preds)


# --- Structuring algorithm ---------------------------------------------


class Structurer:
    """Recursively converts a region of the CFG into structured Nodes.

    `loop_exit_label` / `loop_header_label` track the innermost enclosing
    loop so that jumps to those labels can be recognized as `break`/
    `continue` respectively.
    """

    def __init__(self, cfg: CFGInfo):
        self.cfg = cfg
        self._visited_as_loop_header: set[int] = set()
        # Stack of block-label sets, one per enclosing loop, giving every
        # block that belongs to that loop's body (pushed on loop entry,
        # popped on exit). Used to distinguish `break` (jump to a target
        # outside the current loop body region) from an ordinary forward
        # jump to a shared merge point -- dominance alone cannot make this
        # distinction, because a block reached by both the loop's natural
        # exit edge and a `break` edge is trivially dominated by the loop
        # header regardless of which case it is.
        self._loop_body_regions: list[set[int]] = []

    def structure_from(
        self,
        label: int,
        stop_at: int | None,
        loop_header: int | None,
        loop_exit: int | None,
    ) -> Node:
        """Structure the linear run of blocks starting at `label`, stopping
        at `stop_at` (exclusive) or a terminating node (return/break/continue)."""
        seq = Seq()
        current = label
        while current is not None and current != stop_at:
            node, current = self._structure_one(current, loop_header, loop_exit)
            seq.items.append(node)
        return seq

    def _in_current_loop_body(self, label: int) -> bool:
        if not self._loop_body_regions:
            return True
        return label in self._loop_body_regions[-1]

    def _structure_one(
        self, label: int, loop_header: int | None, loop_exit: int | None
    ) -> tuple[Node, int | None]:
        block = self.cfg.blocks[label]
        term = _terminator(block)
        body_stmts = list(block.body[:-1])

        if label in self._visited_as_loop_header:
            raise UnsupportedFeatureError(
                f"Unsupported control flow: block {label} is reached more "
                "than once as a loop header (irreducible control flow is "
                "not supported)."
            )

        if isinstance(term, ir.Return):
            node = Seq(
                items=[
                    BasicBlockNode(label, body_stmts),
                    ReturnNode(term.value),
                ]
            )
            return node, None

        if isinstance(term, ir.Jump):
            target = term.target
            if loop_header is not None and target == loop_header:
                return (
                    Seq(items=[BasicBlockNode(label, body_stmts), ContinueNode()]),
                    None,
                )
            if loop_exit is not None and target == loop_exit:
                return (
                    Seq(items=[BasicBlockNode(label, body_stmts), BreakNode()]),
                    None,
                )
            if (
                loop_header is not None
                and not self._in_current_loop_body(target)
                and not self.cfg.is_back_edge(label, target)
            ):
                # `target` is outside the current loop's body region --
                # i.e. this is a `break`. Numba's lowering can route a
                # `break` to a shared merge block downstream of the loop's
                # own natural-exit target rather than to that target
                # directly (when the natural exit and a `break` both flow
                # into the same post-loop code via different immediate
                # predecessors), so an exact-match check against
                # `loop_exit` alone is not sufficient; region membership
                # is the correct test because it isn't fooled by that
                # block also being dominated by the loop header.
                return (
                    Seq(items=[BasicBlockNode(label, body_stmts), BreakNode()]),
                    None,
                )
            if self.cfg.is_back_edge(label, target):
                raise UnsupportedFeatureError(
                    f"Unsupported control flow: unexpected back-edge from "
                    f"block {label} to {target} outside of a recognized "
                    "for-range loop."
                )
            return BasicBlockNode(label, body_stmts), target

        if isinstance(term, ir.Branch):
            return self._structure_branch(
                label, body_stmts, term, loop_header, loop_exit
            )

        raise UnsupportedFeatureError(
            f"Unsupported control-flow terminator {type(term).__name__} in "
            f"block {label}."
        )

    def _structure_branch(
        self, label, body_stmts, term: ir.Branch, loop_header, loop_exit
    ) -> tuple[Node, int | None]:
        true_t, false_t = term.truebr, term.falsebr

        # Loop detection: this branch is a loop header if one branch target
        # is dominated by `label` and jumps back to `label` (a natural
        # loop), matching the for-range `iternext` pattern.
        if self._is_loop_header(label, true_t, false_t):
            return self._structure_loop(label, body_stmts, term, true_t, false_t)
        if self._is_loop_header(label, false_t, true_t):
            # exit branch is true_t, loop body is false_t: normalize by
            # treating this as `if (not cond) break` handled inside.
            return self._structure_loop(
                label, body_stmts, term, false_t, true_t, invert=True
            )

        # Otherwise: an if/else diamond (or one-sided if). Find where the
        # two arms re-converge (their common continuation), by walking
        # forward from each until a label dominated only through `label`
        # is found, or one/both arms terminate (return/break/continue).
        merge = self._find_merge(label, true_t, false_t)

        then_node = self.structure_from(true_t, merge, loop_header, loop_exit)
        else_node = (
            self.structure_from(false_t, merge, loop_header, loop_exit)
            if false_t != merge
            else None
        )

        if_node = IfNode(cond=term.cond, then_branch=then_node, else_branch=else_node)
        wrapped = Seq(items=[BasicBlockNode(label, body_stmts), if_node])
        return wrapped, merge

    def _is_loop_header(self, header: int, body_target: int, exit_target: int) -> bool:
        """True if `body_target` leads (via a simple chain of jumps/branches
        that eventually jumps straight back to `header`) back to `header`,
        i.e. `header` dominates a predecessor that jumps to it - the
        defining property of a natural loop header, restricted to the
        shapes numba-metal supports (for-range style: single back-edge from
        a block directly reachable from body_target).
        """
        for pred in self.cfg.preds[header]:
            if pred == header:
                continue
            if self.cfg.dominates(header, pred):
                # There is a back-edge into header. Confirm body_target is
                # on the path to it (i.e. body_target dominates the
                # back-edge predecessor, or is the predecessor itself).
                if body_target == pred or self.cfg.dominates(body_target, pred):
                    return True
        return False

    def _structure_loop(
        self,
        header,
        body_stmts,
        term: ir.Branch,
        body_target,
        exit_target,
        invert=False,
    ) -> tuple[Node, int | None]:
        self._visited_as_loop_header.add(header)
        cond = term.cond
        # Find the back-edge predecessor(s): blocks dominated by `header`
        # that jump directly to `header`.
        back_preds = [
            p
            for p in self.cfg.preds[header]
            if p != header and self.cfg.dominates(header, p)
        ]
        if not back_preds:
            raise UnsupportedFeatureError(
                f"Unsupported loop shape at block {header}: no back-edge found."
            )

        body_region = self._compute_loop_body_region(body_target)
        self._loop_body_regions.append(body_region)
        try:
            body_node = self.structure_from(
                body_target, stop_at=None, loop_header=header, loop_exit=exit_target
            )
        finally:
            self._loop_body_regions.pop()

        loop = LoopNode(
            header_label=header,
            # The header block's own statements (e.g. re-evaluating a
            # `while` condition's operands) must execute both before the
            # loop's first condition test (handled by the `wrapped` Seq
            # below) AND again at the end of every subsequent iteration,
            # immediately before the next condition test -- a `while
            # cond:` loop re-evaluates `cond`'s operands every time
            # through, not just once. `_emit_loop`'s native-`for`-range
            # path never reads this field (the range-shaped header's own
            # statements are pure iterator-protocol bookkeeping, already
            # fully replaced by the native `for` loop's own re-evaluated
            # C-style condition -- see `_detect_for_range`); only the
            # generic `while (true) { ... }` fallback path re-emits
            # `pre_test` at the bottom of the loop body. This was a real,
            # silent-wrong-result bug found while testing a hand-written
            # CAS retry loop (`while not done: ...`): the header's
            # condition-feeding statements ran exactly once, so the loop
            # body's `if (exit_cond) break;` kept re-testing a value that
            # was never recomputed, terminating after a single iteration
            # regardless of the loop's real trip count.
            pre_test=list(body_stmts),
            exit_cond=cond,
            exit_cond_negated=not invert,
            body=body_node,
        )
        wrapped = Seq(items=[BasicBlockNode(header, body_stmts), loop])
        return wrapped, exit_target

    def _compute_loop_body_region(self, body_target: int) -> set[int]:
        """All block labels that belong to this loop's body.

        A block belongs to the body iff `body_target` dominates it -- every
        path from the function entry to that block passes through the
        loop's body entry. This correctly excludes both the loop's natural
        post-exit code and any block a `break` jumps to that happens to
        also be reachable through the natural exit (such a block is
        reachable via two different paths, so `body_target` -- reachable
        only through the loop body -- cannot dominate it, even though the
        loop header itself still trivially does). A naive forward-reachability
        flood-fill from `body_target` does NOT have this property (it would
        wrongly include a `break` target that also lies on the natural-exit
        path), which is why dominance, not reachability, is used here.
        """
        return {
            label for label in self.cfg.blocks if self.cfg.dominates(body_target, label)
        }

    def _find_merge(self, branch_label: int, true_t: int, false_t: int) -> int | None:
        """Find the immediate common continuation of the two arms of an
        if/else starting at true_t and false_t, by walking dominance:
        the merge point is the unique successor-reachable block that both
        arms flow into and that is NOT dominated by branch_label through
        only one arm (i.e. it has predecessors from both arms, or is
        reached after one/both arms return/break/continue).

        Implemented as: collect the set of labels dominated by true_t
        (inclusive) that are pure-linear/if continuations, walking until a
        Return or a label with multiple incoming structurally-relevant
        preds. We use a simpler, sufficient rule for the supported subset:
        walk forward from true_t following single-successor chains and
        the "taken" side of nested ifs is not needed here -- instead we
        find the first label reachable from true_t that is also reachable
        from false_t by scanning dominance sets.

        `true_t` itself is never a valid merge candidate, even when it
        appears in both reachability sets (`false_t` legitimately can be
        the merge -- that's exactly the one-sided-if shape, `branch cond,
        true_t, merge`, where the false edge has no body of its own and
        goes straight to the real continuation). This distinction matters
        for conditions built from `or`/`and` (e.g. `if (v3 or v3):`),
        which Numba lowers as a chain of re-tests of the same boolean
        where the false-arm's own re-test block jumps *directly into* the
        true branch's block (rather than into a separate, later merge
        block) -- so the true branch's own block ends up "reachable from"
        the false arm too, and naively picking the reachable label with
        the fewest dominators can select that in-progress arm block
        itself instead of the real, later reconvergence point.
        Concretely: for `branch v, 112, 98` where block 98 itself branches
        to `112 or 116`, block 112 is reachable from both starts (it IS
        true_t, and it's also directly reachable from false_t=98), and
        had exactly as few dominators as the real merge (116) -- so
        `min()` non-deterministically picked 112, producing an if-node
        whose "merge" was actually still inside the true arm's own body.
        The caller then continued structuring from that bogus merge
        point, re-entering blocks already covered by the true arm and
        eventually re-visiting a loop header a second time ("irreducible
        control flow" false positive). Excluding only true_t (not
        false_t) from candidacy fixes this case while preserving the
        legitimate one-sided-if shape, where false_t genuinely is the
        correct merge and must remain a valid candidate.
        """
        true_reach = self._reachable_without_loop_back(true_t)
        false_reach = self._reachable_without_loop_back(false_t)
        common = (true_reach & false_reach) - {true_t}
        if not common:
            return None
        # The merge point is the common label with the fewest dominators
        # (i.e. the earliest merge point).
        return min(common, key=lambda lbl: len(self.cfg.dominators[lbl]))

    def _reachable_without_loop_back(self, start: int) -> set[int]:
        # Restricted to the current loop's body region (if any): a block
        # reachable only by leaving the loop (a `break` target) must never
        # be offered as an if/else merge candidate, even though it may
        # also be reachable -- through the loop's natural exit edge, which
        # this traversal does not follow -- from the other arm. Without
        # this restriction, a `break` inside one arm of a nested if can be
        # mistaken for an ordinary fallthrough to the same post-loop block
        # the other arm eventually reaches.
        if self._loop_body_regions and start not in self._loop_body_regions[-1]:
            return set()
        region = self._loop_body_regions[-1] if self._loop_body_regions else None
        seen: set[int] = set()
        stack = [start]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for succ in _successors(self.cfg.blocks, cur):
                if self.cfg.is_back_edge(cur, succ):
                    continue
                if region is not None and succ not in region:
                    continue  # would leave the loop body (a break edge)
                stack.append(succ)
        return seen


def structure_function(blocks: dict[int, ir.Block], entry: int) -> Node:
    """Public entry point: structure an entire kernel's CFG."""
    cfg = build_cfg_info(blocks, entry)
    structurer = Structurer(cfg)
    return structurer.structure_from(
        entry, stop_at=None, loop_header=None, loop_exit=None
    )
