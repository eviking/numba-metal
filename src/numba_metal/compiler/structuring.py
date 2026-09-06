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
class RotatedWhileNode:
    """A `while` loop whose header block (the back-edge target, holding
    the loop-carried phi nodes) is NOT itself the block that tests the
    loop-continuation condition -- unlike `LoopNode`, which assumes
    those coincide (true for both native `for x in range(...)` loops,
    which always get a dedicated header block, and the simplest
    straight-line `while` bodies).

    This shape arises specifically when a `while` loop's body starts
    with an `if`/`else` (or otherwise doesn't end its very first
    statement-block with the condition re-test): Numba's rotated-`while`
    lowering still puts the loop-carried phi nodes in one block (the
    real header, target of the back-edge), but the condition test ends
    up in a DIFFERENT, later block, reached via whatever control flow
    (here: an ordinary if/else) makes up the rest of the loop body.
    `structure_from(header_label, ..., loop_header=header_label,
    loop_exit=<the test block's exit target>)` handles everything in
    between using the SAME, already-correct if/else/break/continue
    structuring as a `for`-range loop's body -- the header/test split is
    the only genuinely new wrinkle; once it's found, the body region is
    an ordinary structured region like any other.

    Emitted as `while (true) { <body> }`, where `<body>` already
    contains the real condition test as an ordinary `if (exit_cond)
    break;` (inserted into the body's own IfNode/BreakNode structure by
    the same mechanism a `for`-loop's `break` uses) -- there is no
    separate `pre_test`/`do-while` special-casing here, unlike
    `LoopNode`."""

    header_label: int
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


Node = (
    Seq
    | BasicBlockNode
    | IfNode
    | LoopNode
    | RotatedWhileNode
    | ReturnNode
    | BreakNode
    | ContinueNode
)


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
        at `stop_at` (exclusive) or a terminating node (return/break/continue).

        A `label` (or any block reached along the way) that is exactly
        `loop_exit` or `loop_header` -- with no statements of its own
        contributed by THIS arm, i.e. it is reached as a bare jump
        target -- terminates the region with `BreakNode`/`ContinueNode`
        immediately, without re-structuring that block's real content a
        second time. This mirrors the check `_structure_one` already
        does for a plain `ir.Jump` terminator (`target == loop_exit`/
        `target == loop_header`), but applies it uniformly regardless of
        how `label` was reached -- including as one arm of an if/else
        whose `_find_merge` search legitimately found no common
        reconvergence point (which happens exactly when the OTHER arm
        is a `continue`/back-edge, correctly excluded from the merge
        search's reachable set -- see `_reachable_without_loop_back` --
        leaving this arm's target, the loop's real exit block, as the
        only candidate and therefore un-mergeable by definition). Without
        this check, `_structure_branch` would call `structure_from` on
        that arm anyway and fully re-emit the exit block's contents
        inside the loop body, in addition to the identical content
        emitted once more, correctly, by whatever structures the
        loop's own returned continuation label afterward -- a real,
        confirmed duplication bug found via a `while` loop with an
        if/else in its body (see `RotatedWhileNode`), not merely a
        hypothetical.
        """
        if label == loop_exit:
            return BreakNode()
        if label == loop_header:
            return ContinueNode()
        seq = Seq()
        current = label
        while current is not None and current != stop_at:
            if current == loop_exit:
                seq.items.append(BreakNode())
                current = None
                break
            if current == loop_header:
                seq.items.append(ContinueNode())
                current = None
                break
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

        if (
            label != loop_header
            and self._is_rotated_while_header(label)
            and isinstance(term, ir.Branch)
            and not self._is_genuine_loop_split(label, term.truebr, term.falsebr)
            and not self._is_genuine_loop_split(label, term.falsebr, term.truebr)
        ):
            # `label` is a genuine loop header (a real back-edge targets
            # it -- see `_is_rotated_while_header`), but its OWN
            # terminator is an ordinary if/else branch, not the loop's
            # condition test (which `_is_loop_header` would have
            # recognized above, exactly as it does for a native
            # `for`-range header or a straight-line `while` header).
            # This is Numba's rotated-`while` lowering putting the
            # loop-carried phi nodes in one block and the condition test
            # in a LATER block, connected by whatever control flow (here:
            # this if/else) makes up the rest of the per-iteration work
            # -- see RotatedWhileNode's docstring for the full CFG-shape
            # explanation.
            return self._structure_rotated_while(label, body_stmts)

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
        # loop), matching the for-range `iternext` pattern. `_is_loop_header`
        # alone is not sufficient, though: it only checks that `body_target`
        # dominates a back-edge predecessor, which is also trivially true
        # of a block that is merely downstream of `body_target` in the CFG
        # (e.g. the merge point of an if/else nested INSIDE `body_target`'s
        # own region) without `body_target` itself being a real loop-entry
        # split at all. The additional, decisive check: in a genuine loop
        # header split, taking the exit edge and taking the body edge are
        # mutually exclusive outcomes -- `exit_target` must never be able
        # to flow forward into `body_target` (if it can, `body_target` is
        # actually downstream of `exit_target`, not a sibling branch of
        # it, and this is not a loop split). Confirmed as a real,
        # reproduced false positive during testing: a one-sided `if` (no
        # `else`) inside a `while` body has its if's taken-arm target
        # flow directly into the if's OWN merge block, and that merge
        # block (which happens to also contain the loop's real condition
        # test, itself dominating the eventual back-edge) was
        # misidentified as a `for`-range-style loop body, corrupting the
        # structured tree before `RotatedWhileNode` detection got a
        # chance to run on the real loop header.
        if self._is_genuine_loop_split(label, true_t, false_t):
            return self._structure_loop(label, body_stmts, term, true_t, false_t)
        if self._is_genuine_loop_split(label, false_t, true_t):
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

    def _can_reach(self, start: int, target: int, *, limit: int = 4096) -> bool:
        """True if `target` is reachable from `start` by following
        forward CFG edges, WITHOUT ever crossing a back-edge (an edge
        `src->dst` where `dst` dominates `src`, i.e. a loop's own
        continuation edge -- see `CFGInfo.is_back_edge`). Excluding
        back-edges is essential, not incidental: a nested loop's INNER
        body is always technically reachable from the OUTER loop's exit
        target if you're willing to go all the way back around the
        outer loop's own back-edge first, but that path is a different
        loop's cycle, not evidence that the outer exit and the inner
        body are the same mutually-exclusive branch split. Confirmed as
        a real regression during testing: an unscoped, back-edge
        -crossing reachability search broke genuinely nested `for`-range
        loops (each with no `if`/`while` of their own at all) by
        wrongly concluding the outer loop's exit could reach the inner
        loop's body, since it technically can, via the outer loop's own
        back-edge -- which is exactly the case this exclusion rules
        out, matching `_reachable_without_loop_back`'s same reasoning
        for a related but distinct purpose (that method is scoped to an
        already-known enclosing loop's body region; this one has no
        such region yet, since it runs BEFORE any loop is confirmed to
        exist at all -- hence needing its own, unscoped version of the
        same back-edge exclusion rather than reusing that method
        directly).

        Used to distinguish a genuine loop-header split (where the exit
        and body edges are mutually exclusive: neither can reach the
        other by ordinary forward flow, only by looping back through
        the header) from a false positive where one candidate "exit" is
        actually just upstream of the candidate "body" in ordinary,
        non-loop control flow -- see `_structure_branch`'s loop
        -detection call sites for the exact case this guards against.
        `limit` bounds the search as a defensive safeguard against
        unexpectedly large CFGs; exceeding it returns True (fail toward
        NOT treating the branch as a loop header, the same conservative
        direction as an actual reachability finding, rather than
        silently declaring unreachability on an incomplete search)."""
        if start == target:
            return True
        seen: set[int] = set()
        stack = [start]
        while stack:
            if len(seen) > limit:
                return True
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            if cur == target:
                return True
            for succ in _successors(self.cfg.blocks, cur):
                if self.cfg.is_back_edge(cur, succ):
                    continue
                if succ not in seen:
                    stack.append(succ)
        return False

    def _is_genuine_loop_split(
        self, header: int, body_target: int, exit_target: int
    ) -> bool:
        """True if branching to `body_target` vs. `exit_target` at
        `header` is a genuine loop-header split: `body_target` leads
        back to `header` (`_is_loop_header`) AND `exit_target` cannot
        reach `body_target` by ordinary (non-back-edge-crossing) forward
        flow (`_can_reach`). Both conditions are required -- see each
        method's own docstring for why `_is_loop_header` alone is not
        sufficient (it also matches a block that merely sits downstream
        of `body_target` in ordinary control flow, not a real loop
        split) -- and BOTH call sites that need this determination
        (`_structure_branch`'s loop detection, and `_structure_one`'s
        `RotatedWhileNode` detection, which needs to confirm a block's
        own branch is NOT already a recognized loop split before
        treating it as an unrelated if/else inside a larger rotated
        -while) must use the exact same combined check -- checking only
        one half in one caller and both in the other was a real,
        reproduced bug during development (the two callers silently
        disagreeing about which branches counted as loop splits)."""
        return self._is_loop_header(
            header, body_target, exit_target
        ) and not self._can_reach(exit_target, body_target)

    def _is_loop_header(self, header: int, body_target: int, exit_target: int) -> bool:
        """True if `body_target` leads (via a simple chain of jumps/branches
        that eventually jumps straight back to `header`) back to `header`,
        i.e. `header` dominates a predecessor that jumps to it - the
        defining property of a natural loop header, restricted to the
        shapes numba-metal supports (for-range style: single back-edge from
        a block directly reachable from body_target).

        `exit_target` is intentionally unused by the check itself (kept
        in the signature for symmetry with call sites and potential
        future use). This check alone is not sufficient to conclude
        `header` is a genuine loop header -- see `_is_genuine_loop_split`,
        which combines this with the additional, decisive check that
        `exit_target` cannot reach `body_target`; a block that merely
        sits downstream of `body_target` in ordinary (non-loop) control
        flow also satisfies this method's dominance check without being
        a real loop split.
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

    def _is_rotated_while_header(self, label: int) -> bool:
        """True if `label` is the target of a genuine back-edge (some
        predecessor it dominates jumps to it), independent of what kind
        of terminator `label` itself has -- unlike `_is_loop_header`,
        which additionally requires a specific `body_target` to be on
        the path to that back-edge predecessor. Used to detect a
        rotated-`while` header whose own terminator is an ordinary
        if/else rather than the loop's condition test (see
        `RotatedWhileNode`)."""
        for pred in self.cfg.preds[label]:
            if pred == label:
                continue
            if self.cfg.dominates(label, pred):
                return True
        return False

    def _structure_rotated_while(
        self, header: int, header_body_stmts: list
    ) -> tuple[Node, int | None]:
        """Structure a rotated-`while` loop whose header's own terminator
        is an if/else, not the condition test -- see `RotatedWhileNode`'s
        docstring for the full CFG-shape explanation.

        The real condition-test block is found by walking the region
        dominated by `header` for a `Branch` where one target leads,
        via a chain of plain (unconditional) jumps only, straight back
        to `header` -- exactly the shape of the trivial "continue" edge
        an ordinary straight-line `while` or `for`-range header's own
        back-edge predecessor has (see `_jumps_only_to`). Once found,
        that block's true/false targets are relabeled continue-target/
        exit-target as `_structure_loop` already does for the simple
        case, and the whole region from `header` to the test block is
        structured as one ordinary region (reusing `structure_from`'s
        existing, already-correct if/else/break/continue handling) with
        `loop_header=header`, so a plain `Jump` back to `header` found
        anywhere in that region (the eventual "continue" edge) is
        already handled by `_structure_one`'s existing
        `target == loop_header` check -- no new statement-level logic
        needed there.
        """
        self._visited_as_loop_header.add(header)
        test_label = self._find_rotated_while_test_block(header)
        if test_label is None:
            raise UnsupportedFeatureError(
                f"Unsupported `while` loop shape at block {header}: could "
                "not locate a condition-test block reachable from the "
                "loop header that leads back to it (see docs/limitations.md)."
            )
        test_block = self.cfg.blocks[test_label]
        test_term = _terminator(test_block)
        if not isinstance(test_term, ir.Branch):
            raise UnsupportedFeatureError(
                f"Unsupported `while` loop shape at block {header}: "
                f"condition-test block {test_label} does not end in a "
                "branch."
            )
        # Only `exit_target` is needed by name below -- the continue
        # -target arm is never referenced directly; it resolves to a
        # `ContinueNode` automatically via `_structure_one`'s existing
        # `target == loop_header` check once the recursive
        # `structure_from` call below reaches it.
        if self._jumps_only_to(test_term.truebr, header):
            exit_target = test_term.falsebr
        else:
            exit_target = test_term.truebr

        body_region = self._compute_loop_body_region(header)
        self._loop_body_regions.append(body_region)
        try:
            # Structure the header's OWN if/else directly (bypassing
            # `structure_from`/`_structure_one`'s dispatch, which would
            # otherwise re-run this same rotated-while detection against
            # `header` a second time and incorrectly reject it as
            # "reached more than once as a loop header" -- `header` is
            # genuinely visited twice here: once to detect it as this
            # loop's header, once to structure its own real content, and
            # only the SECOND visit is the one `_visited_as_loop_header`
            # is meant to guard against a THIRD, genuinely-erroneous
            # revisit of). `header`'s terminator is already known to be
            # an `ir.Branch` -- that was this method's own entry
            # condition in `_structure_one`.
            header_term = _terminator(self.cfg.blocks[header])
            assert isinstance(header_term, ir.Branch)
            head_node, head_continue = self._structure_branch(
                header, header_body_stmts, header_term, header, exit_target
            )
            if head_continue is None:
                # The header's own if/else already fully terminates
                # every arm (e.g. both branches return/break/continue),
                # meaning the test block is never actually reached by
                # fallthrough -- not a shape this backend's rotated
                # -while detection expects; treated as a bug in the
                # detection itself rather than silently emitting
                # something the test block's condition never governs.
                raise UnsupportedFeatureError(
                    f"Unsupported `while` loop shape at block {header}: "
                    "every arm of the header's own if/else terminates "
                    "before reaching the condition test."
                )
            rest_node = self.structure_from(
                head_continue, stop_at=None, loop_header=header, loop_exit=exit_target
            )
            body_node = Seq(items=[head_node, rest_node])
        finally:
            self._loop_body_regions.pop()

        # `body_node` already contains the condition test itself,
        # structured as an ordinary IfNode by the recursive
        # `structure_from` call above (the test block's own Branch is
        # just another block encountered along the way) -- the
        # continue-target arm resolves to ContinueNode and the
        # exit-target arm to BreakNode via `_structure_one`'s existing
        # handling, exactly as a `for`-range loop's body already does.
        # `RotatedWhileNode` only needs to wrap that region in
        # `while (true) { ... }`; there is no separate pre_test/do-while
        # special-casing here, unlike `LoopNode`.
        loop = RotatedWhileNode(header_label=header, body=body_node)
        wrapped = Seq(items=[loop])
        return wrapped, exit_target

    def _find_rotated_while_test_block(self, header: int) -> int | None:
        """Find the block, dominated by `header`, whose `Branch`
        terminator has one target leading via pure jumps back to
        `header` -- the real condition test for a rotated-`while` loop
        whose header block itself doesn't have this shape. Returns the
        first such block found via a forward BFS from `header` (there
        should be exactly one for the loop shapes this backend
        supports; the search order does not matter for correctness,
        only for which equally-valid block is picked in a hypothetical
        ambiguous case, which is not expected to arise from any Numba-
        generated CFG this project has observed)."""
        seen: set[int] = set()
        queue = [header]
        while queue:
            cur = queue.pop(0)
            if cur in seen:
                continue
            seen.add(cur)
            term = _terminator(self.cfg.blocks[cur])
            if isinstance(term, ir.Branch):
                if self._jumps_only_to(term.truebr, header) or self._jumps_only_to(
                    term.falsebr, header
                ):
                    return cur
            for succ in _successors(self.cfg.blocks, cur):
                if succ not in seen and self.cfg.dominates(header, succ):
                    queue.append(succ)
        return None

    def _jumps_only_to(self, start: int, target: int) -> bool:
        """True if `start` is `target` itself, or reaches `target` via a
        chain of blocks whose only statements are the trivial back-edge
        pattern (an empty or phi-resolution-only body ending in an
        unconditional `Jump`) -- i.e. `start` is purely the "go back to
        the loop header" edge, with no real branching of its own."""
        seen: set[int] = set()
        cur = start
        while cur != target:
            if cur in seen:
                return False  # a cycle that never reaches target
            seen.add(cur)
            term = _terminator(self.cfg.blocks[cur])
            if not isinstance(term, ir.Jump):
                return False
            cur = term.target
        return True

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

        A second, separate bug in the same "fewest dominators" tie-break:
        for a NESTED if/else -- one whose enclosing block is itself one
        arm of an outer if, with more code (e.g. a sibling `if` at the
        outer nesting level) following the outer if before the whole
        function's true final merge -- the inner if's real, immediate
        merge block is MORE deeply nested (dominated by the outer
        branch's arm entry too) and so has MORE dominators than a
        far-downstream block that also happens to be reachable from both
        of the inner if's arms (because the inner arm's natural
        fallthrough eventually reaches that same far block). Dominator-
        set SIZE tracks nesting depth, not control-flow distance, so
        "fewest dominators" can and did pick the far, wrong block over
        the near, correct one. Concretely: `if a: (if b: x=0); if c: y=1;
        z=2` structures the inner `if b` with a merge candidate set
        containing both the correct immediate merge (dominated by the
        outer if's true-arm entry, so more dominators) and the outer
        if's own eventual merge past the `if c` sibling (fewer
        dominators, because it's reachable directly from the function
        entry too) -- `min()` picked the latter, duplicating the `if c`
        sibling (and everything after) into both arms of the inner if,
        and recursively so for the inner if's own two branches -- verified
        directly to be exponential in the number of such nested/sequential
        if pairs, and observed inflating a ~50-line real kernel to nearly
        700 lines of generated MSL. Fixed by requiring a real merge
        candidate to actually be dominated by `branch_label` itself (the
        branch doing the diverging) -- a block that is NOT dominated by
        the branch can't be "this if statement's own continuation," by
        definition, regardless of its dominator-set size. Falls back to
        the unrestricted candidate set only if that filter would empty it
        (keeps every previously-passing case, including the one-sided-if
        and repeated-boolean-test shapes above, working exactly as
        before, since in both of those `branch_label` already dominates
        every real candidate).
        """
        true_reach = self._reachable_without_loop_back(true_t)
        false_reach = self._reachable_without_loop_back(false_t)
        common = (true_reach & false_reach) - {true_t}
        if not common:
            return None
        dominated_common = {
            lbl for lbl in common if self.cfg.dominates(branch_label, lbl)
        }
        if dominated_common:
            common = dominated_common
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
