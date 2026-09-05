"""Static scanner: finds functions worth a closer look, without executing
any project code.

Pure `ast` module analysis (stdlib only). This module never imports the
scanned project's modules, never calls `exec`/`eval`/`compile(..., "exec")`
on user code, and never spawns a subprocess -- `numba-metal advisor scan`
must be safe to run against an untrusted or partially-broken repository.

Detection is organized as a set of independent checks over one
`ast.FunctionDef`/`ast.AsyncFunctionDef` node at a time (`_analyze_function`
below); each check only ever *adds* a reason/blocker/pattern tag, so a
new check is always additive and cannot change another check's verdict.

No numeric speedup estimate is ever produced here -- see models.Candidate
and models.Confidence. A HIGH/MEDIUM/LOW confidence reflects how many
independent, mutually-reinforcing static signals were found, not a
probability.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from numba_metal.advisor.models import Candidate, Confidence, ScanError, ScanResult

#: Decorator (sub)names that mark a function as already Numba-compiled.
_NUMBA_DECORATOR_NAMES = frozenset(
    {"jit", "njit", "vectorize", "guvectorize", "cfunc", "stencil"}
)

#: Decorator (sub)names that mark a function as already numba-metal.
_METAL_DECORATOR_NAMES = frozenset({"jit", "device_func"})

#: Module names whose mere presence in a function body is an automatic,
#: severe blocker for GPU execution (I/O, UI, subprocess, DB, network).
_BLOCKING_MODULES = frozenset(
    {
        "os",
        "sys",
        "socket",
        "subprocess",
        "sqlite3",
        "tkinter",
        "requests",
        "urllib",
        "http",
        "shutil",
        "pathlib",
        "asyncio",
        "threading",
        "multiprocessing",
    }
)

#: math/random names that make a function look like a Monte Carlo /
#: repeated-random-sampling kernel.
_RANDOM_CALL_NAMES = frozenset(
    {"random", "randn", "uniform", "normal", "randint", "choice", "gauss"}
)


@dataclass
class _FunctionContext:
    """Everything one function-level analysis pass needs, gathered once."""

    node: ast.FunctionDef | ast.AsyncFunctionDef
    qualified_name: str
    file: str
    decorator: str | None
    is_numba_decorated: bool
    is_metal_decorated: bool


def _decorator_name(dec: ast.expr) -> str | None:
    """Return e.g. "numba.njit" or "metal.jit" for a decorator expression,
    or None if it isn't a recognizable dotted-name/call decorator."""
    target = dec
    if isinstance(target, ast.Call):
        target = target.func
    parts: list[str] = []
    while isinstance(target, ast.Attribute):
        parts.append(target.attr)
        target = target.value
    if isinstance(target, ast.Name):
        parts.append(target.id)
    else:
        return None
    return ".".join(reversed(parts))


#: Numba decorator names that are unambiguous even bare (i.e. imported
#: via `from numba import njit` and used as plain `@njit`, the common
#: style this project's own benchmarks/*.py use throughout) -- "jit" and
#: "device_func" are deliberately excluded here since those bare names
#: are genuinely ambiguous with other libraries' own decorators of the
#: same name (including numba-metal's own `@metal.jit`/`@metal.device_func`,
#: whose classification is handled separately by _METAL_DECORATOR_NAMES).
_UNAMBIGUOUS_BARE_NUMBA_NAMES = frozenset(
    {"njit", "vectorize", "guvectorize", "cfunc", "stencil"}
)


def _classify_decorators(
    decorators: list[ast.expr],
    *,
    bare_names_from_metal: frozenset[str] = frozenset(),
    bare_names_from_numba: frozenset[str] = frozenset(),
) -> tuple[str | None, bool, bool]:
    """Returns (decorator_display_name, is_numba, is_metal).

    A decorator is classified by its full dotted path where available
    (`numba.njit`, `metal.jit`) but this project's OWN benchmarks
    overwhelmingly use bare imported names (`from numba import njit`
    then plain `@njit(...)`) -- found directly while testing this
    scanner against benchmarks/mandelbrot.py, which produced zero
    HIGH POTENTIAL candidates until this bare-name path was added, even
    though the file contains real, working @njit/@metal.jit kernels.

    A bare `jit`/`device_func` is GENUINELY AMBIGUOUS between Numba and
    numba-metal (both libraries use these exact names) -- `njit`/
    `vectorize`/`guvectorize`/`cfunc`/`stencil` are unambiguous even bare
    and always resolved as Numba, but a bare `jit`/`device_func` is only
    classified using `bare_names_from_metal`/`bare_names_from_numba`
    (the file's own `from X import jit` statements, collected once per
    file by `_collect_bare_imports` below) -- if neither import is
    present, a bare `jit`/`device_func` is left UNCLASSIFIED (both False)
    rather than guessed, matching the spec's "never claim more than what
    is actually known" principle.
    """
    is_numba = False
    is_metal = False
    display: str | None = None
    for dec in decorators:
        name = _decorator_name(dec)
        if name is None:
            continue
        if display is None:
            display = name
        leaf = name.rsplit(".", 1)[-1]
        is_dotted = "." in name
        if leaf in _NUMBA_DECORATOR_NAMES:
            if is_dotted and "numba" in name:
                is_numba = True
            elif leaf in _UNAMBIGUOUS_BARE_NUMBA_NAMES:
                is_numba = True
            elif not is_dotted and leaf in bare_names_from_numba:
                is_numba = True
        if leaf in _METAL_DECORATOR_NAMES:
            if "metal" in name.lower():
                is_metal = True
            elif not is_dotted and leaf in bare_names_from_metal:
                is_metal = True
    return display, is_numba, is_metal


def _collect_bare_imports(tree: ast.Module) -> tuple[frozenset[str], frozenset[str]]:
    """One pass over a file's top-level imports (any depth -- `ast.walk`,
    since an import inside a function body, as this repo's own
    benchmarks/mandelbrot.py does for `njit`/`prange`, must be found too)
    to resolve which bare names came from numba-metal vs. plain Numba,
    for `_classify_decorators`'s ambiguous-bare-`jit` handling above.
    Returns (bare_names_from_metal, bare_names_from_numba)."""
    from_metal: set[str] = set()
    from_numba: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".")[0]
            for alias in node.names:
                bound_name = alias.asname or alias.name
                if top in ("numba_metal", "metal"):
                    from_metal.add(bound_name)
                elif top == "numba":
                    from_numba.add(bound_name)
    return frozenset(from_metal), frozenset(from_numba)


class _BlockerVisitor(ast.NodeVisitor):
    """Walks one function body collecting blockers/patterns. Does not
    recurse into nested function/class definitions' own bodies for
    blocker purposes beyond noting they exist (a nested def is itself a
    blocker for GPU compilation today)."""

    def __init__(self) -> None:
        self.blockers: list[str] = []
        self.patterns: set[str] = set()
        self.reasons: list[str] = []
        self.unknowns: list[str] = []
        self.inferred_dtypes: set[str] = set()
        self.parallel_dimension: str | None = None
        self.max_loop_depth = 0
        self._loop_depth = 0
        self.has_return = False
        self.has_recursion_call: str | None = None
        self._own_name: str | None = None

    def set_own_name(self, name: str) -> None:
        self._own_name = name

    # -- blockers -----------------------------------------------------

    def visit_Try(self, node: ast.Try) -> None:
        self.blockers.append(
            f"line {node.lineno}: try/except is not supported inside a "
            "Metal kernel or numba-metal device function"
        )
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        self.blockers.append(
            f"line {node.lineno}: raising exceptions is not supported "
            "inside a Metal kernel"
        )
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        self.blockers.append(
            f"line {node.lineno}: dict literal -- dicts are not a "
            "supported Metal kernel type"
        )
        self.generic_visit(node)

    def visit_Set(self, node: ast.Set) -> None:
        self.blockers.append(
            f"line {node.lineno}: set literal -- sets are not a "
            "supported Metal kernel type"
        )
        self.generic_visit(node)

    def visit_List(self, node: ast.List) -> None:
        self.blockers.append(
            f"line {node.lineno}: list literal -- Python lists are not a "
            "supported Metal kernel type (use a fixed-size array instead)"
        )
        self.generic_visit(node)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        self.blockers.append(
            f"line {node.lineno}: f-string -- string formatting is not "
            "supported inside a Metal kernel"
        )
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and node.value:
            self.blockers.append(
                f"line {node.lineno}: string literal -- strings are not a "
                "supported Metal kernel type"
            )
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.blockers.append(
            f"line {node.lineno}: nested class definition is not "
            "supported inside a Metal kernel"
        )
        # Do not recurse into the class body for loop/pattern purposes.

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            top = alias.name.split(".")[0]
            if top in _BLOCKING_MODULES:
                self.blockers.append(
                    f"line {node.lineno}: imports {alias.name!r} -- "
                    "file/network/subprocess/UI operations cannot run on "
                    "a GPU kernel"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and node.module.split(".")[0] in _BLOCKING_MODULES:
            self.blockers.append(
                f"line {node.lineno}: imports from {node.module!r} -- "
                "file/network/subprocess/UI operations cannot run on a "
                "GPU kernel"
            )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Dynamic attribute access via getattr()/setattr() is caught in
        # visit_Call; plain `obj.attr` chains are fine (e.g. np.float32)
        # and are not flagged here to avoid drowning real blockers in
        # false positives from ordinary module.function() calls.
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        name = None
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr

        if name in ("getattr", "setattr", "hasattr", "delattr"):
            self.blockers.append(
                f"line {node.lineno}: {name}() -- dynamic attribute "
                "access has no Metal kernel equivalent"
            )
        if name == "print":
            self.blockers.append(
                f"line {node.lineno}: print() is not supported inside a " "Metal kernel"
            )
        if name == "open":
            self.blockers.append(
                f"line {node.lineno}: open() -- file I/O cannot run on a " "GPU kernel"
            )
        if self._own_name is not None and name == self._own_name:
            self.has_recursion_call = name
            self.blockers.append(
                f"line {node.lineno}: recursive call to {name!r} -- "
                "recursion is not supported in a Metal kernel"
            )
        if name in _RANDOM_CALL_NAMES:
            self.patterns.add("monte_carlo")
        if name in ("sqrt", "exp", "log", "sin", "cos", "hypot"):
            self.patterns.add("elementwise_math")

        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        self.has_return = True
        self.generic_visit(node)

    # -- structural patterns -------------------------------------------

    def visit_For(self, node: ast.For) -> None:
        self._loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self._loop_depth)
        iter_func_name = None
        if isinstance(node.iter, ast.Call):
            if isinstance(node.iter.func, ast.Attribute):
                iter_func_name = node.iter.func.attr
            elif isinstance(node.iter.func, ast.Name):
                iter_func_name = node.iter.func.id
        is_prange = iter_func_name == "prange"
        is_range_like = iter_func_name in ("range", "prange")
        if is_prange:
            self.reasons.append(f"line {node.lineno}: numba.prange loop")
            self.parallel_dimension = f"line {node.lineno} prange loop"
        if is_range_like and isinstance(node.target, ast.Name):
            self._check_array_subscript_iteration(node, node.target.id)
        self.generic_visit(node)
        self._loop_depth -= 1

    def visit_While(self, node: ast.While) -> None:
        self._loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self._loop_depth)
        has_break = any(isinstance(n, ast.Break) for n in ast.walk(node))
        has_nested_if = any(
            isinstance(n, ast.If) for n in node.body if isinstance(n, ast.If)
        )
        if has_break or has_nested_if:
            self.blockers.append(
                f"line {node.lineno}: while loop contains break/continue "
                "or a nested if/else in its body -- numba-metal's "
                "while-loop structurer only supports a straight-line "
                "loop body (see docs/limitations.md)"
            )
        self.generic_visit(node)
        self._loop_depth -= 1

    def _check_array_subscript_iteration(self, node: ast.For, loop_var: str) -> None:
        """Only treat this as an "array loop" pattern when the loop
        variable itself is actually used as a subscript index somewhere
        in the body -- a bare `for k, v in data.items(): cache[k] = ...`
        has a Subscript node too, but isn't a numerical array-indexing
        loop; requiring the loop var to appear inside a Subscript's
        `slice` (not just anywhere in the loop) avoids that false
        positive."""
        for child in ast.walk(node):
            if isinstance(child, ast.Subscript):
                for name_node in ast.walk(child.slice):
                    if isinstance(name_node, ast.Name) and name_node.id == loop_var:
                        self.patterns.add("array_loop")
                        return

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, (ast.Add, ast.Mult)) and self._loop_depth > 0:
            pass  # reduction detection below via AugAssign is more precise
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if self._loop_depth > 0 and isinstance(node.target, ast.Name):
            self.patterns.add("reduction")
        self.generic_visit(node)


def _detect_pairwise_distance(ctx: _FunctionContext, visitor: _BlockerVisitor) -> None:
    if visitor.max_loop_depth >= 2 and "array_loop" in visitor.patterns:
        # Heuristic: two nested loops indexing arrays with each other's
        # loop variable strongly resembles an O(n^2) pairwise computation
        # (distance matrices, N-body forces, etc.).
        visitor.patterns.add("pairwise_or_stencil")


def _analyze_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    file: str,
    qualname_prefix: str,
    *,
    bare_names_from_metal: frozenset[str] = frozenset(),
    bare_names_from_numba: frozenset[str] = frozenset(),
) -> Candidate:
    qualified_name = f"{qualname_prefix}{node.name}"
    decorator, is_numba, is_metal = _classify_decorators(
        node.decorator_list,
        bare_names_from_metal=bare_names_from_metal,
        bare_names_from_numba=bare_names_from_numba,
    )

    visitor = _BlockerVisitor()
    visitor.set_own_name(node.name)
    for stmt in node.body:
        visitor.visit(stmt)

    ctx = _FunctionContext(
        node=node,
        qualified_name=qualified_name,
        file=file,
        decorator=decorator,
        is_numba_decorated=is_numba,
        is_metal_decorated=is_metal,
    )
    _detect_pairwise_distance(ctx, visitor)

    reasons = list(visitor.reasons)
    unknowns = list(visitor.unknowns)
    blockers = list(visitor.blockers)
    patterns = set(visitor.patterns)

    if visitor.max_loop_depth >= 1 and "array_loop" in patterns:
        reasons.append(
            f"Nested numerical loop (max depth {visitor.max_loop_depth}) "
            "indexing arrays"
        )
        patterns.add("nested_loop")
    if "reduction" in patterns:
        reasons.append("Accumulator pattern detected (looks like a reduction)")
    if "monte_carlo" in patterns:
        reasons.append("Repeated random sampling detected (Monte Carlo shape)")
    if "pairwise_or_stencil" in patterns:
        reasons.append(
            "Two or more nested loops indexing arrays -- resembles a "
            "pairwise-distance or stencil/grid computation"
        )
    if "elementwise_math" in patterns:
        reasons.append("Element-wise math function calls inside a loop")

    if is_metal and not blockers:
        # Already running on numba-metal today -- not a "candidate" for
        # conversion at all (it's converted), but still worth surfacing
        # distinctly from an ordinary un-decorated function, since
        # scan's job includes finding EXISTING numba-metal usage too
        # (spec section 4: "Existing Numba-Metal decorators").
        reasons.insert(0, f"Already running on numba-metal (@{decorator})")
    elif is_numba and not blockers:
        reasons.insert(0, f"Already decorated with @{decorator}")
    elif not is_numba and not is_metal and reasons and not blockers:
        unknowns.extend(
            [
                "Runtime array sizes",
                "Actual data types",
                "Call frequency",
                "Transfer cost",
            ]
        )

    if not reasons and not blockers:
        # Nothing structurally interesting found; still return a Candidate
        # only the caller decides whether to keep (scanner.scan filters
        # trivial candidates out by default).
        reasons.append("No strong GPU-candidate signal detected")

    if blockers:
        confidence = Confidence.LOW
    elif is_metal:
        confidence = Confidence.HIGH
    elif is_numba and reasons:
        confidence = Confidence.HIGH
    elif len(reasons) >= 2:
        confidence = Confidence.MEDIUM
    elif reasons and reasons[0] != "No strong GPU-candidate signal detected":
        confidence = Confidence.LOW
    else:
        confidence = Confidence.LOW

    return Candidate(
        file=file,
        line_start=node.lineno,
        line_end=getattr(node, "end_lineno", node.lineno) or node.lineno,
        qualified_name=qualified_name,
        decorator=decorator,
        reasons=tuple(reasons),
        unknowns=tuple(unknowns),
        blockers=tuple(blockers),
        parallel_dimension=visitor.parallel_dimension,
        inferred_dtypes=tuple(sorted(visitor.inferred_dtypes)),
        confidence=confidence,
        is_numba_decorated=is_numba,
        is_metal_decorated=is_metal,
        patterns=tuple(sorted(patterns)),
    )


class _ClassAwareVisitor(ast.NodeVisitor):
    """Top-level walk that tracks class nesting for qualified names and
    hands every function def to `_analyze_function`."""

    def __init__(
        self,
        file: str,
        *,
        bare_names_from_metal: frozenset[str] = frozenset(),
        bare_names_from_numba: frozenset[str] = frozenset(),
    ) -> None:
        self.file = file
        self._class_stack: list[str] = []
        self.candidates: list[Candidate] = []
        self._bare_names_from_metal = bare_names_from_metal
        self._bare_names_from_numba = bare_names_from_numba

    def _prefix(self) -> str:
        return "".join(f"{c}." for c in self._class_stack)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._class_stack.append(node.name)
        for child in node.body:
            self.visit(child)
        self._class_stack.pop()

    def _visit_function_like(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        self.candidates.append(
            _analyze_function(
                node,
                self.file,
                self._prefix(),
                bare_names_from_metal=self._bare_names_from_metal,
                bare_names_from_numba=self._bare_names_from_numba,
            )
        )
        # Recurse into nested function/closure definitions too, each as
        # its OWN candidate -- this repo's own benchmarks (e.g.
        # benchmarks/mandelbrot.py's _make_numba_cpu_impl/
        # _make_metal_kernel) define the actual @njit/@metal.jit-decorated
        # kernel as a nested closure returned by a plain outer function,
        # not as a bare top-level def. Found directly: scanning
        # benchmarks/ without this recursion produced zero HIGH POTENTIAL
        # candidates despite the directory containing multiple real,
        # working @njit/@metal.jit kernels -- every one of them was a
        # nested closure the original (non-recursing) visitor never saw.
        # Qualified name uses Python's own "<locals>" convention
        # (matching CPython's __qualname__ for closures) so a nested
        # candidate's name is unambiguous and traceable back to its
        # enclosing function.
        self._class_stack.append(f"{node.name}.<locals>")
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.visit(child)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function_like(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function_like(node)


def scan_file(path: Path) -> tuple[list[Candidate], ScanError | None]:
    """Parse and analyze one .py file. Never executes it. Returns
    (candidates, error) -- error is set (and candidates is empty) if the
    file could not be read or parsed; this never raises, per the spec's
    "Unsupported syntax must never crash the complete scan"."""
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [], ScanError(file=str(path), message=f"could not read file: {exc}")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [], ScanError(file=str(path), message=f"syntax error: {exc}")
    try:
        bare_from_metal, bare_from_numba = _collect_bare_imports(tree)
        visitor = _ClassAwareVisitor(
            str(path),
            bare_names_from_metal=bare_from_metal,
            bare_names_from_numba=bare_from_numba,
        )
        visitor.visit(tree)
        return visitor.candidates, None
    except Exception as exc:  # noqa: BLE001 -- see module docstring: a
        # single malformed/unusual file must never abort the whole scan;
        # every other exception path in this module raises nothing (pure
        # AST walking), so this is the one deliberate broad catch,
        # explicitly scoped to "this one file's analysis failed" and
        # always recorded, never silently swallowed.
        return [], ScanError(
            file=str(path), message=f"analysis error: {type(exc).__name__}: {exc}"
        )


def _iter_python_files(
    root: Path, include: str | None, exclude: str | None, max_depth: int | None
) -> list[Path]:
    files: list[Path] = []
    root = root.resolve()
    if root.is_file():
        return [root]
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if max_depth is not None and len(rel.parts) > max_depth:
            continue
        if any(part in (".venv", "venv", "__pycache__", ".git") for part in rel.parts):
            continue
        if include and not path.match(include):
            continue
        if exclude and path.match(exclude):
            continue
        files.append(path)
    return files


def scan(
    root: str,
    *,
    include: str | None = None,
    exclude: str | None = None,
    max_depth: int | None = None,
) -> ScanResult:
    """Statically scan `root` (a file or directory) for GPU-candidate
    functions. Never imports or executes any file under `root`."""
    root_path = Path(root)
    files = _iter_python_files(root_path, include, exclude, max_depth)
    candidates: list[Candidate] = []
    errors: list[ScanError] = []
    for f in files:
        file_candidates, error = scan_file(f)
        candidates.extend(file_candidates)
        if error is not None:
            errors.append(error)
    return ScanResult(
        root=str(root_path),
        candidates=tuple(candidates),
        errors=tuple(errors),
        files_scanned=len(files),
    )
