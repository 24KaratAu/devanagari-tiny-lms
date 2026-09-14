"""Cross-module name contracts, verified without importing anything.

Why this exists. ``src/train/finetune.py`` opens with

    from src.eval.reasoning_eval import MAX_NEW, greedy_continuations, score

and a refactor once deleted ``MAX_NEW`` from that module while leaving the
import in place. Nothing caught it: ``ast.parse`` accepts an undefined name,
the modules that would have failed on import all need torch, and the machine
the refactor ran on had none -- so the whole check was skipped and the break
only surfaced on a GPU, four cells into a notebook.

These tests read the source with ``ast`` instead of importing it, so they run
anywhere and cover exactly the gap that let it through: every name one
first-party module imports from another must actually be defined there.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"


def module_path(dotted: str):
    """Map ``src.eval.reasoning_eval`` to its file, or None if not first-party."""
    if not dotted.startswith("src."):
        return None
    p = SRC.parent / (dotted.replace(".", "/") + ".py")
    if p.exists():
        return p
    pkg = SRC.parent / dotted.replace(".", "/") / "__init__.py"
    return pkg if pkg.exists() else None


def is_submodule(module: str, name: str) -> bool:
    """Whether ``from <module> import <name>`` is importing a submodule.

    ``from src.eval import provenance`` binds a module, not a name defined in
    ``src/eval/__init__.py``, so it must not be judged against that file's
    contents.
    """
    if not module.startswith("src."):
        return False
    pkg = SRC.parent / module.replace(".", "/")
    return (pkg / f"{name}.py").exists() or (pkg / name / "__init__.py").exists()


def literal_at_module_level(path: Path, name: str):
    """Evaluate one module-level literal assignment without importing the file.

    Importing is not an option: the modules these constants live in pull in
    torch, which is exactly why the contract they encode went unchecked.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{path.name} has no module-level {name}")


def top_level_names(path: Path) -> set:
    """Names a module binds at module level: defs, classes, assignments, imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                names.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, ast.If):          # e.g. a try/except-free guard
            for sub in node.body:
                if isinstance(sub, ast.Assign):
                    for t in sub.targets:
                        if isinstance(t, ast.Name):
                            names.add(t.id)
    return names


def first_party_imports():
    """Every ``from src.X import a, b`` across src/, as (importer, module, name)."""
    out = []
    for f in sorted(SRC.rglob("*.py")):
        if "__pycache__" in f.parts:
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module \
                    and node.module.startswith("src.") and node.level == 0:
                for a in node.names:
                    if a.name != "*":
                        out.append((f, node.module, a.name))
    return out


IMPORTS = first_party_imports()


def test_there_are_first_party_imports_to_check():
    """Guard against the collector silently finding nothing."""
    assert len(IMPORTS) > 15, len(IMPORTS)


@pytest.mark.parametrize("importer,module,name",
                         IMPORTS,
                         ids=[f"{i.stem}<-{m.split('.')[-1]}.{n}"
                              for i, m, n in IMPORTS])
def test_imported_name_exists_in_its_module(importer, module, name):
    """The name must be bound at module level in the module it is taken from."""
    if is_submodule(module, name):
        return                      # binds a module, not a name in __init__
    target = module_path(module)
    assert target is not None, f"{module} resolves to no first-party file"
    defined = top_level_names(target)
    assert name in defined, (
        f"{importer.name} imports '{name}' from {module}, which does not "
        f"define it. Defined there: {sorted(defined)[:12]}...")


def test_reasoning_eval_still_exports_what_finetune_needs():
    """The exact contract that broke once, pinned by name."""
    target = module_path("src.eval.reasoning_eval")
    defined = top_level_names(target)
    for name in ("MAX_NEW", "greedy_continuations", "score", "extract_answer",
                 "normalise"):
        assert name in defined, f"reasoning_eval lost {name}"


def test_max_new_covers_every_answer_style():
    """A style with no generation budget would KeyError at decode time."""
    budget = literal_at_module_level(
        module_path("src.eval.reasoning_eval"), "MAX_NEW")
    styles = literal_at_module_level(
        module_path("src.train.reasoning_dataset"), "ANSWER_STYLES")
    assert set(budget) == set(styles), (budget, styles)
    assert all(isinstance(v, int) and v > 0 for v in budget.values()), budget
