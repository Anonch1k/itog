#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse
import configparser
import ast
from typing import Dict, List, Set, Tuple, Optional

# tomllib — stdlib в Python 3.11+
try:
    import tomllib  # type: ignore
except Exception:
    tomllib = None

REPO_MODE_CHOICES = ("clone", "local")

# ========== утилиты/валидация ==========

def eprint(msg: str):
    print(msg, file=sys.stderr)

def error(msg: str, code: int = 2):
    eprint(f"Ошибка: {msg}")
    sys.exit(code)

def validate_package(name: str) -> str:
    if not name:
        error("Имя пакета не может быть пустым.")
    if any(c in name for c in " /\\:*?\"<>|"):
        error("Имя пакета содержит недопустимые символы.")
    return name

def validate_repo(value: str) -> str:
    if not value:
        error("Укажите URL/путь к репозиторию (--repo) или используйте --test-graph.")
    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc:
        return value
    p = Path(value)
    if p.exists():
        return str(p.resolve())
    error("Репозиторий должен быть существующим путём или корректным URL.")

def validate_repo_mode(mode: str) -> str:
    if mode not in REPO_MODE_CHOICES:
        error(f"Режим репозитория должен быть одним из: {REPO_MODE_CHOICES}")
    return mode

# ========== git ==========

def is_git_url(s: str) -> bool:
    return s.endswith(".git") or s.startswith(("git@", "ssh://", "https://", "http://"))

def run_git(args, cwd=None, check=True) -> subprocess.CompletedProcess:
    return subprocess.run(["git"] + args, cwd=cwd, check=check,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def git_clone(repo_url: str, dest: Path) -> Path:
    try:
        run_git(["clone", repo_url, str(dest)])
    except subprocess.CalledProcessError as ex:
        error(f"git clone не удался: {ex.stderr.strip() or ex.stdout.strip()}")
    return dest

def git_try_checkout_version(repo_dir: Path, package: str, version: str) -> None:
    candidates = [f"v{version}", version, f"{package}-{version}"]
    try:
        run_git(["fetch", "--tags"], cwd=repo_dir, check=False)
        for tag in candidates:
            run_git(["fetch", "origin", "tag", tag], cwd=repo_dir, check=False)
        out = run_git(["tag", "--list"], cwd=repo_dir, check=False).stdout.splitlines()
        existing = set(t.strip() for t in out if t.strip())
        for tag in candidates:
            if tag in existing:
                run_git(["checkout", "--quiet", f"refs/tags/{tag}"], cwd=repo_dir, check=True)
                return
        # fallback: ветка с таким именем
        for br in candidates:
            try:
                run_git(["checkout", "--quiet", br], cwd=repo_dir, check=True)
                return
            except subprocess.CalledProcessError:
                pass
    except Exception:
        pass

# ========== парсинг зависимостей (как в Этапе 2/3) ==========

def _normalize_lines(raw: str) -> List[str]:
    lines: List[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        s = re.split(r"\s+#", s, maxsplit=1)[0].strip()
        if s:
            lines.append(s)
    return lines

def parse_pyproject_dependencies(root: Path) -> Optional[List[str]]:
    py = root / "pyproject.toml"
    if not py.is_file() or tomllib is None:
        return None
    try:
        data = tomllib.loads(py.read_text(encoding="utf-8"))
        project = data.get("project") or {}
        deps = project.get("dependencies")
        if deps is None:
            return []
        if isinstance(deps, list):
            return [d.strip() for d in deps if isinstance(d, str) and d.strip()]
        return []
    except Exception:
        return None

def parse_setup_cfg_dependencies(root: Path) -> Optional[List[str]]:
    cfg = root / "setup.cfg"
    if not cfg.is_file():
        return None
    cp = configparser.ConfigParser()
    try:
        cp.read(cfg, encoding="utf-8")
        if cp.has_section("options") and cp.has_option("options", "install_requires"):
            raw = cp.get("options", "install_requires")
            return _normalize_lines(raw)
        return []
    except Exception:
        return None

def _literal_eval_node(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        return None

def parse_setup_py_dependencies(root: Path) -> Optional[List[str]]:
    sp = root / "setup.py"
    if not sp.is_file():
        return None
    try:
        tree = ast.parse(sp.read_text(encoding="utf-8"))
        deps: List[str] = []
        class Finder(ast.NodeVisitor):
            def visit_Call(self, node: ast.Call):
                fn = node.func
                name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else None)
                if name == "setup":
                    for kw in node.keywords:
                        if kw.arg == "install_requires":
                            val = _literal_eval_node(kw.value)
                            if isinstance(val, (list, tuple)):
                                for item in val:
                                    if isinstance(item, str) and item.strip():
                                        deps.append(item.strip())
                self.generic_visit(node)
        Finder().visit(tree)
        return deps
    except Exception:
        return None

def parse_requirements_txt(root: Path) -> Optional[List[str]]:
    for fname in ("requirements.txt", "requirements.in"):
        f = root / fname
        if f.is_file():
            lines: List[str] = []
            for s in _normalize_lines(f.read_text(encoding="utf-8")):
                if s.startswith(("-r ", "--requirement", "-c ", "--constraint")):
                    continue
                lines.append(s)
            return lines
    return None

def extract_runtime_dependencies(repo_root: Path) -> List[str]:
    for fn in (parse_pyproject_dependencies,
               parse_setup_cfg_dependencies,
               parse_setup_py_dependencies,
               parse_requirements_txt):
        deps = fn(repo_root)
        if deps is not None:
            return deps
    return []

def read_declared_version(repo_root: Path) -> Optional[str]:
    if tomllib is not None:
        py = repo_root / "pyproject.toml"
        if py.is_file():
            try:
                data = tomllib.loads(py.read_text(encoding="utf-8"))
                proj = data.get("project") or {}
                v = proj.get("version")
                if isinstance(v, str) and v.strip():
                    return v.strip()
            except Exception:
                pass
    cfg = repo_root / "setup.cfg"
    if cfg.is_file():
        cp = configparser.ConfigParser()
        try:
            cp.read(cfg, encoding="utf-8")
            if cp.has_section("metadata") and cp.has_option("metadata", "version"):
                v = cp.get("metadata", "version").strip()
                if v:
                    return v
        except Exception:
            pass
    sp = repo_root / "setup.py"
    if sp.is_file():
        try:
            tree = ast.parse(sp.read_text(encoding="utf-8"))
            class VF(ast.NodeVisitor):
                ver = None
                def visit_Call(self, node: ast.Call):
                    fn = node.func
                    name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else None)
                    if name == "setup":
                        for kw in node.keywords:
                            if kw.arg == "version":
                                val = _literal_eval_node(kw.value)
                                if isinstance(val, str) and val.strip():
                                    self.ver = val.strip()
                    self.generic_visit(node)
            vf = VF()
            vf.visit(tree)
            return vf.ver
        except Exception:
            pass
    return None

# ========== тестовый режим: парсер текстового графа A..Z ==========

def parse_test_graph(path: Path) -> Dict[str, List[str]]:
    if not path.is_file():
        error("Файл тестового графа не найден.")
    graph: Dict[str, List[str]] = {}
    lineno = 0
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            lineno += 1
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                error(f"Ожидалось 'X: ...' в строке {lineno}")
            left, right = [s.strip() for s in line.split(":", 1)]
            if not re.fullmatch(r"[A-Z]", left):
                error(f"Имя узла слева должно быть одной большой буквой (строка {lineno})")
            deps = []
            if right:
                for tok in right.split():
                    if not re.fullmatch(r"[A-Z]", tok):
                        error(f"Зависимость должна быть одной большой буквой (строка {lineno})")
                    deps.append(tok)
            graph[left] = deps
    for _, vs in list(graph.items()):
        for v in vs:
            graph.setdefault(v, [])
    return graph

# ========== карта репозиториев зависимостей (для real-режима) ==========

def load_deps_map(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        error("deps-map файл не найден.")
    mapping: Dict[str, str] = {}
    with p.open("r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split(None, 1)
            if len(parts) != 2:
                error(f"Некорректная строка в deps-map: '{s}' (ожидалось: 'имя URL/путь')")
            name, loc = parts
            mapping[name.lower()] = loc
    return mapping

# ========== построение графа DFS с глубиной и циклами ==========

def _norm_name(spec: str) -> str:
    return re.split(r"[<>=!;\[\s]", spec, maxsplit=1)[0].strip()

class GraphBuilder:
    def __init__(self, max_depth: int, deps_map: Dict[str, str],
                 version: Optional[str], keep_clones: bool = False):
        self.max_depth = max_depth
        self.deps_map = deps_map
        self.version = version
        self.keep_clones = keep_clones
        self.tmpdirs: List[Path] = []
        self.graph: Dict[str, List[str]] = {}
        self._resolved_cache: Dict[str, List[str]] = {}

    def _cleanup(self):
        if self.keep_clones:
            return
        for d in self.tmpdirs:
            shutil.rmtree(d, ignore_errors=True)

    def _resolve_pkg_deps(self, pkg: str) -> List[str]:
        key = pkg.lower()
        if key in self._resolved_cache:
            return self._resolved_cache[key]
        if key not in self.deps_map:
            self._resolved_cache[key] = []
            return []
        loc = self.deps_map[key]
        if is_git_url(loc):
            work = Path(tempfile.mkdtemp(prefix=f"dep_{key}_"))
            self.tmpdirs.append(work)
            root = work / "repo"
            git_clone(loc, root)
            if self.version:
                git_try_checkout_version(root, pkg, self.version)
            deps = extract_runtime_dependencies(root)
        else:
            root = Path(loc).resolve()
            if not root.is_dir():
                eprint(f"Предупреждение: путь '{loc}' для '{pkg}' недоступен.")
                deps = []
            else:
                deps = extract_runtime_dependencies(root)
        names = []
        for spec in deps:
            nm = _norm_name(spec)
            if nm:
                names.append(nm)
        self._resolved_cache[key] = names
        return names

    def dfs(self, node: str, depth: int, visit: Set[str], stack: List[str],
            cycles: List[List[str]], resolver_mode: str):
        if depth > self.max_depth:
            return
        if node in visit:
            return
        visit.add(node)
        stack.append(node)

        if resolver_mode == "test":
            neighbors = self.graph.get(node, [])
        else:
            neighbors = self._resolve_pkg_deps(node)

        self.graph.setdefault(node, [])
        for v in neighbors:
            if v not in self.graph[node]:
                self.graph[node].append(v)
            if v not in visit:
                self.dfs(v, depth + 1, visit, stack, cycles, resolver_mode)
            elif v in stack:
                # цикл: извлечём путь
                i = stack.index(v)
                cyc = stack[i:] + [v]
                cycles.append(cyc)

        stack.pop()

    def build_from_test(self, tg: Dict[str, List[str]], root: str) -> Tuple[Dict[str, List[str]], List[List[str]]]:
        self.graph = {k: list(vs) for k, vs in tg.items()}
        cycles: List[List[str]] = []
        self.dfs(root, 0, set(), [], cycles, "test")
        return self.graph, cycles

    def build_from_real(self, root_repo: Path, root_pkg: str) -> Tuple[Dict[str, List[str]], List[List[str]]]:
        root_specs = extract_runtime_dependencies(root_repo)
        root_neighbors = []
        for spec in root_specs:
            nm = _norm_name(spec)
            if nm:
                root_neighbors.append(nm)
        self.graph = {root_pkg: root_neighbors}
        cycles: List[List[str]] = []
        self.dfs(root_pkg, 0, set(), [], cycles, "real")
        return self.graph, cycles

# ========== операции над графом: печать, топосорт, порядок загрузки ==========

def print_graph(graph: Dict[str, List[str]]):
    for u, vs in graph.items():
        if not vs:
            print(f"{u}:")
        else:
            print(f"{u}: " + " ".join(vs))

def topo_sort(graph: Dict[str, List[str]]) -> Optional[List[str]]:
    indeg: Dict[str, int] = {}
    for u in graph:
        indeg.setdefault(u, 0)
        for v in graph[u]:
            indeg[v] = indeg.get(v, 0) + 1
    queue = [u for u, d in indeg.items() if d == 0]
    order: List[str] = []
    gcopy = {u: list(vs) for u, vs in graph.items()}
    while queue:
        u = queue.pop(0)
        order.append(u)
        for v in gcopy.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
        gcopy[u] = []
    if any(d > 0 for d in indeg.values()):
        return None
    return order

# ---- Таръян: СКС ----
def tarjans_scc(graph: Dict[str, List[str]]) -> List[List[str]]:
    index = 0
    indices: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    stack: List[str] = []
    onstack: Set[str] = set()
    sccs: List[List[str]] = []

    sys.setrecursionlimit(max(10000, len(graph) * 2))

    def strongconnect(v: str):
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        onstack.add(v)

        for w in graph.get(v, []):
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in onstack:
                lowlink[v] = min(lowlink[v], indices[w])

        # корень СКС?
        if lowlink[v] == indices[v]:
            comp: List[str] = []
            while True:
                w = stack.pop()
                onstack.remove(w)
                comp.append(w)
                if w == v:
                    break
            sccs.append(comp)

    for v in list(graph.keys()):
        if v not in indices:
            strongconnect(v)
    return sccs

def condense_graph(graph: Dict[str, List[str]], sccs: List[List[str]]) -> Tuple[Dict[int, List[int]], Dict[str, int]]:
    comp_id: Dict[str, int] = {}
    for i, comp in enumerate(sccs):
        for v in comp:
            comp_id[v] = i
    dag: Dict[int, List[int]] = {i: [] for i in range(len(sccs))}
    for u, vs in graph.items():
        cu = comp_id[u]
        for v in vs:
            cv = comp_id[v]
            if cu != cv and cv not in dag[cu]:
                dag[cu].append(cv)
    return dag, comp_id

def load_order(graph: Dict[str, List[str]], root: str) -> List[str]:
    """
    Порядок загрузки «листья -> ... -> корень».
    Если граф ацикличен — это просто топосорт подграфа, ограниченного достижимыми из root,
    и развернутый так, что зависимости идут раньше.
    Если есть циклы — печатаем СКС блоками вида [A B], считая, что эти узлы требуют
    совместной загрузки/разрешения.
    """
    # ограничим граф достижимыми из root
    reachable: Set[str] = set()
    def _dfs(u: str):
        if u in reachable:
            return
        reachable.add(u)
        for v in graph.get(u, []):
            _dfs(v)
    _dfs(root)
    sub = {u: [v for v in graph.get(u, []) if v in reachable] for u in reachable}

    # СКС
    sccs = tarjans_scc(sub)
    if len(sccs) == len(sub):  # ацикличен
        order = topo_sort(sub)
        return order if order else []

    dag, comp_id = condense_graph(sub, sccs)
    topo_comps = topo_sort({i: dag.get(i, []) for i in dag}) or []
    # в порядке топосорта компонент расширяем: внутри СКС сохраним детерминированность (лекс. порядок)
    result: List[str] = []
    for cid in topo_comps:
        comp = sccs[cid]
        if len(comp) == 1:
            result.append(comp[0])
        else:
            # пометим группу циклом
            group = "[" + " ".join(sorted(comp)) + "]"
            result.append(group)
    return result

# ========== сравнение с pip (по запросу) ==========

def pip_install_order(package: str, version: Optional[str]) -> List[str]:
    """
    Пытается получить порядок установки pip через --dry-run.
    Требует интернет/доступ к индексам.
    Возвращает имена пакетов (без версий).
    """
    spec = package if not version else f"{package}=={version}"
    # --ignore-installed чтобы не влияли локальные пакеты
    cmd = [sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed", spec]
    try:
        cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        out = cp.stdout.splitlines()
        collected = None
        for i, line in enumerate(out):
            # Ищем финальную строку вида "Installing collected packages: A, B, C"
            if "Installing collected packages:" in line:
                collected = line
        if not collected:
            return []
        after_colon = collected.split("Installing collected packages:", 1)[1]
        # разделение по ", " с обрезкой версий "name (x.y.z)" -> name
        parts = [p.strip() for p in after_colon.split(",")]
        names: List[str] = []
        for p in parts:
            if not p:
                continue
            # возможные форматы: "pydantic", "pydantic (2.6.3)"
            nm = p.split("(", 1)[0].strip()
            names.append(nm)
        # pip печатает порядок «зависимости раньше зависящего»
        return names
    except Exception:
        return []

def diff_orders(ours: List[str], pips: List[str]) -> Tuple[List[str], List[str], Optional[Tuple[int, str, str]]]:
    """
    Возвращает:
      - only_ours: узлы, которых нет у pip
      - only_pip: узлы, которых нет у нас
      - first_mismatch: (позиция, ours_item, pip_item) либо None
    """
    # уберём групповые [A B] как цельные токены
    ours_flat = ours[:]
    pset = set(pips)
    oset = set(ours_flat)

    only_ours = [x for x in ours_flat if x.strip("[]") not in pset]
    only_pip = [x for x in pips if x not in oset and f"[{x}]" not in oset]

    # найдём первую позицию, где различается относительный порядок общих элементов
    commons = [x for x in pips if x in oset or f"[{x}]" in oset]
    oi = {x.strip("[]"): i for i, x in enumerate(ours_flat)}
    for i, x in enumerate(commons):
        y = commons[i]
        j = oi.get(y, None)
        if j is None:
            continue
        # ищем следующий общий элемент и сравниваем относительный порядок
        if i + 1 < len(commons):
            y2 = commons[i + 1]
            j2 = oi.get(y2, None)
            if j2 is not None and not (j < j2):
                return (only_ours, only_pip, (i, y, y2))
    return (only_ours, only_pip, None)

# ========== CLI ==========

def main():
    ap = argparse.ArgumentParser(
        description="Этап 4: дополнительные операции над графом зависимостей. "
                    "Порядок загрузки, сравнение с pip, тестовый режим."
    )
    # тестовый режим
    ap.add_argument("--test-graph", help="Файл тестового графа (узлы — A..Z). Включает тестовый режим.")
    ap.add_argument("--root", help="Корневой узел (тест) или имя корневого пакета (реальный режим).")

    # реальный режим
    ap.add_argument("--package", "-p", help="Имя анализируемого пакета (реальный режим).")
    ap.add_argument("--repo", "-r", help="URL или путь к репозиторию (реальный режим).")
    ap.add_argument("--repo-mode", "-m", choices=REPO_MODE_CHOICES, help="clone|local (реальный режим).")
    ap.add_argument("--version", "-V", help="Версия корневого пакета (попытка checkout соответствующего тега).")

    # карта зависимостей для рекурсии
    ap.add_argument("--deps-map", help="Файл карты: 'name <space> url_or_path' для зависимостей.")
    ap.add_argument("--keep-clones", action="store_true", help="Не удалять временные клоны зависимостей (отладка).")

    # операции
    ap.add_argument("--max-depth", "-d", type=int, default=10, help="Максимальная глубина DFS (включая корень).")
    ap.add_argument("--print-graph", action="store_true", help="Печать списка смежности.")
    ap.add_argument("--print-cycles", action="store_true", help="Поиск и печать циклов.")
    ap.add_argument("--toposort", action="store_true", help="Печать топологической сортировки (если без циклов).")
    ap.add_argument("--load-order", action="store_true", help="Печать порядка загрузки зависимостей (листья -> корень).")
    ap.add_argument("--compare-with-pip", action="store_true", help="Сравнить порядок с pip --dry-run (только реальный режим).")

    ns = ap.parse_args()

    # ---- тестовый режим ----
    if ns.test_graph:
        tg = parse_test_graph(Path(ns.test_graph))
        root = ns.root or sorted(tg.keys())[0]
        if root not in tg:
            error(f"Узел '{root}' отсутствует в тестовом графе.")
        builder = GraphBuilder(max_depth=ns.max_depth, deps_map={}, version=None, keep_clones=False)
        graph, cycles = builder.build_from_test(tg, root)

        if ns.print_graph:
            print_graph(graph)
        if ns.print_cycles:
            for cyc in cycles:
                print("CYCLE:", " -> ".join(cyc))
        if ns.toposort:
            order = topo_sort(graph)
            print("TOPO:" if order else "TOPO: невозможно — граф содержит цикл.", end="")
            if order:
                print(" " + " ".join(order))
            else:
                print()
        if ns.load_order:
            lo = load_order(graph, root)
            print("LOAD-ORDER:", " ".join(lo))
        if not (ns.print_graph or ns.print_cycles or ns.toposort or ns.load_order):
            lo = load_order(graph, root)
            print("LOAD-ORDER:", " ".join(lo))
        return

    # ---- реальный режим ----
    if not (ns.package and ns.repo and ns.repo_mode):
        error("Для реального режима укажите --package, --repo и --repo-mode (или используйте --test-graph).")

    package = validate_package(ns.package)
    repo_arg = validate_repo(ns.repo)
    mode = validate_repo_mode(ns.repo_mode)

    deps_map = load_deps_map(ns.deps_map)
    builder = GraphBuilder(max_depth=ns.max_depth, deps_map=deps_map, version=ns.version, keep_clones=ns.keep_clones)

    workdir: Optional[Path] = None
    repo_root: Optional[Path] = None

    try:
        if mode == "clone":
            if not is_git_url(repo_arg):
                error("Для режима clone ожидается git-URL.")
            workdir = Path(tempfile.mkdtemp(prefix="stage4_"))
            repo_root = workdir / "rootrepo"
            git_clone(repo_arg, repo_root)
            if ns.version:
                git_try_checkout_version(repo_root, package, ns.version)
        else:
            p = Path(repo_arg)
            if not p.is_dir():
                error("Локальный путь к репозиторию не существует или это не каталог.")
            repo_root = p.resolve()

        declared_ver = read_declared_version(repo_root)
        if declared_ver and ns.version and declared_ver != ns.version:
            eprint(f"Предупреждение: версия в репо = {declared_ver}, а запрошена = {ns.version}.")

        graph, cycles = builder.build_from_real(repo_root, package)

        if ns.print_graph:
            print_graph(graph)
        if ns.print_cycles:
            for cyc in cycles:
                print("CYCLE:", " -> ".join(cyc))
        if ns.toposort:
            order = topo_sort(graph)
            print("TOPO:" if order else "TOPO: невозможно — граф содержит цикл.", end="")
            if order:
                print(" " + " ".join(order))
            else:
                print()

        if ns.load_order or (not (ns.print_graph or ns.print_cycles or ns.toposort)):
            lo = load_order(graph, package)
            print("LOAD-ORDER:", " ".join(lo))

        if ns.compare_with_pip:
            if not ns.version:
                eprint("Предупреждение: для сравнения с pip лучше указать --version.")
            pip_order = pip_install_order(package, ns.version)
            if not pip_order:
                print("PIP-ORDER: (не удалось получить порядок; проверьте интернет/индексы)")
            else:
                print("PIP-ORDER:", " ".join(pip_order))
                only_ours, only_pip, mismatch = diff_orders(lo, pip_order)
                if only_ours:
                    print("ONLY-OURS:", " ".join(only_ours))
                if only_pip:
                    print("ONLY-PIP:", " ".join(only_pip))
                if mismatch:
                    pos, a, b = mismatch
                    print(f"MISMATCH@{pos}: порядок различается около '{a}' vs '{b}'")

                # краткое объяснение возможных расхождений:
                print("# NOTE: Возможные причины расхождений с pip:")
                print("# - условные маркеры/экстры (например, 'platform_system == \"Windows\"') — мы их не вычисляем;")
                print("# - неполная карта --deps-map (мы не заходили в репозитории некоторых зависимостей);")
                print("# - различия версий/решения конфликтов у pip (бек-трекинг, выбор подходящих версий);")
                print("# - динамическая генерация install_requires в setup.py;")
                print("# - ограничение --max-depth.")
    finally:
        if workdir and workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)
        builder._cleanup()

if __name__ == "__main__":
    main()
