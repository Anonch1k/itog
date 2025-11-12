#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
import configparser
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

# tomllib — stdlib c Python 3.11+
try:
    import tomllib  # type: ignore
except Exception:
    tomllib = None

REPO_MODE_CHOICES = ("clone", "local")

# -------------------- утилиты и валидация --------------------

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

def git_try_checkout_version(repo_dir: Path, package: str, version: Optional[str]) -> None:
    if not version:
        return
    candidates = [f"v{version}", version, f"{package}-{version}"]
    try:
        run_git(["fetch", "--tags"], cwd=repo_dir, check=False)
        for tag in candidates:
            run_git(["fetch", "origin", "tag", tag], cwd=repo_dir, check=False)
        tags = set(t.strip() for t in run_git(["tag", "--list"], cwd=repo_dir, check=False).stdout.splitlines())
        for t in candidates:
            if t in tags:
                run_git(["checkout", "--quiet", f"refs/tags/{t}"], cwd=repo_dir, check=True)
                return
        # fallback: ветка
        for br in candidates:
            try:
                run_git(["checkout", "--quiet", br], cwd=repo_dir, check=True)
                return
            except subprocess.CalledProcessError:
                pass
    except Exception:
        pass

# -------------------- парсинг зависимостей (как в этапах 2–4) --------------------

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

# -------------------- тестовый граф A..Z --------------------

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

# -------------------- карта репозиториев для транзитивного real-режима --------------------

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

# -------------------- построение графа (DFS, циклы) --------------------

def _norm_name(spec: str) -> str:
    # "pydantic>=2.0.0; python_version >= '3.8'" -> "pydantic"
    return re.split(r"[<>=!;\[\s]", spec, maxsplit=1)[0].strip()

class GraphBuilder:
    def __init__(self, max_depth: int, deps_map: Dict[str, str], version: Optional[str], keep_clones: bool = False):
        self.max_depth = max_depth
        self.deps_map = deps_map
        self.version = version
        self.keep_clones = keep_clones
        self.tmpdirs: List[Path] = []
        self.graph: Dict[str, List[str]] = {}
        self._cache: Dict[str, List[str]] = {}

    def _cleanup(self):
        if self.keep_clones:
            return
        for d in self.tmpdirs:
            shutil.rmtree(d, ignore_errors=True)

    def _resolve_pkg_deps(self, pkg: str) -> List[str]:
        key = pkg.lower()
        if key in self._cache:
            return self._cache[key]
        if key not in self.deps_map:
            self._cache[key] = []
            return []
        loc = self.deps_map[key]
        if is_git_url(loc):
            work = Path(tempfile.mkdtemp(prefix=f"dep_{key}_"))
            self.tmpdirs.append(work)
            root = work / "repo"
            git_clone(loc, root)
            git_try_checkout_version(root, pkg, self.version)
            specs = extract_runtime_dependencies(root)
        else:
            root = Path(loc)
            if not root.is_dir():
                eprint(f"Предупреждение: путь '{loc}' для '{pkg}' недоступен.")
                specs = []
            else:
                specs = extract_runtime_dependencies(root)
        names = []
        for spec in specs:
            nm = _norm_name(spec)
            if nm:
                names.append(nm)
        self._cache[key] = names
        return names

    def dfs(self, node: str, depth: int, visit: Set[str], stack: List[str], cycles: List[List[str]], mode: str):
        if depth > self.max_depth:
            return
        if node in visit:
            return
        visit.add(node)
        stack.append(node)

        if mode == "test":
            neighbors = self.graph.get(node, [])
        else:
            neighbors = self._resolve_pkg_deps(node)

        self.graph.setdefault(node, [])
        for v in neighbors:
            if v not in self.graph[node]:
                self.graph[node].append(v)
            if v not in visit:
                self.dfs(v, depth + 1, visit, stack, cycles, mode)
            elif v in stack:
                i = stack.index(v)
                cycles.append(stack[i:] + [v])

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

# -------------------- SCC (Таръян) для подсветки циклов --------------------

def tarjans_scc(graph: Dict[str, List[str]]) -> List[List[str]]:
    index = 0
    indices: Dict[str, int] = {}
    low: Dict[str, int] = {}
    st: List[str] = []
    onst: Set[str] = set()
    comps: List[List[str]] = []

    sys.setrecursionlimit(max(8192, len(graph) * 4))

    def sc(v: str):
        nonlocal index
        indices[v] = low[v] = index
        index += 1
        st.append(v)
        onst.add(v)
        for w in graph.get(v, []):
            if w not in indices:
                sc(w)
                low[v] = min(low[v], low[w])
            elif w in onst:
                low[v] = min(low[v], indices[w])
        if low[v] == indices[v]:
            comp = []
            while True:
                w = st.pop()
                onst.remove(w)
                comp.append(w)
                if w == v:
                    break
            comps.append(comp)

    for v in list(graph.keys()):
        if v not in indices:
            sc(v)
    return comps

# -------------------- экспорт в DOT и рендер в SVG --------------------

def _dot_id(s: str) -> str:
    # безопасный идентификатор: в DOT лучше использовать в кавычках label и идентификатор одинаково
    return '"' + s.replace('"', '\\"') + '"'

def graph_to_dot(graph: Dict[str, List[str]],
                 root: str,
                 sccs: List[List[str]],
                 rankdir: str = "LR",
                 title: Optional[str] = None) -> str:
    # Индексация узлов по СКС
    scc_by_node: Dict[str, int] = {}
    for i, comp in enumerate(sccs):
        for v in comp:
            scc_by_node[v] = i

    lines: List[str] = []
    lines.append("digraph deps {")
    lines.append(f'  rankdir={rankdir};')
    lines.append('  node [shape=box, style="rounded,filled", fillcolor="#f8f9fb", color="#9aa0a6", fontname="Arial"];')
    lines.append('  edge [color="#9aa0a6"];')
    if title:
        lines.append(f'  labelloc="t"; label={_dot_id(title)}; fontsize=18;')

    # Подсветка корня
    lines.append(f'  {_dot_id(root)} [shape=box, fillcolor="#e8f0fe", color="#5f6368", penwidth=1.5];')

    # Рёбра
    for u, vs in graph.items():
        if not vs:
            # одиночный узел
            lines.append(f'  {_dot_id(u)};')
        else:
            for v in vs:
                lines.append(f'  {_dot_id(u)} -> {_dot_id(v)};')

    # Кластеры для СКС (циклы): только если размер > 1
    for i, comp in enumerate(sccs):
        if len(comp) <= 1:
            continue
        lines.append(f'  subgraph cluster_scc_{i} {{')
        lines.append('    style="rounded,dashed"; color="#d93025"; label="cycle"; fontsize=10;')
        for v in comp:
            lines.append(f'    {_dot_id(v)} [fillcolor="#fdecea", color="#d93025"];')
        lines.append('  }')

    lines.append("}")
    return "\n".join(lines)

def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def render_svg(dot_path: Path, svg_path: Path) -> bool:
    try:
        cp = subprocess.run(["dot", "-Kdot", "-Tsvg", str(dot_path), "-o", str(svg_path)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if cp.returncode != 0:
            eprint("Graphviz 'dot' вернул ошибку:\n" + (cp.stderr or cp.stdout))
            return False
        return True
    except FileNotFoundError:
        eprint("Не найден исполняемый файл 'dot'. Установите Graphviz (например, `sudo apt-get install graphviz`).")
        return False

# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser(
        description="Этап 5: визуализация графа зависимостей в Graphviz DOT и SVG."
    )
    # тестовый режим
    ap.add_argument("--test-graph", help="Файл тестового графа (узлы — A..Z). Включает тестовый режим.")
    ap.add_argument("--root", help="Корневой узел (тест) или имя корневого пакета (реальный режим).")

    # реальный режим
    ap.add_argument("--package", "-p", help="Имя анализируемого пакета (реальный режим).")
    ap.add_argument("--repo", "-r", help="URL или путь к репозиторию (реальный режим).")
    ap.add_argument("--repo-mode", "-m", choices=REPO_MODE_CHOICES, help="clone|local (реальный режим).")
    ap.add_argument("--version", "-V", help="Версия корневого пакета (попытка checkout соответствующего тега).")

    # карта зависимостей
    ap.add_argument("--deps-map", help="Файл карты зависимостей: 'name URL_or_path'. Для транзитивности в реальном режиме.")
    ap.add_argument("--keep-clones", action="store_true", help="Не удалять временные клоны зависимостей (отладка).")

    # настройки графа/вывода
    ap.add_argument("--max-depth", "-d", type=int, default=10, help="Максимальная глубина DFS (включая корень).")
    ap.add_argument("--rankdir", choices=("LR", "TB", "BT", "RL"), default="LR", help="Ориентация графа.")
    ap.add_argument("--dot-out", required=True, help="Куда сохранить DOT (.dot).")
    ap.add_argument("--svg-out", required=True, help="Куда сохранить SVG (.svg).")
    ap.add_argument("--title", help="Заголовок диаграммы (опционально).")

    ns = ap.parse_args()

    # ---- тестовый режим ----
    if ns.test_graph:
        tg = parse_test_graph(Path(ns.test_graph))
        root = ns.root or sorted(tg.keys())[0]
        if root not in tg:
            error(f"Узел '{root}' отсутствует в тестовом графе.")
        builder = GraphBuilder(max_depth=ns.max_depth, deps_map={}, version=None, keep_clones=False)
        graph, _cycles = builder.build_from_test(tg, root)
        sccs = tarjans_scc(graph)
        title = ns.title or f"Test Graph (root={root})"
        dot = graph_to_dot(graph, root, sccs, rankdir=ns.rankdir, title=title)
        dot_path = Path(ns.dot_out)
        svg_path = Path(ns.svg_out)
        write_text(dot_path, dot)
        ok = render_svg(dot_path, svg_path)
        if ok:
            print(f"DOT: {dot_path}")
            print(f"SVG: {svg_path}")
        return

    # ---- реальный режим ----
    if not (ns.package and ns.repo and ns.repo_mode):
        error("Для реального режима укажите --package, --repo и --repo-mode (или используйте --test-graph).")

    package = validate_package(ns.package)
    repo_arg = validate_repo(ns.repo)
    repo_mode = validate_repo_mode(ns.repo_mode)
    deps_map = load_deps_map(ns.deps_map) if ns.deps_map else {}
    builder = GraphBuilder(max_depth=ns.max_depth, deps_map=deps_map, version=ns.version, keep_clones=ns.keep_clones)

    workdir: Optional[Path] = None
    repo_root: Optional[Path] = None

    try:
        if repo_mode == "clone":
            if not is_git_url(repo_arg):
                error("Для режима clone ожидается git-URL.")
            workdir = Path(tempfile.mkdtemp(prefix="stage5_"))
            repo_root = workdir / "rootrepo"
            git_clone(repo_arg, repo_root)
            git_try_checkout_version(repo_root, package, ns.version)
        else:
            p = Path(repo_arg)
            if not p.is_dir():
                error("Локальный путь к репозиторию не существует или это не каталог.")
            repo_root = p.resolve()

        declared_ver = read_declared_version(repo_root)
        if declared_ver and ns.version and declared_ver != ns.version:
            eprint(f"Предупреждение: версия в репо = {declared_ver}, а запрошена = {ns.version}.")

        graph, _cycles = builder.build_from_real(repo_root, package)
        sccs = tarjans_scc(graph)
        title = ns.title or (f"{package} dependencies" + (f" (v{ns.version})" if ns.version else ""))

        dot = graph_to_dot(graph, package, sccs, rankdir=ns.rankdir, title=title)
        dot_path = Path(ns.dot_out)
        svg_path = Path(ns.svg_out)
        write_text(dot_path, dot)
        ok = render_svg(dot_path, svg_path)
        if ok:
            print(f"DOT: {dot_path}")
            print(f"SVG: {svg_path}")
    finally:
        if workdir and workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)
        builder._cleanup()

if __name__ == "__main__":
    main()
