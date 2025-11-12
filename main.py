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

# ========== общие утилиты/валидация ==========

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

def validate_version(ver: str) -> str:
    if not ver:
        error("Версия пакета не может быть пустой.")
    return ver

# ========== git ==========

def is_git_url(s: str) -> bool:
    return s.endswith(".git") or s.startswith(("git@", "ssh://", "https://", "http://"))

def run_git(args, cwd=None, check=True) -> subprocess.CompletedProcess:
    return subprocess.run(["git"] + args, cwd=cwd, check=check,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def git_clone(repo_url: str, dest: Path) -> Path:
    # Без --depth=1, чтобы checkout тега гарантированно работал
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

# ========== парсинг зависимостей pip-пакетов (как в Этапе 2) ==========

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

# ========== тестовый режим: парсер текстового графа ==========

def parse_test_graph(path: Path) -> Dict[str, List[str]]:
    """
    Формат строк:
      A: B C D
      B: D
      C:
    Узлы — большие латинские буквы. Пробелы/пустые строки/комментарии '#' допускаются.
    """
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
    # убедимся, что все упомянутые зависимые узлы есть в графе (если не объявлены отдельно — добавим пустыми)
    for u, vs in list(graph.items()):
        for v in vs:
            graph.setdefault(v, [])
    return graph

# ========== чтение карты репозиториев ==========

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

# ========== построение графа DFS с рекурсией и ограничением глубины ==========

class GraphBuilder:
    def __init__(self,
                 max_depth: int,
                 deps_map: Dict[str, str],
                 version: Optional[str],
                 keep_clones: bool = False):
        self.max_depth = max_depth
        self.deps_map = deps_map
        self.version = version
        self.keep_clones = keep_clones
        self.tmpdirs: List[Path] = []
        self.graph: Dict[str, List[str]] = {}
        # кэш, чтобы не парсить один и тот же пакет несколько раз
        self._resolved_deps_cache: Dict[str, List[str]] = {}

    def _cleanup(self):
        if self.keep_clones:
            return
        for d in self.tmpdirs:
            shutil.rmtree(d, ignore_errors=True)

    # ---- извлечение прямых deps для реального пакета (по карте реп) ----
    def _resolve_package_deps(self, pkg: str) -> List[str]:
        key = pkg.lower()
        if key in self._resolved_deps_cache:
            return self._resolved_deps_cache[key]

        if key not in self.deps_map:
            # нет источника — считаем, что неизвестно (пусто)
            self._resolved_deps_cache[key] = []
            return []

        loc = self.deps_map[key]
        # локальный путь или git
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
            if not root.exists() or not root.is_dir():
                eprint(f"Предупреждение: путь '{loc}' для '{pkg}' недоступен — пропускаю.")
                deps = []
            else:
                deps = extract_runtime_dependencies(root)

        # нормализуем имена: срезаем опции (версии/маркеры), оставляем «имя» до первого запрещённого символа
        normalized = []
        for spec in deps:
            # пример: "pydantic>=2.0.0; python_version >= '3.8'"
            name = re.split(r"[<>=!;\[\s]", spec, maxsplit=1)[0]
            if name:
                normalized.append(name.strip())
        self._resolved_deps_cache[key] = normalized
        return normalized

    # ---- DFS рекурсией ----
    def dfs(self, node: str, depth: int, visit: Set[str], stack: Set[str],
            detect_cycles: List[List[str]], resolver_mode: str):
        if depth > self.max_depth:
            return
        visit.add(node)
        stack.add(node)

        # получаем прямые зависимости
        if resolver_mode == "test":
            neighbors = self.graph.get(node, [])
        else:
            neighbors = self._resolve_package_deps(node)

        # создаём запись в графе (даже если пусто)
        self.graph.setdefault(node, [])
        for v in neighbors:
            if v not in self.graph[node]:
                self.graph[node].append(v)

            if v not in visit:
                self.dfs(v, depth + 1, visit, stack, detect_cycles, resolver_mode)
            elif v in stack:
                # цикл найден — извлечём путь
                cycle = self._extract_cycle_path(start=v, stack_order=list(stack))
                if cycle:
                    detect_cycles.append(cycle)

        stack.remove(node)

    @staticmethod
    def _extract_cycle_path(start: str, stack_order: List[str]) -> List[str]:
        # stack_order — порядок входа (в set порядка нет, поэтому сюда передают list, который строится извлекателем)
        if start not in stack_order:
            return []
        i = stack_order.index(start)
        path = stack_order[i:] + [start]
        return path

    def build_from_test(self, test_graph: Dict[str, List[str]], root: str) -> Tuple[Dict[str, List[str]], List[List[str]]]:
        # загрузим тестовый граф, затем прогон DFS
        self.graph = {k: list(vs) for k, vs in test_graph.items()}
        cycles: List[List[str]] = []
        self.dfs(root, 0, set(), set(), cycles, resolver_mode="test")
        return self.graph, cycles

    def build_from_real(self, root_repo: Path, root_pkg: str) -> Tuple[Dict[str, List[str]], List[List[str]]]:
        # корневые зависимости:
        root_deps = extract_runtime_dependencies(root_repo)
        # нормализуем имена
        root_neighbors = []
        for spec in root_deps:
            name = re.split(r"[<>=!;\[\s]", spec, maxsplit=1)[0]
            if name:
                root_neighbors.append(name.strip())

        self.graph = {root_pkg: root_neighbors}
        cycles: List[List[str]] = []
        self.dfs(root_pkg, 0, set(), set(), cycles, resolver_mode="real")
        return self.graph, cycles

# ========== вывод/операции ==========

def print_graph(graph: Dict[str, List[str]]):
    for u, vs in graph.items():
        if not vs:
            print(f"{u}:")
        else:
            print(f"{u}: " + " ".join(vs))

def reachable_nodes(graph: Dict[str, List[str]], root: str) -> List[str]:
    seen: Set[str] = set()
    def _dfs(u: str):
        if u in seen:
            return
        seen.add(u)
        for v in graph.get(u, []):
            _dfs(v)
    _dfs(root)
    return sorted(seen)

def topo_sort(graph: Dict[str, List[str]]) -> Optional[List[str]]:
    indeg: Dict[str, int] = {u: 0 for u in graph}
    for u, vs in graph.items():
        for v in vs:
            indeg[v] = indeg.get(v, 0) + 1
            indeg.setdefault(u, 0)
    # Kahn
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
    # если остались рёбра, был цикл
    if any(indeg[u] > 0 for u in indeg):
        return None
    return order

# ========== CLI ==========

def main():
    ap = argparse.ArgumentParser(
        description="Этап 3: построение графа зависимостей (DFS, глубина, циклы) и базовые операции. "
                    "Поддерживается тестовый режим с файлом графа."
    )
    # Режимы
    ap.add_argument("--test-graph", help="Файл тестового графа (узлы — A..Z). Включает тестовый режим.")
    ap.add_argument("--root", help="Корневой узел (тестовый режим) или имя корневого пакета (реальный режим).", required=False)

    # Реальный режим (как в этапе 2)
    ap.add_argument("--package", "-p", help="Имя анализируемого пакета (реальный режим).")
    ap.add_argument("--repo", "-r", help="URL или путь к репозиторию (реальный режим).")
    ap.add_argument("--repo-mode", "-m", choices=REPO_MODE_CHOICES, help="clone|local (реальный режим).")
    ap.add_argument("--version", "-V", help="Версия корневого пакета (попытка checkout соответствующего тега).")

    # Карта репозиториев зависимостей для рекурсивного обхода
    ap.add_argument("--deps-map", help="Путь к файлу карты зависимостей: 'name <space> url_or_path' (для рекурсивного real-режима).")
    ap.add_argument("--keep-clones", action="store_true", help="Не удалять временные клоны зависимостей (для отладки).")

    # Управление/операции
    ap.add_argument("--max-depth", "-d", type=int, default=10, help="Максимальная глубина DFS (включая корень).")
    ap.add_argument("--print-graph", action="store_true", help="Печать графа (adjacency list).")
    ap.add_argument("--print-cycles", action="store_true", help="Поиск и печать циклов.")
    ap.add_argument("--toposort", action="store_true", help="Печать топологической сортировки (если граф ацикличен).")

    ns = ap.parse_args()

    # ---- тестовый режим ----
    if ns.test_graph:
        tg = parse_test_graph(Path(ns.test_graph))
        if not ns.root:
            # если не задан корень, возьмём лексикографически первый узел
            ns.root = sorted(tg.keys())[0]
        root = ns.root
        if root not in tg:
            error(f"Узел '{root}' отсутствует в тестовом графе.")
        builder = GraphBuilder(max_depth=ns.max_depth, deps_map={}, version=None, keep_clones=False)
        graph, cycles = builder.build_from_test(tg, root)
        # операции
        if ns.print_graph:
            print_graph(graph)
        if ns.print_cycles:
            for cyc in cycles:
                print("CYCLE:", " -> ".join(cyc))
        if ns.toposort:
            order = topo_sort(graph)
            if order is None:
                print("TOPO: невозможно — граф содержит цикл.")
            else:
                print("TOPO:", " ".join(order))
        # по умолчанию — выведем достижимые узлы
        if not (ns.print_graph or ns.print_cycles or ns.toposort):
            print("REACHABLE:", " ".join(reachable_nodes(graph, root)))
        return

    # ---- реальный режим ----
    if not (ns.package and ns.repo and ns.repo_mode):
        error("Для реального режима укажите --package, --repo и --repo-mode (или используйте --test-graph).")

    package = validate_package(ns.package)
    repo_arg = validate_repo(ns.repo)
    mode = validate_repo_mode(ns.repo_mode)
    version = ns.version

    deps_map = load_deps_map(ns.deps_map)
    builder = GraphBuilder(max_depth=ns.max_depth, deps_map=deps_map, version=version, keep_clones=ns.keep_clones)

    workdir: Optional[Path] = None
    repo_root: Optional[Path] = None

    try:
        if mode == "clone":
            if not is_git_url(repo_arg):
                error("Для режима clone ожидается git-URL (https://.../.git, ssh://..., git@...).")
            workdir = Path(tempfile.mkdtemp(prefix="stage3_"))
            repo_root = workdir / "rootrepo"
            git_clone(repo_arg, repo_root)
            if version:
                git_try_checkout_version(repo_root, package, version)
        else:
            p = Path(repo_arg)
            if not p.exists() or not p.is_dir():
                error("Локальный путь к репозиторию не существует или это не каталог.")
            repo_root = p.resolve()

        declared_ver = read_declared_version(repo_root)
        if declared_ver and version and declared_ver != version:
            eprint(f"Предупреждение: версия в репо = {declared_ver}, а запрошена = {version}.")

        graph, cycles = builder.build_from_real(repo_root, package)

        # операции
        if ns.print_graph:
            print_graph(graph)
        if ns.print_cycles:
            for cyc in cycles:
                print("CYCLE:", " -> ".join(cyc))
        if ns.toposort:
            order = topo_sort(graph)
            if order is None:
                print("TOPO: невозможно — граф содержит цикл.")
            else:
                print("TOPO:", " ".join(order))
        if not (ns.print_graph or ns.print_cycles or ns.toposort):
            print("REACHABLE:", " ".join(reachable_nodes(graph, package)))

    finally:
        # чистим корневой клон
        if workdir and workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)
        # чистим клоны зависимостей
        builder._cleanup()

if __name__ == "__main__":
    main()
