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

# tomllib есть в Python 3.11+. Для 3.10 и ниже можно попробовать tomli, но сторонние либы запрещены,
# поэтому просто отключаем TOML-парсер, если stdlib-модуль недоступен.
try:
    import tomllib  # Python 3.11+
except Exception:  # pragma: no cover
    tomllib = None

REPO_MODE_CHOICES = ("clone", "local")

# ------------------------- утилиты ошибок и валидации -------------------------

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
        error("Для этапа 2 требуется указать URL/путь к репозиторию (--repo).")
    parsed = urlparse(value)
    # URL (в т.ч. git+https и ssh)
    if parsed.scheme and parsed.netloc:
        return value
    # Локальный путь
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

# ------------------------- работа с git -------------------------

def is_git_url(s: str) -> bool:
    # Простейшие признаки git-URL
    return s.endswith(".git") or s.startswith(("git@", "ssh://", "https://", "http://"))

def run_git(args, cwd=None, check=True) -> subprocess.CompletedProcess:
    return subprocess.run(["git"] + args, cwd=cwd, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def git_clone(repo_url: str, dest: Path) -> Path:
    try:
        run_git(["clone", "--depth", "1", repo_url, str(dest)])
    except subprocess.CalledProcessError as ex:
        error(f"git clone не удался: {ex.stderr.strip() or ex.stdout.strip()}")
    return dest

def git_try_checkout_version(repo_dir: Path, package: str, version: str) -> None:
    """
    Пробуем найти тег для версии и сделать checkout.
    Паттерны: v{ver}, {ver}, {package}-{ver}
    """
    tag_candidates = [f"v{version}", version, f"{package}-{version}"]
    try:
        # Подгрузим все теги, если с --depth=1
        run_git(["fetch", "--tags"], cwd=repo_dir, check=False)
        out = run_git(["tag", "--list"], cwd=repo_dir, check=False).stdout.splitlines()
        existing = set(t.strip() for t in out if t.strip())
        for tag in tag_candidates:
            if tag in existing:
                # checkout в детачнутую HEAD на тег
                run_git(["checkout", "--quiet", f"tags/{tag}"], cwd=repo_dir, check=True)
                return
        # Если точного тега нет — оставляем как есть (default branch)
    except Exception:
        # Не считаем фатальной ошибкой
        pass

# ------------------------- парсинг зависимостей -------------------------

def _normalize_lines(raw: str) -> list[str]:
    lines = []
    for line in raw.splitlines():
        # удалим комментарии и хвостовые пробелы
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        # удалим инлайн-комментарии, если отделены пробелом
        s = re.split(r"\s+#", s, maxsplit=1)[0].strip()
        if s:
            lines.append(s)
    return lines

def parse_pyproject_dependencies(root: Path) -> list[str] | None:
    py = root / "pyproject.toml"
    if not py.is_file():
        return None
    if tomllib is None:
        # Нельзя парсить pyproject без tomllib — пропустим
        return None
    try:
        data = tomllib.loads(py.read_text(encoding="utf-8"))
        project = data.get("project") or {}
        deps = project.get("dependencies")
        if deps and isinstance(deps, list):
            return [d.strip() for d in deps if isinstance(d, str) and d.strip()]
        # Иногда зависимости динамические: пропустим — попробуем другие файлы
        return []
    except Exception:
        return None  # пусть другие парсеры попробуют

def parse_setup_cfg_dependencies(root: Path) -> list[str] | None:
    cfg = root / "setup.cfg"
    if not cfg.is_file():
        return None
    cp = configparser.ConfigParser()
    try:
        cp.read(cfg, encoding="utf-8")
        if cp.has_section("options") and cp.has_option("options", "install_requires"):
            raw = cp.get("options", "install_requires")
            # Значение может быть многострочным
            return _normalize_lines(raw)
        return []
    except Exception:
        return None

def _literal_eval_node(node):
    """Осторожная попытка извлечь литералы из AST."""
    try:
        return ast.literal_eval(node)
    except Exception:
        return None

def parse_setup_py_dependencies(root: Path) -> list[str] | None:
    sp = root / "setup.py"
    if not sp.is_file():
        return None
    try:
        tree = ast.parse(sp.read_text(encoding="utf-8"))
        deps: list[str] = []
        class Finder(ast.NodeVisitor):
            def visit_Call(self, node: ast.Call):
                # Ищем вызов setup(...)
                fn = node.func
                name = None
                if isinstance(fn, ast.Name):
                    name = fn.id
                elif isinstance(fn, ast.Attribute):
                    name = fn.attr
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

def parse_requirements_txt(root: Path) -> list[str] | None:
    for fname in ("requirements.txt", "requirements.in"):
        f = root / fname
        if f.is_file():
            lines = []
            for s in _normalize_lines(f.read_text(encoding="utf-8")):
                # игнорируем include'ы и опции pip (нам нельзя пользоваться менеджерами)
                if s.startswith(("-r ", "--requirement", "-c ", "--constraint")):
                    continue
                lines.append(s)
            return lines
    return None

def extract_direct_dependencies(repo_root: Path) -> list[str]:
    """
    Возвращает список прямых зависимостей (строки спецификаций).
    Порядок приоритета источников:
      1) pyproject.toml ([project].dependencies)
      2) setup.cfg ([options] install_requires)
      3) setup.py (install_requires)
      4) requirements.txt / requirements.in (как крайний fallback)
    """
    # pyproject
    deps = parse_pyproject_dependencies(repo_root)
    if deps is not None:
        return deps
    # setup.cfg
    deps = parse_setup_cfg_dependencies(repo_root)
    if deps is not None:
        return deps
    # setup.py
    deps = parse_setup_py_dependencies(repo_root)
    if deps is not None:
        return deps
    # requirements*
    deps = parse_requirements_txt(repo_root)
    if deps is not None:
        return deps
    # Ничего не нашли
    return []

# ------------------------- проверка версии (информативно) -------------------------

def read_declared_version(repo_root: Path) -> str | None:
    """
    Пытаемся прочитать версию пакета для информативной сверки.
    """
    # pyproject.toml
    if tomllib is not None:
        py = repo_root / "pyproject.toml"
        if py.is_file():
            try:
                data = tomllib.loads(py.read_text(encoding="utf-8"))
                project = data.get("project") or {}
                ver = project.get("version")
                if isinstance(ver, str):
                    return ver.strip() or None
            except Exception:
                pass
    # setup.cfg
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
    # setup.py (очень приблизительно)
    sp = repo_root / "setup.py"
    if sp.is_file():
        try:
            tree = ast.parse(sp.read_text(encoding="utf-8"))
            class VerFinder(ast.NodeVisitor):
                version = None
                def visit_Call(self, node: ast.Call):
                    fn = node.func
                    name = None
                    if isinstance(fn, ast.Name):
                        name = fn.id
                    elif isinstance(fn, ast.Attribute):
                        name = fn.attr
                    if name == "setup":
                        for kw in node.keywords:
                            if kw.arg == "version":
                                val = _literal_eval_node(kw.value)
                                if isinstance(val, str) and val.strip():
                                    self.version = val.strip()
                    self.generic_visit(node)
            vf = VerFinder()
            vf.visit(tree)
            return vf.version
        except Exception:
            pass
    return None

# ------------------------- CLI -------------------------

def main():
    parser = argparse.ArgumentParser(description="Этап 2: сбор прямых зависимостей из репозитория Python-пакета (pip-формат).")
    parser.add_argument("--package", "-p", required=True, help="Имя анализируемого пакета")
    parser.add_argument("--repo", "-r", required=True, help="URL или путь к репозиторию")
    parser.add_argument("--repo-mode", "-m", required=True, choices=REPO_MODE_CHOICES, help="Режим работы с репозиторием: clone или local")
    parser.add_argument("--version", "-V", required=True, help="Требуемая версия пакета (попробуем переключиться на соответствующий тег)")
    # Параметры ниже пока не используются на Этапе 2, оставлены для совместимости с Этапом 1/будущими этапами
    parser.add_argument("--image-file", "-o", help="Имя файла с изображением графа (будущие этапы)")
    parser.add_argument("--max-depth", "-d", type=int, help="Максимальная глубина (будущие этапы)")

    ns = parser.parse_args()

    package = validate_package(ns.package)
    repo_arg = validate_repo(ns.repo)
    mode = validate_repo_mode(ns.repo_mode)
    version = validate_version(ns.version)

    workdir: Path | None = None
    repo_root: Path | None = None

    try:
        if mode == "clone":
            if not is_git_url(repo_arg):
                error("Для режима clone ожидается git-URL (например, https://.../.git или ssh://..., git@...).")
            workdir = Path(tempfile.mkdtemp(prefix="depstage2_"))
            repo_root = workdir / "repo"
            git_clone(repo_arg, repo_root)
            git_try_checkout_version(repo_root, package, version)
        else:  # local
            p = Path(repo_arg)
            if not p.exists() or not p.is_dir():
                error("Локальный путь к репозиторию не существует или это не каталог.")
            repo_root = p.resolve()

        # Информативная сверка версии (не критично)
        declared_ver = read_declared_version(repo_root)
        if declared_ver and declared_ver != version:
            # Не прерываем выполнение — требования этапа не требуют строгой проверки соответствия.
            eprint(f"Предупреждение: версия в репо = {declared_ver}, а запрошена = {version}.")

        deps = extract_direct_dependencies(repo_root)

        # ЭТАП 2: печатаем только прямые зависимости по одной на строку
        for d in deps:
            print(d)

        # Если ничего не найдено — это тоже валидный результат (пустой вывод), но дадим подсказку в stderr.
        if not deps:
            eprint("Прямые зависимости не найдены в pyproject.toml/setup.cfg/setup.py/requirements*. Возможно, пакет не имеет зависимостей или использует динамическую генерацию.")

    finally:
        if workdir and workdir.exists():
            # аккуратно удалим временную папку
            shutil.rmtree(workdir, ignore_errors=True)

if __name__ == "__main__":
    main()
