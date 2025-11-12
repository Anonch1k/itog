#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from urllib.parse import urlparse

REPO_MODE_CHOICES = ("clone", "local", "skip")


def error(msg: str):
    print(f"Ошибка: {msg}", file=sys.stderr)
    sys.exit(1)


def validate_package(name: str) -> str:
    if not name:
        error("Имя пакета не может быть пустым.")
    if any(c in name for c in " /\\:*?\"<>|"):
        error("Имя пакета содержит недопустимые символы.")
    return name


def validate_repo(value: str) -> str:
    if value is None:
        return value

    # URL
    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc:
        return value

    # Локальный путь
    if os.path.exists(value):
        return value

    error("Репозиторий должен быть существующим путём или корректным URL.")


def validate_repo_mode(mode: str) -> str:
    if mode not in REPO_MODE_CHOICES:
        error(f"Режим репозитория должен быть {REPO_MODE_CHOICES}")
    return mode


def validate_version(ver: str) -> str:
    if not ver:
        error("Версия пакета не может быть пустой.")
    return ver


def validate_image_file(path: str) -> str:
    if "." not in path:
        error("Имя файла изображения должно иметь расширение (.png/.jpg/.svg).")
    return path


def validate_depth(value: str) -> int:
    try:
        num = int(value)
    except ValueError:
        error("Глубина должна быть целым числом.")
    if num < 0:
        error("Глубина не может быть отрицательной.")
    return num


def main():
    parser = argparse.ArgumentParser(description="Минимальный прототип этапа 1")

    parser.add_argument("--package", "-p", required=True, help="Имя анализируемого пакета")
    parser.add_argument("--repo", "-r", help="URL или путь к тестовому репозиторию")
    parser.add_argument("--repo-mode", "-m", required=True, help="Режим работы репозитория", choices=REPO_MODE_CHOICES)
    parser.add_argument("--version", "-V", required=True, help="Версия пакета")
    parser.add_argument("--image-file", "-o", required=True, help="Имя файла с изображением графа")
    parser.add_argument("--max-depth", "-d", required=True, help="Максимальная глубина", type=str)

    ns = parser.parse_args()

    config = {
        "package": validate_package(ns.package),
        "repo": validate_repo(ns.repo) if ns.repo else None,
        "repo_mode": validate_repo_mode(ns.repo_mode),
        "version": validate_version(ns.version),
        "image_file": validate_image_file(ns.image_file),
        "max_depth": validate_depth(ns.max_depth)
    }

    # Этап 1: вывести все параметры в формате ключ=значение
    for k, v in config.items():
        print(f"{k}={v}")


if __name__ == "__main__":
    main()
