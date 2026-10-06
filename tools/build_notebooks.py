"""Construit les notebooks .ipynb à partir des sources texte (notebooks_src/*.py) puis les exécute.

Format source : cellules séparées par des lignes « #%% md » ou « #%% code ».
Usage : python tools/build_notebooks.py [nom_sans_extension ...]
"""
import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
SRC, OUT = ROOT / "notebooks_src", ROOT / "notebooks"


def parse(text: str):
    cells, kind, buf = [], None, []
    for line in text.splitlines():
        if line.strip() in ("#%% md", "#%% code"):
            if kind and "".join(buf).strip():
                cells.append((kind, "\n".join(buf).strip("\n")))
            kind, buf = line.strip().split()[1], []
        else:
            buf.append(line)
    if kind and "".join(buf).strip():
        cells.append((kind, "\n".join(buf).strip("\n")))
    return cells


def build(name: str, execute: bool = True):
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    for kind, src in parse((SRC / f"{name}.py").read_text(encoding="utf-8")):
        nb.cells.append(nbformat.v4.new_markdown_cell(src) if kind == "md" else nbformat.v4.new_code_cell(src))
    if execute:
        ExecutePreprocessor(timeout=3600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(OUT)}})
    nbformat.write(nb, OUT / f"{name}.ipynb")
    print("OK", name)


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(p.stem for p in SRC.glob("*.py"))
    for n in names:
        build(n)
