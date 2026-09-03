"""Coherencia entre lo que el repo declara y lo que el repo tiene.

Los tres fallos mas caros de este proyecto fueron la misma cosa: estado que no
coincide con lo declarado, sin aviso. Un indice caducado que seguia cargando
porque el nombre del fichero cuadraba aunque el texto ya no; un `ablation.json`
publicado desde un arbol sucio, con la procedencia diciendolo y nadie mirando; y
un commit cuyo Dockerfile copiaba `README.md` y `LICENSE` mientras su propio
arbol no los tenia, de modo que la imagen que ese commit presentaba no podia
construirse.

Ninguno rompia nada al ejecutarse. Por eso viven aqui y no en una revision a
ojo: una declaracion que apunta al vacio solo se detecta comprobandola.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from compliance_mcp.config import project_root

ROOT = project_root()

# Claves de config.yaml que nombran ficheros de ENTRADA, versionados. Las rutas
# de salida (records_path, index_dir, los output_path de evaluacion) quedan
# fuera a proposito: son artefactos regenerables y exigir su presencia haria
# fallar la suite en un clon limpio.
INPUT_PATH_KEYS = [
    "corpus.catalog_path",
    "corpus.baseline_profiles.low",
    "corpus.baseline_profiles.moderate",
    "corpus.baseline_profiles.high",
    "corpus.baseline_profiles.privacy",
    "evaluation.golden_set_path",
]

READMES = ["README.md", "README.es.md"]


def missing(declared: list[tuple[str, str]]) -> list[str]:
    """Filtra las rutas que no existen, conservando quien las declaro."""
    return [
        f"{path}  (declarado en {source})"
        for source, path in declared
        if not (ROOT / path).exists()
    ]


def test_las_rutas_de_entrada_de_config_existen(config):
    declared = [("config.yaml:" + key, config.get(key)) for key in INPUT_PATH_KEYS]
    assert not missing(declared)


def test_los_copy_del_dockerfile_existen():
    """Un COPY a un fichero ausente rompe el build, pero solo al construir.

    Es el fallo que tuvo un commit de este repo: el Dockerfile que introducia
    copiaba README.md y LICENSE, y en su arbol ninguno de los dos existia.
    """
    dockerfile = ROOT / "Dockerfile"
    if not dockerfile.exists():
        pytest.skip("no hay Dockerfile")
    declared: list[tuple[str, str]] = []
    for line in re.findall(r"^COPY\s+(?!--from)(.*)$", dockerfile.read_text(), re.MULTILINE):
        parts = re.sub(r"--\w[\w-]*=\S+\s*", "", line).split()
        # El ultimo token es el destino dentro de la imagen, no una ruta local.
        declared += [("Dockerfile", p) for p in parts[:-1]]
    assert declared, "no se parseo ningun COPY: el parser esta roto, no el Dockerfile"
    assert not missing(declared)


def test_las_rutas_del_compose_existen():
    """`env_file` se comprueba contra su plantilla.

    El fichero de entorno real no se versiona y no debe versionarse: lleva
    claves. Lo que si tiene que existir es la plantilla que el repo promete, o
    quien clone no sabe que variables hacen falta.
    """
    compose = ROOT / "docker-compose.yml"
    if not compose.exists():
        pytest.skip("no hay docker-compose.yml")
    data = yaml.safe_load(compose.read_text())
    declared: list[tuple[str, str]] = []
    for name, service in (data.get("services") or {}).items():
        build = service.get("build")
        if isinstance(build, dict):
            declared.append((f"docker-compose.yml:{name}.build.context", build["context"]))
            if "dockerfile" in build:
                ctx = build["context"].rstrip("/")
                declared.append(
                    (f"docker-compose.yml:{name}.build.dockerfile", f"{ctx}/{build['dockerfile']}")
                )
        elif isinstance(build, str):
            declared.append((f"docker-compose.yml:{name}.build", build))
        env_file = service.get("env_file")
        for entry in [env_file] if isinstance(env_file, str) else (env_file or []):
            path = entry if isinstance(entry, str) else entry["path"]
            if not (ROOT / path).exists():
                declared.append((f"docker-compose.yml:{name}.env_file (plantilla)", f"{path}.example"))
    assert not missing(declared)


def test_las_rutas_de_pyproject_existen():
    data = (ROOT / "pyproject.toml").read_text()
    declared: list[tuple[str, str]] = []
    readme = re.search(r'^readme\s*=\s*"(.+?)"', data, re.MULTILINE)
    if readme:
        declared.append(("pyproject.toml:readme", readme.group(1)))
    packages = re.search(r"^packages\s*=\s*\[(.+?)\]", data, re.MULTILINE | re.DOTALL)
    if packages:
        declared += [
            ("pyproject.toml:packages", p) for p in re.findall(r'"(.+?)"', packages.group(1))
        ]
    assert declared, "no se leyo ninguna ruta de pyproject.toml: el parser esta roto"
    assert not missing(declared)


def test_los_objetivos_make_que_documentan_los_readme_existen():
    targets = set(re.findall(r"^([a-z][\w-]*):", (ROOT / "Makefile").read_text(), re.MULTILINE))
    unknown: list[str] = []
    for name in READMES:
        text = (ROOT / name).read_text()
        for match in re.findall(r"(?:^|\s)make ((?:[a-z][\w-]*)(?: [a-z][\w-]*)*)", text, re.MULTILINE):
            unknown += [f"make {t}  (documentado en {name})" for t in match.split() if t not in targets]
    assert not unknown


def test_los_enlaces_relativos_de_los_readme_existen():
    declared: list[tuple[str, str]] = []
    for name in READMES:
        for target in re.findall(r"\]\(([^)]+)\)", (ROOT / name).read_text()):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            declared.append((name, target.split("#")[0]))
    assert not missing(declared)


# ------------------------------------------------------------------ procedencia


def tracked_derived_artifacts() -> list[Path]:
    """Solo lo versionado. Un artefacto local recien generado sobre un arbol en
    curso puede llevar `-dirty` legitimamente; lo que no puede es estar
    publicado asi."""
    result = subprocess.run(
        ["git", "ls-files", "data/derived"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("no es un repositorio git")
    return [ROOT / p for p in result.stdout.split() if p.endswith(".json")]


def git_shas(node, trail="") -> list[tuple[str, str]]:
    """Todos los git_sha del documento, incluidos los de bloques anidados como
    `generation`, que trae su propia procedencia."""
    found: list[tuple[str, str]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "git_sha" and isinstance(value, str):
                found.append((f"{trail}.{key}".lstrip("."), value))
            else:
                found += git_shas(value, f"{trail}.{key}")
    elif isinstance(node, list):
        for index, item in enumerate(node):
            found += git_shas(item, f"{trail}[{index}]")
    return found


def test_ningun_artefacto_publicado_viene_de_un_arbol_sucio():
    """Un `-dirty` dice que no se sabe que codigo produjo esos numeros.

    Paso: `ablation.json` se publico con `git_sha` sucio y sobrevivio a un
    commit y a un push. La procedencia lo decia y nadie la leyo, que es la razon
    de que esto sea un test y no un aviso.
    """
    offenders: list[str] = []
    for path in tracked_derived_artifacts():
        for field, sha in git_shas(json.loads(path.read_text(encoding="utf-8"))):
            if sha.endswith("-dirty"):
                offenders.append(f"{path.relative_to(ROOT)}: {field} = {sha}")
    assert not offenders
