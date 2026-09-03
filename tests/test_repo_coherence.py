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


# ------------------------------------------------------- CI frente a local

WORKFLOW = ".github/workflows/ci.yml"

# `uv run` y `uv sync` resuelven el proyecto y ESCRIBEN uv.lock en la raiz. Ese
# fichero no esta versionado, asi que ensucia el arbol y los generadores de
# artefactos se niegan a producir numeros desde ahi. `uv venv` y `uv pip install`
# no tienen ese efecto y siguen permitidos.
FORBIDDEN_RUNNERS = ("uv run", "uv sync")
INTERPRETER = ".venv/bin/python"


def workflow_steps() -> list[tuple[str, str]]:
    """(nombre del paso, comando) de cada `run:` del workflow."""
    path = ROOT / WORKFLOW
    if not path.exists():
        pytest.skip("no hay workflow de CI")
    data = yaml.safe_load(path.read_text())
    return [
        (f"{job}: {step.get('name', '?')}", step["run"])
        for job, spec in (data.get("jobs") or {}).items()
        for step in spec.get("steps", [])
        if step.get("run")
    ]


def test_ci_no_usa_lanzadores_que_ensucian_el_arbol():
    """Rompio una corrida entera, y el modo de fallo no era evidente.

    CI invocaba con `uv run`, que escribe uv.lock, y el paso de ablacion aborto
    con "el arbol tiene cambios sin commitear: uv.lock". El guardia acerto; lo
    que fallaba era que CI y local no ejecutaban por el mismo camino.
    """
    offenders = [
        f"{name}: usa `{runner}`  ({command.strip().splitlines()[0][:60]})"
        for name, command in workflow_steps()
        for runner in FORBIDDEN_RUNNERS
        if runner in command
    ]
    assert not offenders


def test_ci_invoca_el_mismo_interprete_que_el_makefile():
    """Un solo modo de ejecutar el proyecto.

    El comentario del workflow afirmaba que CI y local instalaban igual, y era
    cierto; lo que no decia es que ejecutaban distinto, y ahi vivia el fallo.
    Esto comprueba la divergencia, no su sintoma.

    Limite conocido: cubre esta familia de lanzadores, no cualquier efecto
    colateral imaginable. Un `poetry install` nuevo no lo veria.
    """
    offenders = [
        f"{name}: {line.strip()[:70]}"
        for name, command in workflow_steps()
        for line in command.splitlines()
        if "python -m " in line and INTERPRETER not in line
    ]
    assert not offenders


def test_la_clave_de_cache_nombra_el_modelo_que_declara_la_config(config):
    """La clave de cache del modelo tiene que seguir a `retrieval.dense.model`.

    Antes hasheaba config.yaml entero, asi que un comentario invalidaba la cache
    y costaba volver a bajar 419 MB. Nombrar el modelo lo arregla, pero crea una
    invariante que solo vivia en un comentario: si el modelo cambia y la clave
    no, CI serviria pesos cacheados de otro modelo. Aqui se hace cumplir.
    """
    path = ROOT / WORKFLOW
    if not path.exists():
        pytest.skip("no hay workflow de CI")
    keys = re.findall(r"^\s*key:\s*(hf-\S+)\s*$", path.read_text(), re.MULTILINE)
    if not keys:
        pytest.skip("el workflow no cachea el modelo")
    expected = f"hf-{config.get('retrieval.dense.model').split('/')[-1]}"
    assert keys == [expected] * len(keys)


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


def dirt_markers(node, trail="") -> list[tuple[str, str]]:
    """Marcas de suciedad en cualquier profundidad del documento.

    Son dos y hay que mirar las dos: el campo booleano `dirty`, que estampan los
    generadores desde que existe `require_clean_tree`, y el sufijo `-dirty` del
    `git_sha`, que es como se marcaba antes. Se recorre en profundidad porque el
    bloque `generation` que inyecta la evaluacion trae su propia procedencia.
    """
    found: list[tuple[str, str]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            where = f"{trail}.{key}".lstrip(".")
            if key == "git_sha" and isinstance(value, str) and value.endswith("-dirty"):
                found.append((where, value))
            elif key == "dirty" and value is True:
                found.append((where, "true"))
            else:
                found += dirt_markers(value, f"{trail}.{key}")
    elif isinstance(node, list):
        for index, item in enumerate(node):
            found += dirt_markers(item, f"{trail}[{index}]")
    return found


def test_ningun_artefacto_publicado_viene_de_un_arbol_sucio():
    """Un `-dirty` dice que no se sabe que codigo produjo esos numeros.

    Paso: `ablation.json` se publico con `git_sha` sucio y sobrevivio a un
    commit y a un push. La procedencia lo decia y nadie la leyo, que es la razon
    de que esto sea un test y no un aviso.
    """
    offenders: list[str] = []
    for path in tracked_derived_artifacts():
        offenders += [
            f"{path.relative_to(ROOT)}: {field} = {value}"
            for field, value in dirt_markers(json.loads(path.read_text(encoding="utf-8")))
        ]
    assert not offenders
