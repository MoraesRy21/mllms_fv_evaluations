import os
from pathlib import Path
from typing import Union, Optional, Any

import yaml


def find_project_root(start: Union[str, Path] = None,
                      marker_names: Optional[list] = None) -> Path:
    """
    Recursively searches upwards from `start` until it finds a directory
    containing any file or folder listed in `marker_names`.

    Returns the path to the project root or raises a `FileNotFoundError` if not found.
    """
    if marker_names is None:
        marker_names = ["pyproject.toml", "setup.py", ".git", "README.md", "requirements.txt", "src"]

    if start is None:
        current = Path.cwd().resolve()
    else:
        current = Path(start).resolve()

    for parent in [current] + list(current.parents):
        for marker in marker_names:
            if (parent / marker).exists():
                return parent
    # fallback: se chegou na raiz do FS sem achar, usar current como root (mais seguro que raise)
    return current


def _get_value_by_dotted_key(d: dict, dotted_key: str, default: Any = None):
    if dotted_key is None:
        return default
    parts = dotted_key.split(".")
    cur = d
    try:
        for p in parts:
            cur = cur[p]
        return cur
    except Exception:
        return default


class Config:
    """
    Loader robusto de YAML configurado para procurar arquivos relativos ao project root.
    - Se `path` for None, tenta carregar "configs/config.yaml" no project root.
    - Se `path` for um nome simples (sem separadores), tenta procurar dentro de PROJECT_ROOT/configs/.
    - Se `path` for absoluto ou relativo contendo diretório, usa exatamente o caminho (após expanduser()).
    - fallback: configs/config.template.yaml ou arquivo template no mesmo dir do escolhido.
    """

    def __init__(self,
                 path: Union[str, Path, None] = None,
                 project_root: Union[str, Path, None] = None,
                 markers: Optional[list] = None):
        self.project_root = Path(find_project_root(project_root, marker_names=markers))
        self.config_dir = self.project_root / "configs"
        self._raw = {}
        self.source = None

        # Normaliza path argument
        self.path = self._resolve_path_arg(path)

        # Carrega o arquivo
        self._load()

    def _resolve_path_arg(self, path_arg: Union[str, Path, None]) -> Path:
        # default: configs/config.yaml
        if path_arg is None:
            candidate = self.config_dir / "config.yaml"
            return candidate

        p = Path(path_arg)
        # Expande ~ e variables
        p = Path(os.path.expanduser(str(p)))

        # Se o usuário passou só um nome (ex: "config.notebooks.yaml") ou apenas um filename
        if not p.is_absolute() and ("/" not in str(path_arg) and "\\" not in str(path_arg)):
            # procura primeiro em configs/, depois no project root, depois como está
            candidate = self.config_dir / path_arg
            if candidate.exists():
                return candidate
            candidate2 = self.project_root / path_arg
            if candidate2.exists():
                return candidate2
            # se não existe, retornamos candidate (configs/...) para que o loader faça fallback
            return candidate

        # Se path_arg contém diretório relativo (./something or ../something), resolvemos em relação ao cwd
        if not p.is_absolute():
            # normaliza para absoluto a partir do wd atual
            p = (Path.cwd() / p).resolve()

        return p

    def _load(self):
        # se existir o arquivo diretamente, carrega; caso contrário tenta fallback
        file_to_load = self.path
        template_candidates = [
            self.config_dir / "config.template.yaml",
            self.config_dir / "config.template.yml",
            self.project_root / "config.template.yaml",
            Path("config.template.yaml")
        ]

        if not file_to_load.exists():
            # tenta alguns fallbacks
            fallback = None
            for cand in template_candidates:
                if cand.exists():
                    fallback = cand
                    break
            if fallback is None:
                # Se arquivo não existir e não houver template, levantamos erro claro
                raise FileNotFoundError(
                    f"Config file not found at {file_to_load!s} and no template found in {self.config_dir} or project root."
                )
            file_to_load = fallback

        with open(file_to_load, "r", encoding="utf-8") as f:
            self._raw = yaml.safe_load(f) or {}
            self.source = file_to_load

    def get(self, key: str, default: Any = None) -> Any:
        """Suporta dotted keys: 'model.lr'"""
        return _get_value_by_dotted_key(self._raw, key, default)

    def __getitem__(self, key: str) -> Any:
        val = self.get(key)
        if val is None and key not in self._raw:
            # manter KeyError para compatibilidade com dict-like
            raise KeyError(key)
        return val

    def as_dict(self) -> dict:
        return dict(self._raw)

    def __repr__(self):
        return f"Config(source={self.source!s})"