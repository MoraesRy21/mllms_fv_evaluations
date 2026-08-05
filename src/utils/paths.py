from pathlib import Path

from utils.config_loader import Config


class PathBuilder:
    """
    Bulder fluent for constructing paths.
    Allow cascade parts and combine with configuration values.
    """

    def __init__(self, resolver: 'ProjectPathResolver', base_path: Path):
        self._resolver = resolver
        self._path = base_path

    def join(self, *parts: str) -> 'PathBuilder':
        """Add parts to the path."""
        return PathBuilder(self._resolver, self._path.joinpath(*parts))

    def config(self, key: str) -> 'PathBuilder':
        """Add a value from the configuration to the current path."""
        config_value = self._resolver._config[key]
        return PathBuilder(self._resolver, self._path / config_value)

    def with_name(self, name: str) -> 'PathBuilder':
        """Replace the name of the file while keeping the directory."""
        return PathBuilder(self._resolver, self._path.with_name(name))

    def with_suffix(self, suffix: str) -> 'PathBuilder':
        """Modify the file extension."""
        return PathBuilder(self._resolver, self._path.with_suffix(suffix))

    @property
    def path(self) -> Path:
        """Retorna o Path final construído. Return the final Path built"""
        return self._path

    def __truediv__(self, other: str) -> 'PathBuilder':
        """Allows uses the operator / to add parts."""
        return self.join(other)

    def __str__(self) -> str:
        return str(self._path)

    def __fspath__(self) -> str:
        """Allows to use directly with open(), pd.read_csv(), etc."""
        return str(self._path)



class ProjectPathResolver:
    """
    Resolves paths relative to the project root directory.

    This class is used to manage and construct file paths that are based on
    the root directory of a project. It ensures all paths are resolved and
    computed relative to the root directory, providing a centralized way
    to handle project file structure.

    :ivar project_root: The resolved absolute path to the project root directory.
    :type project_root: Path
    """

    def __init__(self, config: Config):
        self._config = config
        self.project_root = Path(config.project_root).resolve()

    def path(self, *relative_parts) -> PathBuilder:
        """
        Build a path relative to the project root directory.

        Uses:
            resolver.path("outputs", "plots", "file.png")
            resolver.path("notebooks", "analysis") / "insights.md"
        """
        base = self.project_root.joinpath(*relative_parts)
        return PathBuilder(self, base)

    def from_config(self, key: str) -> PathBuilder:
        """
        Initialize a path from a configuration key.

        Uso:
            resolver.from_config("dataset.analyses.path") / "subdir" / "file.csv"
            resolver.from_config("dataset.analyses.plots_path").join("experiment_01", "plot.png")
        """
        config_value = self._config[key]
        return PathBuilder(self, self.project_root / config_value)

    def __getitem__(self, key: str) -> PathBuilder:
        """
        Shortcut for from_config().

        Uso:
            resolver["dataset.analyses.plots_path"] / "my_plot.png"
        """
        return self.from_config(key)

    # --- Atalhos para diretórios comuns (opcional) ---
    @property
    def results(self) -> PathBuilder:
        """Atalho para o diretório results/"""
        return self.path("results")

    @property
    def notebooks(self) -> PathBuilder:
        """Atalho para o diretório notebooks/"""
        return self.path("notebooks")

    @property
    def configs(self) -> PathBuilder:
        """Atalho para o diretório configs/"""
        return self.path("configs")
