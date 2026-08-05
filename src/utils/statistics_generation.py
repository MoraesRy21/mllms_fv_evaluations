import json
from pathlib import Path
from typing import Any, Dict, Union

class StatisticsManager:
    """
    Centralized statistics manager for collecting, updating, and persisting dataset statistics in the JSON format.
    """

    def __init__(self, output_path: Union[str, Path], filename: str):
        """
        :param output_path (Path/str): Directory to save the file.
        :param filename (str): The name of the output JSON file.
        """
        self.output_path = Path(output_path)
        self.file_path = self.output_path / filename
        self.data: Dict[str, Any] = {}
        self.output_path.mkdir(parents=True, exist_ok=True)
        self._load_existing()

    def _load_existing(self):
        """Load the existent data in the disc if the file already exists."""

        if self.file_path.exists():
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    print("File already exists, loading existing data...")
                    self.data = json.load(f)
            except Exception as e:
                print(f"⚠️ Warning: Fail in load {self.file_path}: {e}")
        else:
            print("File does not exist yet! Use 'load_from_dict' method for initializing.")

    def load_from_dict(self, data_dict: Dict[str, Any], overwrite: bool = False) -> 'StatisticsManager':
        """
        Initialize or merge the internal state with a dictionary.
        Useful for loading initial metadata.
        """

        if overwrite:
            self.data = data_dict
        else:
            self.data.update(data_dict)
        return self

    def update(self, key: str, value: Any, append: bool = False) -> 'StatisticsManager':
        """
        Add or modify a UM metric.

        Parameters:
            key (str): The identifier or category for data insertion.
            value: The statistical information to be included.
            append (bool): Determines whether to add data to an existing list (True) or overwrite the key's value (False).

        Returns:
            The current instance for method chaining (fluent design).
        """
        if append:
            if key not in self.data or not isinstance(self.data[key], list):
                self.data[key] = []
            self.data[key].append(value)
        else:
            self.data[key] = value
        return self

    def save(self):
        """Persist the current state in the disc with type complex transformation."""
        self.output_path.mkdir(parents=True, exist_ok=True)

        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=4, ensure_ascii=False, default=self._json_serializer)
            print(f"✅ Estatísticas salvas com sucesso em: {self.file_path}")
        except Exception as e:
            print(f"❌ Erro ao salvar estatísticas: {e}")

    @staticmethod
    def _json_serializer(obj: Any) -> Any:
        """Serialize types NumPy, Pandas e Pathlib that JSON does not support."""
        if hasattr(obj, "item"): # NumPy scalars (int64, float64)
            return obj.item()
        if hasattr(obj, "tolist"): # NumPy arrays
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)