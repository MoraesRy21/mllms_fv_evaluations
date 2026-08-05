import datetime
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import pandas as pd


class DatasetFiltration:
    """
    Class for loading and filtering CSV datasets based on configurations.

    Allows extremely parameterizable filtering of CSV data using multiple criteria.
    """

    def __init__(self,
                 dataframe: Optional[pd.DataFrame] = None,
                 csv_path: Optional[Union[str, Path]] = None):
        """
        Initialize the dataset loader.

        Args:
            dataframe: DataFrame to use directly. If None, loads from csv_path.
            csv_path: Path to CSV file. If None, tries to load from config.
            config_path: Path to YAML configuration file.
        """
        self.csv_path: Optional[Path] = None

        # If dataframe is provided, use it directly
        if dataframe is not None:
            self._df = dataframe.copy()
        else:
            # Otherwise, load from CSV
            if csv_path is None:
                raise ValueError("CSV path not provided and not found in config")

            self.csv_path = Path(csv_path)
            if not self.csv_path.exists():
                raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

            self._df: Optional[pd.DataFrame] = None

    @property
    def load(self) -> pd.DataFrame:
        """Load DataFrame on demand (lazy loading)."""
        if self._df is None:
            if self.csv_path is None:
                raise ValueError("No DataFrame loaded and no CSV path available")
            self._df = pd.read_csv(self.csv_path, sep=';')
        return self._df

    def reload(self) -> 'DatasetFiltration':
        """Reload CSV from disk."""
        if self.csv_path is None:
            raise ValueError("Cannot reload: no CSV path available")
        self._df = None
        return self

    def filter(self,
               column_filters: Optional[Dict[str, Any]] = None,
               range_filters: Optional[Dict[str, tuple]] = None,
               value_filters: Optional[Dict[str, List[Any]]] = None,
               boolean_filters: Optional[Dict[str, bool]] = None,
               custom_filter: Optional[callable] = None,
               drop_na_columns: Optional[List[str]] = None,
               keep_columns: Optional[List[str]] = None,
               inplace: bool = False) -> pd.DataFrame:
        """
        Universal filtering method with extreme parameterization.

        Args:
            column_filters: Exact equality filters {column: value}
                Example: {'op': 'OP', 'day': '2025-10-08'}

            range_filters: Range filters {column: (min, max)}
                Use None for open bounds.
                Example: {'brightness': (50, 150), 'confidence': (0.5, None)}

            value_filters: Multiple value filters {column: [val1, val2, ...]}
                Example: {'buss_line': [612, 5116, 6240]}

            boolean_filters: Boolean filters {column: True/False}
                Example: {'is_blurry': False, 'face_detected': True}

            custom_filter: Custom function that receives DataFrame and returns boolean mask
                Example: lambda df: (df['w_bbox'] * df['h_bbox']) > 50000

            drop_na_columns: List of columns to remove rows with NA
                Example: ['x_bbox', 'y_bbox', 'confidence']

            keep_columns: List of columns to keep in result (None = all)
                Example: ['filename', 'confidence', 'brightness', 'is_blurry']

            inplace: If True, modifies self._df. If False, returns new DataFrame.

        Returns:
            Filtered DataFrame

        Examples:
            >>> # Filter only detected faces with good quality
            >>> df = filtration.filter(
            ...     boolean_filters={'face_detected': True, 'is_blurry': False},
            ...     range_filters={'confidence': (0.7, None), 'brightness': (50, 200)},
            ...     drop_na_columns=['x_bbox', 'y_bbox']
            ... )

            >>> # Filter by specific bus lines on a day
            >>> df = filtration.filter(
            ...     column_filters={'day': '2025-10-08'},
            ...     value_filters={'buss_line': [612, 5116, 6240]},
            ...     keep_columns=['filename', 'buss_line', 'confidence']
            ... )

            >>> # Complex custom filter
            >>> df = filtration.filter(
            ...     custom_filter=lambda df: (df['main_bbox_area'] > 30000) &
            ...                              (df['face_ratio'] > 0.15),
            ...     boolean_filters={'is_outlier': False}
            ... )
        """
        df = self.load.copy()

        # Apply equality filters
        if column_filters:
            for col, value in column_filters.items():
                if col in df.columns:
                    df = df[df[col] == value]

        # Apply range filters
        if range_filters:
            for col, (min_val, max_val) in range_filters.items():
                if col in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        if isinstance(min_val, (datetime.date, datetime.datetime)) and not isinstance(min_val, pd.Timestamp):
                            min_val = pd.to_datetime(min_val)
                    if isinstance(max_val, (datetime.date, datetime.datetime)) and not isinstance(max_val, pd.Timestamp):
                        max_val = pd.to_datetime(max_val)
                    if min_val is not None:
                        df = df[df[col] >= min_val]
                    if max_val is not None:
                        df = df[df[col] <= max_val]

        # Apply multiple value filters (IN)
        if value_filters:
            for col, values in value_filters.items():
                if col in df.columns:
                    df = df[df[col].isin(values)]

        # Apply boolean filters
        if boolean_filters:
            for col, bool_value in boolean_filters.items():
                if col in df.columns:
                    df = df[df[col] == bool_value]

        # Remove rows with NA in specific columns
        if drop_na_columns:
            valid_cols = [c for c in drop_na_columns if c in df.columns]
            if valid_cols:
                df = df.dropna(subset=valid_cols)

        # Apply custom filter
        if custom_filter:
            mask = custom_filter(df)
            df = df[mask]

        # Select only desired columns
        if keep_columns:
            valid_cols = [c for c in keep_columns if c in df.columns]
            df = df[valid_cols]

        # Update internal DataFrame if inplace=True
        if inplace:
            self._df = df

        return df

    def get_column_info(self) -> Dict[str, Any]:
        """
        Returns information about dataset columns.

        Returns:
            Dictionary with information for each column (type, unique values, etc.)
        """
        df = self.load()
        info = {}

        for col in df.columns:
            info[col] = {
                'dtype': str(df[col].dtype),
                'non_null_count': int(df[col].count()),
                'null_count': int(df[col].isna().sum()),
                'unique_count': int(df[col].nunique())
            }

            # Add statistics for numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                info[col].update({
                    'min': float(df[col].min()) if not df[col].isna().all() else None,
                    'max': float(df[col].max()) if not df[col].isna().all() else None,
                    'mean': float(df[col].mean()) if not df[col].isna().all() else None,
                    'median': float(df[col].median()) if not df[col].isna().all() else None
                })

            # Add unique values for categorical columns (if few values)
            if df[col].nunique() <= 20:
                info[col]['unique_values'] = df[col].unique().tolist()

        return info

    def get_summary_stats(self) -> pd.DataFrame:
        """Returns descriptive statistics of the dataset."""
        return self.load.describe(include='all')

    def __repr__(self):
        rows = len(self._df) if self._df is not None else "not loaded"
        csv_name = self.csv_path.name if self.csv_path else "DataFrame"
        return f"DatasetFiltration(csv={csv_name}, rows={rows})"
