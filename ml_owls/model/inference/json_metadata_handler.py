import json
from typing import Any

import numpy as np

from interfaces.model.metadata_handler import MetadataHandler


class JSONMetadataHandler(MetadataHandler):
    """JSON implementation of metadata handler."""

    class NumpyEncoder(json.JSONEncoder):
        """Custom JSON encoder for NumPy types."""

        def default(self, obj: Any) -> Any:
            """Override default method to handle NumPy types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    def save_metadata(self, metadata: dict[str, Any], path: str) -> None:
        """Save metadata to JSON file."""
        print(f"💾 Saving metadata: {path}")

        with open(path, "w") as f:
            json.dump(metadata, f, indent=2, cls=self.NumpyEncoder)

    def load_metadata(self, path: str) -> dict[str, Any]:
        """Load metadata from JSON file."""
        print(f"📂 Loading metadata: {path}")

        with open(path, "r") as f:
            result: dict[str, Any] = json.load(f)
            return result
