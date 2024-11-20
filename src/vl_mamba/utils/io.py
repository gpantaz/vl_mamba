import datetime
import json
from pathlib import Path
from typing import Any, Union


def json_serializer(object_to_serialize: Any) -> str:
    """Serialize an object."""
    if isinstance(object_to_serialize, datetime):  # type: ignore[arg-type]
        return str(object_to_serialize)

    raise TypeError(
        f"Object of type {object_to_serialize.__class__.__name__} is not JSON serializable"
    )


def read_json(file_path: Union[Path, str]) -> Any:
    """Read JSON file."""
    with open(file_path) as fp:
        return json.load(fp)


def write_json(file_path: Union[Path, str], data: Any) -> None:
    """Write JSON file."""
    with open(file_path, "w") as fp:
        json.dump(data, fp, indent=4)
