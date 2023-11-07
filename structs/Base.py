from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional

@dataclass
class Base:
    def __post_init__(self) -> None:
        self.update()
    def update(self) -> None:
        pass

    def to_dict(self, filter_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        dc_dict = {f.name: getattr(self, f.name) for f in fields(self)}
        if filter_keys:
            [dc_dict.pop(k) for k in list(dc_dict.keys()) if k not in filter_keys]
        return dc_dict