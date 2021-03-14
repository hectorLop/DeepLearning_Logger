import json
import numpy as np
from typing import Any

class ConfigsJSONEncoder(json.JSONEncoder):
    """
    JSON encoder to allow serialize numpy data types
    """
    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        
        return json.JSONEncoder.default(self, o)