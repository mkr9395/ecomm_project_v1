from pathlib import Path
import os
import yaml

def _project_root() -> Path:
    """
    This function always returns the project root directory, regardless of where Python is executed from.
    Leading underscore _project_root
    - Means internal/private helper function
    - Not intended to be imported elsewhere
    """
    # __file__ -> Path object pointing to config_loader.py
    # .resolve() → Converts to an absolute path (resolves symlinks)
    # .parents[1] -> parents[0] → utils/ -> parents[1] → project root
    return Path(__file__).resolve().parents[1] 

def load_config(config_path : str | None = None) -> dict: # str → user provides a path , None → use default resolution logic
    """
    Resolve config path reliably irrespective of CWD.
    Priority: explicit arg > CONFIG_PATH env > <project_root>/config/config.yaml
    """
    env_path = os.getenv("CONFIG_PATH") # Reads environment variable CONFIG_PATH
    if config_path is None:
        config_path = env_path or str(_project_root()/"config"/"config.yaml" ) 
        # If user did not pass config_path: Use CONFIG_PATH env var if it exists, Otherwise fallback to: <project_root>/config/config.yaml
        
    path = Path(config_path)
    if not path.is_absolute():
        path = _project_root() / path
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, "r", encoding='utf-8') as f:
        return yaml.safe_load(f) or {}