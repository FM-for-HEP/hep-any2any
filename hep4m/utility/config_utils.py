"""Configuration utilities.

Environment-variable expansion in loaded YAML configs (used by
``hep4m.paths.safe_load_expanded``).
"""

import os


def expand_env_vars(obj):
    """Recursively expand ${VAR} and $VAR references in strings.

    Works on nested dicts, lists, and plain strings. Non-string leaves
    are returned unchanged. Unset variables are left as-is by os.path.expandvars.

    Args:
        obj: A string, list, dict, or other value.

    Returns:
        The input with all string values expanded.
    """
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, list):
        return [expand_env_vars(x) for x in obj]
    if isinstance(obj, dict):
        return {k: expand_env_vars(v) for k, v in obj.items()}
    return obj


