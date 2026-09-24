"""Tests for hep4m.utility.config_utils."""

import os


def test_expand_env_vars_simple(monkeypatch):
    """${FOO} is replaced in string values."""
    from hep4m.utility.config_utils import expand_env_vars

    monkeypatch.setenv("FOO", "/some/path")
    assert expand_env_vars("${FOO}/data") == "/some/path/data"
    assert expand_env_vars("$FOO/data") == "/some/path/data"


def test_expand_env_vars_nested(monkeypatch):
    """Recursion into nested dicts and lists."""
    from hep4m.utility.config_utils import expand_env_vars

    monkeypatch.setenv("BASE", "/base")
    obj = {
        "path": "${BASE}/train",
        "items": ["${BASE}/a", "${BASE}/b"],
        "nested": {"deep": "${BASE}/deep"},
        "number": 42,
        "flag": True,
    }
    result = expand_env_vars(obj)
    assert result["path"] == "/base/train"
    assert result["items"] == ["/base/a", "/base/b"]
    assert result["nested"]["deep"] == "/base/deep"
    assert result["number"] == 42
    assert result["flag"] is True


def test_expand_env_vars_missing_var():
    """Unset variables are left as-is by os.path.expandvars."""
    from hep4m.utility.config_utils import expand_env_vars

    # os.path.expandvars leaves unknown ${VARS} unchanged
    key = "VERY_UNLIKELY_ENV_VAR_12345"
    if key in os.environ:
        del os.environ[key]
    result = expand_env_vars(f"${{{key}}}/path")
    assert key in result  # left unexpanded


