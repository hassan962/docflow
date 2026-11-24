import pytest

from docflow_core.config import DEFAULT_REVIEW_THRESHOLD, review_threshold
from docflow_core.security import generate_api_key, hash_api_key
from docflow_core.taxid import gstin_check_char, is_valid_gstin


@pytest.mark.parametrize("gstin", ["27AAPFU0939F1ZV", "29AAGCB7383J1Z4"])
def test_published_gstins_pass_the_checksum(gstin: str) -> None:
    assert gstin_check_char(gstin[:14]) == gstin[14]
    assert is_valid_gstin(gstin.lower())


@pytest.mark.parametrize("gstin", ["27AAPFU0939F1ZX", "27AAPFU0939F1Z", "27-APFU0939F1ZV", ""])
def test_bad_gstins_fail(gstin: str) -> None:
    assert not is_valid_gstin(gstin)


def test_review_threshold_defaults_and_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REVIEW_THRESHOLD_INVOICE", raising=False)
    assert review_threshold("invoice") == DEFAULT_REVIEW_THRESHOLD
    monkeypatch.setenv("REVIEW_THRESHOLD_INVOICE", "0.97")
    assert review_threshold("invoice") == 0.97  # read live, no restart needed
    monkeypatch.setenv("REVIEW_THRESHOLD_INVOICE", "1.5")
    with pytest.raises(ValueError):
        review_threshold("invoice")


def test_api_keys_store_only_a_hash() -> None:
    key = generate_api_key()
    assert key.secret.startswith(key.prefix + "_")
    assert key.key_hash == hash_api_key(key.secret)
    assert key.secret not in key.key_hash
    assert generate_api_key().secret != key.secret
