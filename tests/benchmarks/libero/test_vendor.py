from vlalagbench.benchmarks.libero import _vendor


def test_libero_root_can_reference_a_read_only_checkout(tmp_path, monkeypatch):
    checkout = tmp_path / "libero"
    monkeypatch.setenv("VLALAGBENCH_LIBERO_ROOT", str(checkout))

    assert _vendor._configured_libero_root() == checkout.resolve()


def test_existing_libero_config_path_is_never_modified(tmp_path, monkeypatch):
    custom = tmp_path / "custom-libero"
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(custom))

    _vendor._ensure_libero_config()

    assert not custom.exists()
