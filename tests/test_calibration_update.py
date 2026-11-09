"""Offline updater integration tests; no CUDA, network, or third-party runtime needed."""

import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _load(Path(__file__).resolve().parents[1] / "scripts/build_calibration_update.py", "calibration_update_builder")


@pytest.fixture
def update_fixture(tmp_path, monkeypatch):
    # Put the new file first so rollback exercises deletion and removal of new parents.
    source_names = ("newpkg/added.py", "slam.py", "poses/mini_ba.py")
    monkeypatch.setattr(builder, "SOURCE_FILES", source_names)
    monkeypatch.setattr(builder, "OPTIONAL_FILES", ())
    monkeypatch.setattr(builder, "REMOVE_PATHS", ())
    source = tmp_path / "source"
    checkout = tmp_path / "server checkout"
    for directory in (source, checkout):
        (directory / "poses").mkdir(parents=True)
    new_contents = {name: ("# new " + name + "\n").encode() for name in source_names}
    old_contents = {name: ("# old " + name + "\n").encode() for name in source_names[1:]}
    for name, content in new_contents.items():
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        (source / name).write_bytes(content)
    for name, content in old_contents.items():
        (checkout / name).write_bytes(content)
    script = builder.build_update(source, tmp_path / "apply_calibration_update.py")
    module = _load(script, "generated_calibration_update")
    return checkout, script, module, new_contents, old_contents


def test_update_old_checkout_and_record_originals(update_fixture):
    checkout, script, module, new, old = update_fixture
    result = subprocess.run([sys.executable, str(script), "--root", str(checkout)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "UPDATE OK" in result.stdout
    assert "Restart SLAM" in result.stdout
    assert "MiniBA after SHA256: " + hashlib.sha256(new["poses/mini_ba.py"]).hexdigest() in result.stdout
    for name, data in new.items():
        assert (checkout / name).read_bytes() == data
    backups = list((checkout / ".calibration-backups").iterdir())
    assert len(backups) == 1
    journal = json.loads((backups[0] / "backup_manifest.json").read_text())
    assert journal["status"] == "complete"
    assert journal["files"]["newpkg/added.py"]["existed"] is False
    for name, data in old.items():
        assert (backups[0] / name).read_bytes() == data
    assert module.apply_update(checkout, check_only=True) == 0
    assert module.apply_update(checkout) == 0
    assert list((checkout / ".calibration-backups").iterdir()) == backups


def test_check_only_and_required_root(update_fixture):
    checkout, script, module, _, old = update_fixture
    result = subprocess.run([sys.executable, str(script), "--root", str(checkout), "--check-only"], capture_output=True, text=True)
    assert result.returncode == 1
    assert "DIFF poses/mini_ba.py" in result.stdout
    for name, data in old.items():
        assert (checkout / name).read_bytes() == data
    assert not (checkout / "newpkg").exists()
    assert not (checkout / ".calibration-backups").exists()
    missing_root = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert missing_root.returncode == 2
    assert "--root" in missing_root.stderr


def test_archive_corruption_fails_before_mutation(update_fixture):
    checkout, _, module, _, old = update_fixture
    module.PAYLOAD_SHA256 = "0" * 64
    with pytest.raises(ValueError, match="archive SHA256"):
        module.apply_update(checkout)
    assert not (checkout / ".calibration-backups").exists()
    assert (checkout / "slam.py").read_bytes() == old["slam.py"]


@pytest.mark.parametrize("name", ("../escape.py", "/absolute.py", "C:/outside.py", "poses/../escape.py", "poses\\escape.py", "poses//escape.py"))
def test_payload_rejects_unsafe_paths(update_fixture, name):
    checkout, _, module, _, old = update_fixture
    content = b"must not be written"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, content)
    module.PAYLOAD_B64 = base64.b64encode(buffer.getvalue()).decode()
    module.PAYLOAD_SHA256 = hashlib.sha256(buffer.getvalue()).hexdigest()
    module.MANIFEST = {name: hashlib.sha256(content).hexdigest()}
    with pytest.raises(ValueError, match="Unsafe payload path"):
        module.apply_update(checkout)
    assert not (checkout / ".calibration-backups").exists()
    assert (checkout / "slam.py").read_bytes() == old["slam.py"]


def test_failure_rolls_back_existing_and_missing_files(update_fixture, monkeypatch):
    checkout, _, module, _, old = update_fixture
    original_write = module._atomic_write
    calls = 0

    def fail_second_write(target, data, mode=0o644):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected write failure")
        original_write(target, data, mode)

    monkeypatch.setattr(module, "_atomic_write", fail_second_write)
    with pytest.raises(RuntimeError, match="original files restored"):
        module.apply_update(checkout)
    for name, data in old.items():
        assert (checkout / name).read_bytes() == data
    assert not (checkout / "newpkg").exists()
    journals = list((checkout / ".calibration-backups").glob("*/backup_manifest.json"))
    assert len(journals) == 1
    assert json.loads(journals[0].read_text())["status"] == "rolled_back"


def test_post_write_verification_failure_rolls_back(update_fixture, monkeypatch):
    checkout, _, module, _, old = update_fixture
    original_write = module._atomic_write
    calls = 0

    def corrupt_first_write(target, data, mode=0o644):
        nonlocal calls
        calls += 1
        original_write(target, b"corrupted" if calls == 1 else data, mode)

    monkeypatch.setattr(module, "_atomic_write", corrupt_first_write)
    with pytest.raises(RuntimeError, match="Post-update SHA256 mismatch"):
        module.apply_update(checkout)
    for name, data in old.items():
        assert (checkout / name).read_bytes() == data
    assert not (checkout / "newpkg").exists()


def test_symlink_target_rejected(update_fixture, tmp_path):
    checkout, _, module, _, _ = update_fixture
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (checkout / "newpkg").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlinks unavailable on this system")
    with pytest.raises(ValueError, match="Refusing symlink"):
        module.apply_update(checkout)
    assert not (outside / "added.py").exists()
    assert not (checkout / ".calibration-backups").exists()


def test_builder_requires_all_declared_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "SOURCE_FILES", ("missing.py",))
    with pytest.raises(FileNotFoundError, match="Required update source is missing"):
        builder.build_update(tmp_path, tmp_path / "update.py")
    assert not (tmp_path / "update.py").exists()


@pytest.fixture
def cleanup_fixture(update_fixture, monkeypatch):
    checkout, script, _, new, old = update_fixture
    removed = ("legacy/obsolete.py", "legacy_models")
    (checkout / "legacy").mkdir()
    (checkout / removed[0]).write_bytes(b"original obsolete source\n")
    model = checkout / removed[1] / "nested" / "weights.bin"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"original model data\x00\xff")
    monkeypatch.setattr(builder, "REMOVE_PATHS", removed)
    builder.build_update(checkout.parent / "source", script)
    module = _load(script, "generated_source_cleanup_update")
    assert module.REMOVE_PATHS == removed
    return checkout, script, module, new, old


def test_removed_file_and_directory_move_to_backup_and_update_is_idempotent(cleanup_fixture, monkeypatch):
    checkout, _, module, new, old = cleanup_fixture
    copied = []
    original_copy = module.shutil.copy2

    def record_copy(source, destination):
        copied.append(Path(source))
        return original_copy(source, destination)

    monkeypatch.setattr(module.shutil, "copy2", record_copy)
    assert module.apply_update(checkout) == 0
    assert set(copied) == {checkout / name for name in old}
    assert all((checkout / name).read_bytes() == data for name, data in new.items())
    assert all(not (checkout / name).exists() for name in module.REMOVE_PATHS)
    backups = list((checkout / ".calibration-backups").iterdir())
    assert len(backups) == 1
    saved = backups[0] / "removed"
    assert (saved / "legacy/obsolete.py").read_bytes() == b"original obsolete source\n"
    assert (saved / "legacy_models/nested/weights.bin").read_bytes() == b"original model data\x00\xff"
    journal = json.loads((backups[0] / "backup_manifest.json").read_text())
    assert journal["status"] == "complete"
    assert journal["removed"]["legacy/obsolete.py"]["kind"] == "file"
    assert journal["removed"]["legacy_models"]["kind"] == "directory"
    assert module.apply_update(checkout, check_only=True) == 0
    assert module.apply_update(checkout) == 0
    assert list((checkout / ".calibration-backups").iterdir()) == backups


def test_check_only_reports_removals_when_replacement_sources_already_match(cleanup_fixture, capsys):
    checkout, _, module, new, _ = cleanup_fixture
    for name, data in new.items():
        target = checkout / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    assert module.apply_update(checkout, check_only=True) == 1
    output = capsys.readouterr().out
    assert "REMOVE legacy/obsolete.py" in output
    assert "REMOVE legacy_models" in output
    assert "DIFF " not in output
    assert not (checkout / ".calibration-backups").exists()
    assert (checkout / "legacy_models/nested/weights.bin").is_file()
    assert module.apply_update(checkout) == 0
    assert not (checkout / "legacy_models").exists()


@pytest.mark.parametrize("removed", (
    ("../outside",), ("legacy\\outside",), ("/absolute",), ("C:/outside",),
    ("slam.py",), ("poses",), ("slam.py/child",), ("legacy", "legacy/child"),
    ("legacy", "legacy"), (".calibration-backups",),
))
def test_unsafe_removal_plan_fails_before_any_checkout_writes(update_fixture, removed):
    checkout, _, module, _, old = update_fixture
    module.REMOVE_PATHS = removed
    with pytest.raises(ValueError):
        module.apply_update(checkout)
    assert not (checkout / ".calibration-backups").exists()
    assert not (checkout / "newpkg").exists()
    assert all((checkout / name).read_bytes() == data for name, data in old.items())


def test_builder_rejects_removal_conflicts_before_writing_output(update_fixture, monkeypatch, tmp_path):
    checkout, _, _, _, _ = update_fixture
    monkeypatch.setattr(builder, "REMOVE_PATHS", ("poses",))
    destination = tmp_path / "must-not-exist.py"
    with pytest.raises(ValueError, match="Conflicting"):
        builder.build_update(checkout.parent / "source", destination)
    assert not destination.exists()


@pytest.mark.parametrize("source_names, removed", (
    (("removed/legacy.py",), ("legacy.py",)),
    (("removed",), ("legacy.py",)),
    (("backup_manifest.json",), ()),
))
def test_plan_rejects_collisions_with_backup_metadata(source_names, removed):
    with pytest.raises(ValueError, match="Conflicting backup paths"):
        builder._validate_plan(source_names, removed)


def test_write_failure_restores_removed_file_and_directory(cleanup_fixture, monkeypatch):
    checkout, _, module, _, old = cleanup_fixture
    original_write = module._atomic_write
    calls = 0

    def fail_second_write(target, data, mode=0o644):
        nonlocal calls
        calls += 1
        assert calls > 2 or not (checkout / "legacy_models").exists()
        if calls == 2:
            raise OSError("injected cleanup write failure")
        original_write(target, data, mode)

    monkeypatch.setattr(module, "_atomic_write", fail_second_write)
    with pytest.raises(RuntimeError, match="original files restored"):
        module.apply_update(checkout)
    assert all((checkout / name).read_bytes() == data for name, data in old.items())
    assert not (checkout / "newpkg").exists()
    assert (checkout / "legacy/obsolete.py").read_bytes() == b"original obsolete source\n"
    assert (checkout / "legacy_models/nested/weights.bin").read_bytes() == b"original model data\x00\xff"
    journals = list((checkout / ".calibration-backups").glob("*/backup_manifest.json"))
    assert json.loads(journals[0].read_text())["status"] == "rolled_back"


def test_failed_directory_rename_restores_previously_moved_file(cleanup_fixture, monkeypatch):
    checkout, _, module, _, old = cleanup_fixture
    original_rename = module.os.rename

    def fail_model_move(source, destination):
        if Path(source) == checkout / "legacy_models":
            raise OSError("injected rename failure")
        original_rename(source, destination)

    monkeypatch.setattr(module.os, "rename", fail_model_move)
    with pytest.raises(RuntimeError, match="original files restored"):
        module.apply_update(checkout)
    assert all((checkout / name).read_bytes() == data for name, data in old.items())
    assert (checkout / "legacy/obsolete.py").is_file()
    assert (checkout / "legacy_models/nested/weights.bin").is_file()
    assert not (checkout / "newpkg").exists()


def test_removal_symlink_rejected_before_mutation(update_fixture, tmp_path):
    checkout, _, module, _, _ = update_fixture
    outside = tmp_path / "outside_models"
    outside.mkdir()
    try:
        (checkout / "legacy_models").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlinks unavailable on this system")
    module.REMOVE_PATHS = ("legacy_models",)
    with pytest.raises(ValueError, match="Refusing symlink"):
        module.apply_update(checkout)
    assert not (checkout / ".calibration-backups").exists()
    assert outside.is_dir()


def test_repository_root_symlink_rejected_before_mutation(update_fixture, tmp_path):
    checkout, _, module, _, _ = update_fixture
    linked_root = tmp_path / "linked_checkout"
    try:
        linked_root.symlink_to(checkout, target_is_directory=True)
    except OSError:
        pytest.skip("Symlinks unavailable on this system")
    with pytest.raises(ValueError, match="Refusing symlink repository root"):
        module.apply_update(linked_root)
    assert not (checkout / ".calibration-backups").exists()


def test_selected_source_update_preserves_unrelated_files(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = builder.build_update(
        root, tmp_path / "apply_clidd_import_fix.py",
        source_files=("poses/clidd_adapter.py",), optional_files=(), remove_paths=(),
    )
    module = _load(script, "generated_clidd_import_fix")
    assert set(module.MANIFEST) == {"poses/clidd_adapter.py"}
    assert module.REMOVE_PATHS == ()
    checkout = tmp_path / "server"
    checkout.mkdir()
    originals = {
        "slam.py": b"# existing server entry\n",
        "local_settings.py": b"# existing local settings\n",
        "notes/experiment.txt": b"existing experiment notes\n",
    }
    for name, data in originals.items():
        target = checkout / name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(data)
    result = subprocess.run(
        [sys.executable, "-S", str(script), "--root", str(checkout)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "pip install" not in result.stdout
    assert (checkout / "poses/clidd_adapter.py").read_bytes() == (root / "poses/clidd_adapter.py").read_bytes()
    assert all((checkout / name).read_bytes() == data for name, data in originals.items())
    # A real package import in a fresh process, without site-packages or a CLIDD
    # checkout: importing the dimension helper must not load models/weights.
    probe = subprocess.run(
        [sys.executable, "-S", "-c",
         "import sys, types; sys.modules['torch'] = types.ModuleType('torch'); "
         "from poses.clidd_adapter import clidd_descriptor_dim; "
         "assert clidd_descriptor_dim('S64') == 64; "
         "assert clidd_descriptor_dim('G128') == 128"],
        cwd=checkout, capture_output=True, text=True,
    )
    assert probe.returncode == 0, probe.stderr
    assert module.apply_update(checkout, check_only=True) == 0
    assert module.apply_update(checkout) == 0


def test_full_source_update_replaces_descriptor_pipeline_and_backs_up_old_assets(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = builder.build_update(root, tmp_path / "apply_clidd_source_update.py")
    module = _load(script, "generated_complete_clidd_update")
    core_modules = (
        "poses/feature_detector.py", "poses/matcher.py", "scene/dense_extractor.py",
        "poses/clidd_adapter.py",
    )
    assert set(core_modules) <= set(module.MANIFEST)
    checkout = tmp_path / "server"
    checkout.mkdir()
    originals = {
        "slam.py": b"# original server entry\n",
        "CLIDD-main/weights/S64.pth": b"obsolete default descriptor weights",
        **{name: ("# obsolete implementation: " + name + "\n").encode()
           for name in core_modules[:-1]},
    }
    for name, data in originals.items():
        target = checkout / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    other_weights = checkout / "CLIDD-main/weights/N64.pth"
    other_weights.write_bytes(b"existing other descriptor weights")
    removed_files = {}
    for name in module.REMOVE_PATHS:
        path = checkout / name
        target = path if path.suffix else path / "nested/old_asset.bin"
        target.parent.mkdir(parents=True, exist_ok=True)
        data = ("obsolete asset: " + name).encode()
        target.write_bytes(data)
        removed_files[target.relative_to(checkout).as_posix()] = data
    before = subprocess.run(
        [sys.executable, "-S", str(script), "--root", str(checkout), "--check-only"],
        capture_output=True, text=True,
    )
    assert before.returncode == 1, before.stderr
    assert not (checkout / ".calibration-backups").exists()
    applied = subprocess.run(
        [sys.executable, "-S", str(script), "--root", str(checkout)],
        capture_output=True, text=True,
    )
    assert applied.returncode == 0, applied.stderr
    for name, expected_digest in module.MANIFEST.items():
        data = (checkout / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected_digest
        assert data == (root / name).read_bytes()
    assert other_weights.read_bytes() == b"existing other descriptor weights"
    backups = list((checkout / ".calibration-backups").iterdir())
    assert len(backups) == 1
    for name, data in originals.items():
        assert (backups[0] / name).read_bytes() == data
    for name, data in removed_files.items():
        assert (backups[0] / "removed" / name).read_bytes() == data
    assert all(not (checkout / name).exists() for name in module.REMOVE_PATHS)
    checked = subprocess.run(
        [sys.executable, "-S", str(script), "--root", str(checkout), "--check-only"],
        capture_output=True, text=True,
    )
    assert checked.returncode == 0, checked.stderr
    assert "CHECK OK" in checked.stdout
    assert module.apply_update(checkout) == 0
    assert list((checkout / ".calibration-backups").iterdir()) == backups
