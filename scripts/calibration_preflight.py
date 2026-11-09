"""Check deployed calibration, local-BA and trajectory interfaces without runtime imports."""

import argparse
import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_FILES = (
    "poses/mini_ba.py",
    "poses/pose_initializer.py",
    "utils.py",
    "dataloaders/waymo_dataset.py",
    "args.py",
)
LOCAL_BA_FILES = (
    "poses/local_ba.py",
    "poses/local_ba_solver.py",
    "scene/scene_model.py",
    "slam_backend.py",
    "slam.py",
)
TRAJECTORY_FILES = ("slam_outputs.py", "slam.py")
EXPECTED_TRAJECTORY_STAGES = ("frontend_miniba", "online_mapping")


def _method(tree, class_name, method_name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for method in node.body:
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) and method.name == method_name:
                    return method
    return None


def _camera_pair_sample(method, variable):
    if method is None:
        return False
    for node in ast.walk(method):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == variable for target in node.targets):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        shape = node.value.args
        if shape and isinstance(shape[0], (ast.Tuple, ast.List)):
            shape = shape[0].elts
        if len(shape) >= 2:
            if isinstance(shape[0], ast.Name) and shape[0].id == "n_cams":
                if isinstance(shape[1], ast.Constant) and shape[1].value == 2:
                    return True
    return False


def _normalizes_before_replay(method):
    if method is None:
        return False
    normalizers = []
    replays = []
    for node in ast.walk(method):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "_camera_intrinsics":
            normalizers.append(node.lineno)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "optimizer":
            replays.append(node.lineno)
    return bool(normalizers and replays and min(normalizers) < min(replays))


def _calls(node, name):
    if node is None:
        return []
    return [
        call for call in ast.walk(node) if isinstance(call, ast.Call)
        and ((isinstance(call.func, ast.Name) and call.func.id == name)
             or (isinstance(call.func, ast.Attribute) and call.func.attr == name))
    ]


def _assigns_self_attribute(node, name):
    return node is not None and any(
        isinstance(child, ast.Attribute) and isinstance(child.ctx, ast.Store)
        and isinstance(child.value, ast.Name) and child.value.id == "self"
        and child.attr == name for child in ast.walk(node)
    )


def _has_parameters(method, names):
    if method is None:
        return False
    parameters = {arg.arg for arg in (
        method.args.posonlyargs + method.args.args + method.args.kwonlyargs
    )}
    return set(names).issubset(parameters)


def _uses_names(node, names):
    if node is None:
        return False
    used = {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}
    return set(names).issubset(used)


def _preserves_gaussian_owners(method):
    if method is None or not _has_parameters(method, ("extension_owner_ids",)):
        return False
    old_owners = set()
    preserves = []
    derives_creation_owner = False
    for node in ast.walk(method):
        if not isinstance(node, ast.Assign):
            continue
        if _calls(node.value, "_fit_gaussian_owner_keyframe_ids"):
            old_owners.update(target.id for target in node.targets if isinstance(target, ast.Name))
        if any(isinstance(target, ast.Name) and target.id == "extension_owner_ids" for target in node.targets):
            for child in ast.walk(node.value):
                if (isinstance(child, ast.Subscript) and isinstance(child.value, ast.Name)
                        and child.value.id == "extension_observer_ids"
                        and isinstance(child.slice, ast.Tuple) and len(child.slice.elts) == 2
                        and isinstance(child.slice.elts[1], ast.Constant)
                        and child.slice.elts[1].value == 0):
                    derives_creation_owner = True
        if any(isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
               and target.value.id == "self" and target.attr == "gaussian_owner_keyframe_ids"
               for target in node.targets):
            preserves.append(node.value)
    return derives_creation_owner and any(
        _uses_names(value, (old_name, "valid_mask", "extension_owner_ids"))
        for value in preserves for old_name in old_owners
    )


def _has_creation_owner_path(method):
    """Recognize owners passed directly or derived from creation observer slot 0."""
    if method is None:
        return False
    creation_observers = set()
    creation_owners = set()
    for node in ast.walk(method):
        if not isinstance(node, ast.Assign):
            continue
        # The initial observer must be the creation keyframe, not a later viewer.
        if not any(isinstance(child, ast.Attribute) and child.attr == "index"
                   and isinstance(child.value, ast.Name) and child.value.id == "keyframe"
                   for child in ast.walk(node.value)):
            continue
        targets = {target.id for target in node.targets if isinstance(target, ast.Name)}
        creation_owners.update(targets)
        if _calls(node.value, "_observer_ids_for_keyframe"):
            creation_observers.update(targets)
    for call in _calls(method, "_apply_gaussian_add_and_prune"):
        for keyword in call.keywords:
            if keyword.arg == "extension_owner_ids":
                if isinstance(keyword.value, ast.Name) and keyword.value.id in creation_owners:
                    return True
            if keyword.arg == "extension_observer_ids" and isinstance(keyword.value, ast.Name):
                if keyword.value.id in creation_observers:
                    return True
    return False


def _check_local_ba_sources(sources, errors):
    trees = {}
    for relative in LOCAL_BA_FILES:
        source = sources.get(relative)
        if source is None:
            continue
        try:
            trees[relative] = ast.parse(source)
        except SyntaxError as exc:
            errors.append(f"Local BA source {relative} cannot be parsed: {exc}")

    scene = trees.get("scene/scene_model.py")
    if scene is not None:
        commit = _method(scene, "SceneModel", "apply_local_ba_result")
        if not _has_parameters(commit, ("pose_updates", "landmark_updates")):
            errors.append(
                "SceneModel.apply_local_ba_result(pose_updates, landmark_updates) is missing or stale; "
                "the local BA solver cannot commit accepted poses/landmarks."
            )
        elif not _calls(commit, "_fit_gaussian_owner_keyframe_ids"):
            errors.append("SceneModel.apply_local_ba_result is missing Gaussian owner correction support.")

        owners = "gaussian_owner_keyframe_ids"
        if not _assigns_self_attribute(_method(scene, "SceneModel", "__init__"), owners):
            errors.append("SceneModel.__init__ is missing Gaussian creation-owner initialization.")
        fit = _method(scene, "SceneModel", "_fit_gaussian_owner_keyframe_ids")
        if not _assigns_self_attribute(fit, owners):
            errors.append("SceneModel._fit_gaussian_owner_keyframe_ids is missing or stale.")
        prune = _method(scene, "SceneModel", "_apply_gaussian_add_and_prune")
        if not _preserves_gaussian_owners(prune):
            errors.append(
                "SceneModel._apply_gaussian_add_and_prune is missing extension_owner_ids "
                "or owner preservation through add/prune."
            )
        if not _has_creation_owner_path(_method(scene, "SceneModel", "add_new_gaussians")):
            errors.append("SceneModel.add_new_gaussians is missing the Gaussian creation-owner path.")

    local_ba = trees.get("poses/local_ba.py")
    if local_ba is not None:
        run = _method(local_ba, "CovisibilityLocalBA", "run")
        if not (_calls(run, "solve_local_ba") and _calls(run, "apply_local_ba_result")):
            errors.append("CovisibilityLocalBA.run is missing the solve/scene commit interface.")
    solver = trees.get("poses/local_ba_solver.py")
    if solver is not None:
        solve = next((node for node in solver.body if isinstance(node, ast.FunctionDef)
                      and node.name == "solve_local_ba"), None)
        if not _has_parameters(solve, ("Rts", "points", "observations", "intrinsics", "num_optimized_cameras")):
            errors.append("poses/local_ba_solver.py is missing the joint solve_local_ba interface.")
    backend = trees.get("slam_backend.py")
    if backend is not None:
        if not _calls(_method(backend, "SLAMBackend", "__init__"), "CovisibilityLocalBA"):
            errors.append("SLAMBackend is missing CovisibilityLocalBA initialization.")
        register = _method(backend, "SLAMBackend", "register_incremental_keyframe")
        timed_dispatch = any(
            call.args and isinstance(call.args[0], ast.Attribute)
            and isinstance(call.args[0].value, ast.Name) and call.args[0].value.id == "self"
            and call.args[0].attr == "run_local_ba" for call in _calls(register, "_timed")
        )
        if not (_calls(register, "run_local_ba") or timed_dispatch):
            errors.append("SLAMBackend.register_incremental_keyframe is missing local BA dispatch.")


def _check_trajectory_sources(sources, errors):
    trees = {}
    for relative in TRAJECTORY_FILES:
        if relative not in sources:
            continue
        try:
            trees[relative] = ast.parse(sources[relative])
        except SyntaxError as exc:
            errors.append(f"Trajectory source {relative} cannot be parsed: {exc}")

    expected = ", ".join(EXPECTED_TRAJECTORY_STAGES)
    outputs = trees.get("slam_outputs.py")
    if outputs is not None:
        declarations = []
        for node in outputs.body:
            targets = node.targets if isinstance(node, ast.Assign) else (
                [node.target] if isinstance(node, ast.AnnAssign) else []
            )
            if any(isinstance(target, ast.Name) and target.id == "TRAJECTORY_STAGES"
                   for target in targets):
                declarations.append(node.value)
        stages = None
        if len(declarations) == 1:
            try:
                stages = ast.literal_eval(declarations[0])
            except (ValueError, TypeError, SyntaxError):
                pass
        if not isinstance(stages, (tuple, list)) or tuple(stages) != EXPECTED_TRAJECTORY_STAGES:
            errors.append(
                f"slam_outputs.py TRAJECTORY_STAGES must be exactly ({expected}); "
                "the legacy final_refinement stage is incompatible with the two-stage SLAM entry point."
            )

    entrypoint = trees.get("slam.py")
    if entrypoint is not None:
        calls = _calls(entrypoint, "write_trajectory_stage_outputs")
        if not calls:
            errors.append("slam.py is missing the two-stage write_trajectory_stage_outputs call.")
        for call in calls:
            records = call.args[1] if len(call.args) >= 2 else next(
                (keyword.value for keyword in call.keywords if keyword.arg == "stage_records"), None
            )
            keys = (
                [key.value if isinstance(key, ast.Constant) else None for key in records.keys]
                if isinstance(records, ast.Dict) else None
            )
            if (keys is None or len(keys) != len(EXPECTED_TRAJECTORY_STAGES)
                    or set(keys) != set(EXPECTED_TRAJECTORY_STAGES)):
                errors.append(
                    f"slam.py:{call.lineno} write_trajectory_stage_outputs must provide "
                    f"a literal stage dictionary with exactly ({expected})."
                )


def _read_sources(root, relatives):
    root = Path(root).resolve()
    report = {"root": str(root), "files": {}, "errors": []}
    sources = {}
    for relative in dict.fromkeys(relatives):
        path = (root / relative).resolve()
        entry = {"path": str(path), "sha256": None}
        report["files"][relative] = entry
        try:
            content = path.read_bytes()
            entry["sha256"] = hashlib.sha256(content).hexdigest()
            sources[relative] = content.decode("utf-8-sig")
        except (OSError, UnicodeDecodeError) as exc:
            report["errors"].append(f"Cannot read {path}: {exc}")
    return report, sources


def _finish_preflight(report, label):
    print(f"{label} source check: {report['root']}")
    for entry in report["files"].values():
        print(f"  {entry['path']}\n    SHA256 {entry['sha256'] or 'unavailable'}")
    if report["errors"]:
        details = "\n".join(f"  - {error}" for error in report["errors"])
        raise RuntimeError(
            f"{label} preflight failed; no SLAM sequence was started.\n"
            f"{details}\n"
            f"Synchronize all reported source files into {report['root']}, then restart Python."
        )


def check_trajectory_sources(root=ROOT):
    """Check only the two-stage producer/exporter contract, without importing them."""
    report, sources = _read_sources(root, TRAJECTORY_FILES)
    _check_trajectory_sources(sources, report["errors"])
    _finish_preflight(report, "Trajectory")
    print("Trajectory source interface OK: frontend_miniba and online_mapping only.")
    return report


def check_calibration_sources(root=ROOT, *, check_local_ba=False, check_trajectory_stages=False):
    """Return source paths/hashes or fail before dispatching a mixed build.

    Optional local-BA and trajectory checks are independent of the MiniBA
    calibration check. This is a deployment check, not a numerical/CUDA test.
    """
    relatives = (CALIBRATION_FILES + (LOCAL_BA_FILES if check_local_ba else ())
                 + (TRAJECTORY_FILES if check_trajectory_stages else ()))
    report, sources = _read_sources(root, relatives)

    mini_ba_source = sources.get("poses/mini_ba.py")
    if mini_ba_source is not None:
        try:
            tree = ast.parse(mini_ba_source)
        except SyntaxError as exc:
            report["errors"].append(f"MiniBA source cannot be parsed: {exc}")
        else:
            init = _method(tree, "MiniBA", "__init__")
            for variable, label in (("f_init", "focal"), ("centre", "principal point")):
                if not _camera_pair_sample(init, variable):
                    report["errors"].append(
                        f"MiniBA CUDA graph {label} sample ({variable}) must have shape (n_cams, 2); "
                        "the old [1] focal sample cannot accept bootstrap [8, 2]."
                    )
            if not _normalizes_before_replay(_method(tree, "MiniBA", "__call__")):
                report["errors"].append(
                    "MiniBA.__call__ is missing _camera_intrinsics normalization before graph replay."
                )

    if check_local_ba:
        _check_local_ba_sources(sources, report["errors"])
    if check_trajectory_stages:
        _check_trajectory_sources(sources, report["errors"])

    _finish_preflight(report, "Calibration")
    print("MiniBA source interface OK: focal and principal point samples are (n_cams, 2).")
    if check_local_ba:
        print("Local BA source interface OK: joint solve, scene commit, and Gaussian creation owners.")
    if check_trajectory_stages:
        print("Trajectory source interface OK: frontend_miniba and online_mapping only.")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="TEST2-SLAM checkout to inspect.")
    parser.add_argument("--check_local_ba", "--check-local-ba", action="store_true",
                        help="Also check/hash the local BA solver and scene correction interfaces.")
    parser.add_argument("--check_trajectory_stages", "--check-trajectory-stages", action="store_true",
                        help="Also check/hash the two-stage trajectory producer and exporter.")
    args = parser.parse_args(argv)
    try:
        check_calibration_sources(args.root, check_local_ba=args.check_local_ba,
                                  check_trajectory_stages=args.check_trajectory_stages)
    except RuntimeError as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
