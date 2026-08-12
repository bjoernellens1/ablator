"""Tests for runner.py's mandatory output-folder preflight check (free
space + write speed), run before every job dispatch."""
import os

from ablator import runner


def test_disk_free_bytes_returns_positive_for_existing_dir(tmp_path):
    free = runner._disk_free_bytes(str(tmp_path))
    assert free is not None
    assert free > 0


def test_disk_free_bytes_none_for_nonexistent_path():
    assert runner._disk_free_bytes("/definitely/does/not/exist/xyz") is None


def test_measure_write_speed_returns_positive_and_cleans_up(tmp_path):
    speed = runner._measure_write_speed_mb_s(str(tmp_path), size_mb=1)
    assert speed is not None
    assert speed > 0
    # the probe file must not be left behind
    assert list(tmp_path.iterdir()) == []


def test_measure_write_speed_none_for_unwritable_path():
    assert runner._measure_write_speed_mb_s("/proc/definitely-not-writable") is None


def test_output_folder_preflight_resolves_relative_model_path(tmp_path):
    line = runner.output_folder_preflight("output/run1", str(tmp_path))
    resolved = os.path.join(str(tmp_path), "output/run1")
    assert resolved in line
    assert "free=" in line
    assert "write_speed=" in line


def test_output_folder_preflight_uses_absolute_model_path_verbatim(tmp_path):
    abs_path = str(tmp_path / "abs_run")
    line = runner.output_folder_preflight(abs_path, cwd="/some/other/cwd")
    assert abs_path in line
    assert "/some/other/cwd" not in line


def test_output_folder_preflight_never_raises_on_bad_path():
    # Must be best-effort -- a preflight check can never itself fail a job.
    line = runner.output_folder_preflight("/proc/impossible/path", cwd=None)
    assert "free=unknown" in line or "free=" in line


# ------------------------------------------- container --name auto-injection

def test_ensure_container_name_injects_for_docker_run():
    argv = ["docker", "run", "--rm", "--gpus", "all", "image", "python", "train.py"]
    out = runner._ensure_container_name(argv, {"id": "my_job_1"})
    assert out[:3] == ["docker", "run", "--name"]
    assert out[3] == "splat_train_my_job_1"
    assert out[4:] == ["--rm", "--gpus", "all", "image", "python", "train.py"]


def test_ensure_container_name_injects_for_podman_run():
    argv = ["podman", "run", "--rm", "image"]
    out = runner._ensure_container_name(argv, {"id": "abc"})
    assert out == ["podman", "run", "--name", "splat_train_abc", "--rm", "image"]


def test_ensure_container_name_respects_existing_explicit_name():
    argv = ["docker", "run", "--rm", "--name", "custom_name", "image"]
    out = runner._ensure_container_name(argv, {"id": "abc"})
    assert out == argv  # unchanged


def test_ensure_container_name_respects_existing_name_equals_form():
    argv = ["docker", "run", "--rm", "--name=custom_name", "image"]
    out = runner._ensure_container_name(argv, {"id": "abc"})
    assert out == argv  # unchanged


def test_ensure_container_name_noop_for_non_container_commands():
    argv = ["python", "some_script.py"]
    assert runner._ensure_container_name(argv, {"id": "abc"}) == argv


def test_ensure_container_name_noop_for_docker_non_run_subcommand():
    argv = ["docker", "ps", "-a"]
    assert runner._ensure_container_name(argv, {"id": "abc"}) == argv


def test_ensure_container_name_sanitizes_unsafe_job_id_characters():
    argv = ["docker", "run", "image"]
    out = runner._ensure_container_name(argv, {"id": "job/with:bad.chars"})
    injected_name = out[3]
    assert injected_name.startswith("splat_train_")
    assert "/" not in injected_name and ":" not in injected_name


def test_injected_name_is_findable_by_the_splat_train_busy_guard():
    # This is the actual bug: the busy_guard greps `docker ps` output for
    # "splat_train". Assert the injected name would satisfy it.
    argv = ["docker", "run", "--rm", "image"]
    out = runner._ensure_container_name(argv, {"id": "rspixel10a_offline_continue"})
    name = runner.container_name_from_argv(out)
    assert name is not None
    assert "splat_train" in name
