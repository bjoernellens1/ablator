"""Passive observation through the existing owner; no alternate runner/reaper."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ablator import runner
from test_execution_receipt import _cfg, _repo, _wait_supervisor


def _receipt(cfg):
    paths = list(Path(cfg['queue']['log_dir']).glob('outer-lifecycle-cost-*.json'))
    assert len(paths) == 1, 'opted-in existing owner must publish one fresh receipt'
    return json.loads(paths[0].read_text())


def _job(sha, **metadata):
    return {'id': 'owner-job', 'type': 'replay', 'requested_git_sha': sha,
            'external_id': 'opaque-scheduler-job', 'external_metadata': {
                'record_outer_lifecycle_cost': True,
                'researchflow_run_id': 'opaque-run',
                'researchflow_plan_sha256': 'b' * 64,
                'researchflow_job_id': 'rf-job', **metadata}}


def test_owner_encloses_real_child_publication_and_source_release(tmp_path, monkeypatch):
    # Moving the clock into supervision or ahead of release must fail enclosure.
    repo, sha = _repo(tmp_path)
    journal = tmp_path / 'events.jsonl'
    output = tmp_path / 'durable-output'
    code = ('import pathlib,time,json; '
            f'j=pathlib.Path({str(journal)!r}); '
            'j.open("a").write(json.dumps(["child",time.monotonic_ns()])+"\\n"); '
            'time.sleep(.03); '
            f'pathlib.Path({str(output)!r}).write_text("persisted"); '
            'j.open("a").write(json.dumps(["published",time.monotonic_ns()])+"\\n")')
    cfg = _cfg(tmp_path, repo, [sys.executable, '-c', code])
    job = _job(sha)
    prepare, release = runner.sourcecheckout.prepare_job_source, runner.sourcecheckout.release_source

    def event(name):
        with journal.open('a') as handle:
            handle.write(json.dumps([name, time.monotonic_ns()]) + '\n')

    def delayed_prepare(*args):
        event('preparation-start')
        time.sleep(.03)
        return prepare(*args)

    def delayed_release(*args):
        event('release-start')
        time.sleep(.03)
        release(*args)
        event('release-finished')

    monkeypatch.setattr(runner.sourcecheckout, 'prepare_job_source', delayed_prepare)
    monkeypatch.setattr(runner.sourcecheckout, 'release_source', delayed_release)
    monkeypatch.setattr(runner, 'supervise', _wait_supervisor)
    before = time.monotonic_ns()
    assert runner.run_job(cfg, job, 'main') == ('done', 0)
    after = time.monotonic_ns()
    receipt = _receipt(cfg)
    events = dict(json.loads(line) for line in journal.read_text().splitlines())
    assert before <= receipt['started_monotonic_ns'] <= events['preparation-start']
    assert events['published'] <= receipt['child_wait_completed_monotonic_ns']
    assert events['release-finished'] <= receipt['closed_monotonic_ns'] <= after
    assert receipt['elapsed_seconds'] >= .09
    assert receipt['elapsed_seconds'] == (receipt['closed_monotonic_ns'] - receipt['started_monotonic_ns']) / 1e9
    assert receipt['closure']['host_owner'] == 'closed'
    assert receipt['closure']['workload'] == 'unknown'
    assert receipt['bindings']['execution_attestation']['verdict'] == 'ACCEPTED'
    assert receipt['identity']['researchflow_run_id'] == 'opaque-run'
    assert receipt['identity']['scheduler_job_id'] == 'owner-job'
    assert receipt['identity']['external_id'] == 'opaque-scheduler-job'
    assert receipt['resources']['cpu_seconds'] is None
    assert receipt['resources']['gpu_active_seconds'] is None
    assert receipt['resources']['energy_joules'] is None
    assert receipt['clock_domain']['status'] == 'known'
    assert receipt['clock_domain']['time_namespace'].startswith('time:[')
    assert all(type(receipt[k]) is int for k in ('started_monotonic_ns', 'closed_monotonic_ns', 'observed_through_monotonic_ns'))


@pytest.mark.parametrize('failure', ['launch', 'fit', 'cancel', 'release'])
def test_failures_preserve_existing_result_and_scope(tmp_path, monkeypatch, failure):
    repo, sha = _repo(tmp_path)
    command = ['/missing-executable'] if failure == 'launch' else [sys.executable, '-c', 'raise SystemExit(7)' if failure == 'fit' else 'pass']
    cfg = _cfg(tmp_path, repo, command)
    job = _job(sha)
    if failure == 'release':
        def fail_release(_prepared):
            raise runner.sourcecheckout.SourcePreparationError('release fault')
        monkeypatch.setattr(runner.sourcecheckout, 'release_source', fail_release)
    def supervise(*args, **kwargs):
        _wait_supervisor(*args, **kwargs)
        return 'cancelled' if failure == 'cancel' else None
    monkeypatch.setattr(runner, 'supervise', supervise)
    expected = {'launch': ('failed', None), 'fit': ('failed', 7), 'cancel': ('cancelled', 0), 'release': ('failed', 0)}[failure]
    assert runner.run_job(cfg, job, 'main') == expected
    receipt = _receipt(cfg)
    assert receipt['closure']['workload'] == 'unknown'
    assert receipt['outcome']['status'] == expected[0]
    if failure == 'launch':
        assert receipt['child_started_monotonic_ns'] is None
        assert 'missing-executable' in receipt['outcome']['error']
    if failure == 'release':
        assert receipt['closed_monotonic_ns'] is None
        assert receipt['elapsed_seconds'] is None
        assert receipt['closure']['source_release'] == 'unknown'


def test_default_does_not_publish_and_repeated_invocations_do_not_overwrite(tmp_path, monkeypatch):
    repo, sha = _repo(tmp_path)
    cfg = _cfg(tmp_path, repo, [sys.executable, '-c', 'pass'])
    monkeypatch.setattr(runner, 'supervise', _wait_supervisor)
    job = _job(sha, record_outer_lifecycle_cost=False)
    assert runner.run_job(cfg, job, 'main') == ('done', 0)
    assert not list(Path(cfg['queue']['log_dir']).glob('outer-lifecycle-cost-*'))
    job['external_metadata']['record_outer_lifecycle_cost'] = True
    assert runner.run_job(cfg, job, 'main') == ('done', 0)
    first = next(Path(cfg['queue']['log_dir']).glob('outer-lifecycle-cost-*'))
    first_bytes = first.read_bytes()
    assert runner.run_job(cfg, job, 'main') == ('done', 0)
    assert len(list(Path(cfg['queue']['log_dir']).glob('outer-lifecycle-cost-*'))) == 2
    assert first.read_bytes() == first_bytes


def test_observed_bindings_are_local_to_invocation_not_stale_job_values(tmp_path, monkeypatch):
    repo, sha = _repo(tmp_path)
    cfg = _cfg(tmp_path, repo, ['/missing-executable'])
    job = _job(sha)
    job['actual_launch'] = {'stale': True}
    assert runner.run_job(cfg, job, 'main') == ('failed', None)
    assert _receipt(cfg)['bindings']['actual_launch'] is None


def test_canonical_rf_raw_bytes_observed_without_claiming_complete_roster(tmp_path, monkeypatch):
    repo, sha = _repo(tmp_path)
    jobs = tmp_path / 'state' / 'jobs'
    jobs.mkdir(parents=True)
    for attempt_id, stamp, rf_job in [('prior-failed', 12, 'rf-job'), ('terminal', 34, 'rf-job'), ('foreign', 1, 'another-job')]:
        directory = jobs / 'attempts' / attempt_id
        directory.mkdir(parents=True)
        record = dict(schema='researchflow.execution-attempt/v1', attempt_id=attempt_id,
                      job_id=rf_job, run_id='opaque-run', plan_sha256='b' * 64,
                      attempt_started_monotonic_ns=stamp)
        (directory / 'attempt.json').write_text(json.dumps(record))
        (directory / 'execution.json').write_text('{"opaque":"raw bytes"}\n')
    (jobs / 'rf-job.execution.json').symlink_to('attempts/terminal/execution.json')
    cfg = _cfg(tmp_path, repo, [sys.executable, '-c', 'pass'])
    job = _job(sha, researchflow_origin_path=str(jobs / 'rf-job.origin.json'))
    monkeypatch.setattr(runner, 'supervise', _wait_supervisor)
    assert runner.run_job(cfg, job, 'main') == ('done', 0)
    receipt = _receipt(cfg)
    assert [a['attempt_id'] for a in receipt['observed_attempts']] == ['prior-failed', 'terminal']
    for attempt in receipt['observed_attempts']:
        for role in ['record', 'execution']:
            ref = attempt[role]
            assert hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest() == ref['sha256']
    assert receipt['attempt_roster'] == 'observed_only'
    assert receipt['durable_artifacts'][0]['path'] == str(jobs / 'attempts' / 'terminal' / 'execution.json')


def test_publication_failure_does_not_replace_first_execution_error(tmp_path, monkeypatch, capsys):
    from ablator import lifecycle_cost
    repo, sha = _repo(tmp_path)
    cfg = _cfg(tmp_path, repo, ['/missing-executable'])
    real_open = lifecycle_cost.os.open
    def fail_receipt(path, *args, **kwargs):
        if 'outer-lifecycle-cost-' in str(path):
            raise OSError('publication fault')
        return real_open(path, *args, **kwargs)
    monkeypatch.setattr(lifecycle_cost.os, 'open', fail_receipt)
    assert runner.run_job(cfg, _job(sha), 'main') == ('failed', None)
    output = capsys.readouterr().out
    assert 'missing-executable' in output
    assert 'publication fault' in output
    assert not list(Path(cfg['queue']['log_dir']).glob('outer-lifecycle-cost-*'))


def test_nested_opaque_child_never_qualifies_workload_closure(tmp_path, monkeypatch):
    repo, sha = _repo(tmp_path)
    descendant_done = tmp_path / 'descendant-done'
    descendant = f'import time,pathlib; time.sleep(.3); pathlib.Path({str(descendant_done)!r}).write_text("done")'
    code = f'import subprocess,sys; subprocess.Popen([sys.executable,"-c",{descendant!r}],start_new_session=True)'
    cfg = _cfg(tmp_path, repo, [sys.executable, '-c', code])
    monkeypatch.setattr(runner, 'supervise', _wait_supervisor)
    assert runner.run_job(cfg, _job(sha), 'main') == ('done', 0)
    assert _receipt(cfg)['closure']['workload'] == 'unknown'
    deadline = time.monotonic() + 5
    while not descendant_done.exists() and time.monotonic() < deadline:
        time.sleep(.02)
    assert descendant_done.exists(), 'bounded test descendant must finish'


def test_supervisor_unwind_preserves_interrupt_and_unknown_close(tmp_path, monkeypatch):
    repo, sha = _repo(tmp_path)
    cfg = _cfg(tmp_path, repo, [sys.executable, '-c', 'pass'])
    def interrupted(_cfg, _job, proc, *_args, **_kwargs):
        proc.wait(timeout=5)
        raise KeyboardInterrupt('original interruption')
    monkeypatch.setattr(runner, 'supervise', interrupted)
    with pytest.raises(KeyboardInterrupt, match='original interruption'):
        runner.run_job(cfg, _job(sha), 'main')
    receipt = _receipt(cfg)
    assert receipt['closure']['host_owner'] == 'unknown'
    assert receipt['closed_monotonic_ns'] is None
    assert receipt['observed_seconds'] > 0


def test_changed_clock_domain_does_not_admit_verified_cross_boundary_clock(tmp_path, monkeypatch):
    from ablator import lifecycle_cost
    repo, sha = _repo(tmp_path)
    cfg = _cfg(tmp_path, repo, [sys.executable, '-c', 'pass'])
    monkeypatch.setattr(runner, 'supervise', _wait_supervisor)
    real_domain = lifecycle_cost._clock_domain
    calls = 0
    def changing_domain():
        nonlocal calls
        value = real_domain()
        calls += 1
        if calls > 1:
            value['time_namespace'] = 'time:[99999999]'
        return value
    monkeypatch.setattr(lifecycle_cost, '_clock_domain', changing_domain)
    assert runner.run_job(cfg, _job(sha), 'main') == ('done', 0)
    assert _receipt(cfg)['clock_domain']['status'] == 'unavailable'


def test_real_cancellation_closes_host_owner_after_existing_kill_path(tmp_path, monkeypatch):
    repo, sha = _repo(tmp_path)
    cfg = _cfg(tmp_path, repo, [sys.executable, '-c', 'import time; time.sleep(30)'])
    def cancel(_cfg, job, proc, *_args, argv, **_kwargs):
        runner.kill_job(proc, job, argv)
        proc.wait(timeout=5)
        return 'cancelled'
    monkeypatch.setattr(runner, 'supervise', cancel)
    result = runner.run_job(cfg, _job(sha), 'main')
    assert result[0] == 'cancelled'
    assert result[1] != 0
    receipt = _receipt(cfg)
    assert receipt['closure']['direct_child'] == 'reaped'
    assert receipt['closure']['host_owner'] == 'closed'
    assert receipt['closure']['workload'] == 'unknown'


def test_first_execution_error_retained_when_source_release_also_fails(tmp_path, monkeypatch):
    repo, sha = _repo(tmp_path)
    cfg = _cfg(tmp_path, repo, ['/missing-executable'])
    def fail_release(_prepared):
        raise runner.sourcecheckout.SourcePreparationError('later release fault')
    monkeypatch.setattr(runner.sourcecheckout, 'release_source', fail_release)
    assert runner.run_job(cfg, _job(sha), 'main') == ('failed', None)
    receipt = _receipt(cfg)
    assert 'missing-executable' in receipt['outcome']['error']
    assert receipt['closed_monotonic_ns'] is None


def test_rejected_final_source_attestation_records_original_reason(tmp_path, monkeypatch):
    repo, sha = _repo(tmp_path)
    cfg = _cfg(tmp_path, repo, [sys.executable, '-c', 'from pathlib import Path; Path("payload.txt").write_text("changed")'])
    monkeypatch.setattr(runner, 'supervise', _wait_supervisor)
    assert runner.run_job(cfg, _job(sha), 'main') == ('failed', 0)
    receipt = _receipt(cfg)
    assert receipt['outcome']['error'] is not None
    assert 'dirty' in receipt['outcome']['error']
    assert receipt['closure']['workload'] == 'unknown'


@pytest.mark.parametrize('flag', [False, 1, 'true', {}])
def test_opt_in_requires_literal_true_and_has_no_lifecycle_io(flag, monkeypatch):
    from ablator import lifecycle_cost
    def forbidden():
        raise AssertionError('normal jobs must not read lifecycle clock identity')
    monkeypatch.setattr(lifecycle_cost, '_clock_domain', forbidden)
    assert lifecycle_cost.start({'external_metadata': {'record_outer_lifecycle_cost': flag}}, 'main') is None


def _artifact_collector(tmp_path, *, alias_to_record=False):
    from ablator import lifecycle_cost
    state = tmp_path / 'jobs'
    directory = state / 'attempts' / 'attempt'
    directory.mkdir(parents=True)
    record = directory / 'attempt.json'
    record.write_text(json.dumps(dict(schema='researchflow.execution-attempt/v1',
        attempt_id='attempt', job_id='job', run_id='run', plan_sha256='b' * 64,
        attempt_started_monotonic_ns=1)))
    execution = directory / 'execution.json'
    execution.write_bytes(b'x' * (8 * 1024 * 1024))
    target = record if alias_to_record else execution
    (state / 'job.execution.json').symlink_to(target)
    collector = lifecycle_cost.start({'external_metadata': {
        'record_outer_lifecycle_cost': True, 'researchflow_job_id': 'job',
        'researchflow_run_id': 'run', 'researchflow_plan_sha256': 'b' * 64,
        'researchflow_origin_path': str(state / 'job.origin.json')}}, 'cpu-only')
    return collector, record, execution


def test_hash_only_alias_is_read_once_per_collection(tmp_path, monkeypatch):
    collector, _record, execution = _artifact_collector(tmp_path)
    real_open = Path.open
    reads = []
    def counted_open(path, *args, **kwargs):
        if path.resolve() == execution:
            reads.append(path)
        return real_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', counted_open)
    attempts, artifacts, errors = collector._artifacts()
    assert not errors
    assert len(reads) == 1
    assert attempts[0]['execution'] == artifacts[0]


def test_hash_only_large_artifact_has_bounded_temporary_allocation(tmp_path):
    import tracemalloc
    collector, _record, _execution = _artifact_collector(tmp_path)
    tracemalloc.start()
    try:
        attempts, artifacts, errors = collector._artifacts()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert attempts and artifacts and not errors
    assert peak < 1024 * 1024, 'hash-only 8 MiB input must use bounded streaming buffers'


def test_hash_only_then_parsed_replacement_is_not_accepted(tmp_path, monkeypatch):
    collector, record, _execution = _artifact_collector(tmp_path, alias_to_record=True)
    real_read = Path.read_bytes
    def replaced_read(path):
        if path.resolve() == record:
            value = json.loads(record.read_text())
            value['attempt_started_monotonic_ns'] = 2
            record.write_text(json.dumps(value))
        return real_read(path)
    monkeypatch.setattr(Path, 'read_bytes', replaced_read)
    attempts, artifacts, errors = collector._artifacts()
    assert not attempts
    assert artifacts
    assert any('changed during collection' in error for error in errors)
