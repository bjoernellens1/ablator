"""Cold CPU-only receipt measurements; no campaign or device-speed inference.

Run with OUTER_COST_BENCHMARK_JSON to retain raw samples. Fixture construction,
receipt inspection and cleanup are outside measured intervals.
"""
import ast
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import pytest

from ablator import lifecycle_cost


def test_leaf_has_only_passive_stdlib_dependencies():
    tree = ast.parse(Path(lifecycle_cost.__file__).read_text())
    allowed = {'__future__', 'hashlib', 'json', 'os', 're', 'sys', 'time', 'pathlib', 'uuid'}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [node.module] if isinstance(node, ast.ImportFrom) else [n.name for n in node.names]
            assert all(name.split('.')[0] in allowed for name in names)
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, 'id', '')
            assert name not in {'synchronize', 'wait', 'sleep', 'Popen', 'run_job', 'dispatch', 'train', 'fit', 'backward'}


def test_default_off_returns_before_clock_constructor_or_scan(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('default-off lifecycle observation invoked work')
    monkeypatch.setattr(lifecycle_cost, 'LifecycleCost', forbidden)
    monkeypatch.setattr(lifecycle_cost.time, 'monotonic_ns', forbidden)
    monkeypatch.setattr(Path, 'glob', forbidden)
    monkeypatch.setattr(Path, 'read_bytes', forbidden)
    for metadata in (None, {}, {'record_outer_lifecycle_cost': False}, {'record_outer_lifecycle_cost': 1}):
        assert lifecycle_cost.start({'external_metadata': metadata}, 'cpu-only') is None


def test_runner_lifecycle_hooks_are_guarded_and_outside_loops():
    source = Path(lifecycle_cost.__file__).with_name('runner.py')
    tree = ast.parse(source.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_job')
    parents = {child: parent for parent in ast.walk(owner) for child in ast.iter_child_nodes(parent)}
    hooks = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not isinstance(node.func.value, ast.Name):
            continue
        if node.func.value.id not in {'lifecyclecost', 'lifecycle'}:
            continue
        hooks.append(node)
        ancestors = []
        cursor = node
        while cursor in parents:
            cursor = parents[cursor]
            ancestors.append(cursor)
        assert owner in ancestors, 'lifecycle hook escaped run_job'
        assert not any(isinstance(n, (ast.For, ast.While, ast.AsyncFor)) for n in ancestors)
        if node.func.value.id == 'lifecycle':
            assert any(isinstance(n, ast.If) and 'lifecycle is not None' in ast.unparse(n.test) for n in ancestors)
    assert len(hooks) == 2  # one cheap start gate, one post-child publish


@pytest.mark.parametrize('count', [1, 32, 256])
def test_bounded_cpu_collection_and_serialization(tmp_path, monkeypatch, count):
    state = tmp_path / 'jobs'
    state.mkdir()
    for index in range(count):
        attempt_id = f'attempt-{index:04d}'
        directory = state / 'attempts' / attempt_id
        directory.mkdir(parents=True)
        record = dict(schema='researchflow.execution-attempt/v1', attempt_id=attempt_id,
                      job_id='job', run_id='run', plan_sha256='b' * 64,
                      attempt_started_monotonic_ns=index)
        (directory / 'attempt.json').write_text(json.dumps(record))
        (directory / 'execution.json').write_bytes(b'{"raw":"retained execution"}\n')
    job = {'id': 'owner', 'external_metadata': {
        'record_outer_lifecycle_cost': True, 'researchflow_job_id': 'job',
        'researchflow_run_id': 'run', 'researchflow_plan_sha256': 'b' * 64,
        'researchflow_origin_path': str(state / 'job.origin.json')}}
    collector = lifecycle_cost.start(job, 'cpu-only')
    files = list(state.rglob('*.json'))
    input_bytes = sum(path.stat().st_size for path in files)
    reads, opens = [], []
    original_open = Path.open
    class Reader:
        def __init__(self, handle, path): self.handle, self.path = handle, path
        def __enter__(self): return self
        def __exit__(self, *args): return self.handle.__exit__(*args)
        def read(self, size=-1):
            raw = self.handle.read(size)
            reads.append((str(self.path), len(raw)))
            return raw
        def __getattr__(self, name): return getattr(self.handle, name)
    def counted_open(path, *args, **kwargs):
        opens.append(str(path))
        return Reader(original_open(path, *args, **kwargs), path)
    with monkeypatch.context() as probe:
        probe.setattr(Path, 'open', counted_open)
        collector._artifacts()
    collection, publication, output_bytes = [], [], []
    # Each collection is real local read+parse+hash; warm-cache samples, no fsync.
    for _ in range(9):
        began = time.perf_counter_ns()
        captured = collector._artifacts()
        collection.append(time.perf_counter_ns() - began)
        assert len(captured[0]) == count
        assert not captured[2]
    # Keep raw-byte binding outside this separate serialization/write interval.
    monkeypatch.setattr(collector, '_artifacts', lambda: captured)
    for sample in range(9):
        log_dir = tmp_path / f'log-{sample}'
        log_dir.mkdir()
        began = time.perf_counter_ns()
        collector.publish(str(log_dir), bindings={}, status='done', exit_code=0,
                          error=None, proc=None, source_release='not_acquired', owner_unwinding=False)
        publication.append(time.perf_counter_ns() - began)
        receipts = list(log_dir.glob('outer-lifecycle-cost-*.json'))
        assert len(receipts) == 1
        raw = receipts[0].read_bytes()
        output_bytes.append(len(raw))
        receipt = json.loads(raw)
        assert receipt['observed_seconds'] > 0
        assert receipt['elapsed_seconds'] > 0
        assert all(receipt['resources'][key] is None for key in ('cpu_seconds', 'gpu_active_seconds', 'energy_joules'))
        assert len(receipt['observed_attempts']) == count
    evidence_path = os.environ.get('OUTER_COST_BENCHMARK_JSON')
    if evidence_path:
        destination = Path(evidence_path)
        data = json.loads(destination.read_text()) if destination.exists() else {}
        data[str(count)] = dict(input_files=len(files), input_bytes=input_bytes,
            observed_open_calls=len(opens), observed_read_calls=len(reads), observed_read_bytes=sum(size for _, size in reads),
            collection_ns=collection, collection_median_ns=statistics.median(collection),
            serialization_write_ns=publication, serialization_write_median_ns=statistics.median(publication),
            output_files_per_sample=1, output_bytes=output_bytes,
            scope='CPU local warm-cache microbenchmark; fixture construction excluded; no fsync; no GPU/campaign inference',
            source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                (Path(lifecycle_cost.__file__), Path(lifecycle_cost.__file__).with_name('runner.py'), Path(__file__))})
        destination.write_text(json.dumps(data, indent=2) + '\n')


def test_hash_only_large_artifact_and_duplicate_target_probe(tmp_path, monkeypatch):
    import tracemalloc
    state = tmp_path / 'jobs'
    attempt = state / 'attempts' / 'a'
    attempt.mkdir(parents=True)
    (attempt / 'attempt.json').write_text(json.dumps(dict(
        schema='researchflow.execution-attempt/v1', attempt_id='a', job_id='job',
        run_id='run', plan_sha256='b' * 64, attempt_started_monotonic_ns=1)))
    large = attempt / 'execution.json'
    large.write_bytes(b'x' * (8 * 1024 * 1024))
    alias = state / 'job.execution.json'
    alias.symlink_to('attempts/a/execution.json')
    collector = lifecycle_cost.start({'external_metadata': {
        'record_outer_lifecycle_cost': True, 'researchflow_job_id': 'job',
        'researchflow_run_id': 'run', 'researchflow_plan_sha256': 'b' * 64,
        'researchflow_origin_path': str(state / 'job.origin.json')}}, 'cpu-only')
    opens = []
    original_open = Path.open
    def counted_open(path, *args, **kwargs):
        opens.append(str(path.resolve()))
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', counted_open)
    tracemalloc.start()
    captured = collector._artifacts()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert captured[0][0]['execution'] == captured[1][0]
    assert alias.stat().st_ino == large.stat().st_ino
    assert opens.count(str(large.resolve())) == 1, 'same resolved hash-only target must be bound once'
    assert peak < 1024 * 1024, '8 MiB hash-only artifact must use bounded streaming memory'
    destination = os.environ.get('OUTER_COST_BENCHMARK_JSON')
    if destination:
        path = Path(destination)
        data = json.loads(path.read_text())
        data['large_hash_only_probe'] = dict(artifact_bytes=large.stat().st_size,
            tracemalloc_peak_bytes=peak, alias=str(alias), target=str(large),
            resolved_path=str(alias.resolve()), alias_inode=alias.stat().st_ino,
            target_inode=large.stat().st_ino, device=large.stat().st_dev,
            target_open_count=opens.count(str(large.resolve())), hash_only=True, scope='8 MiB CPU fixture, allocation tracing outside timed benchmark')
        path.write_text(json.dumps(data, indent=2) + '\n')
