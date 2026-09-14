import fcntl
import json
import multiprocessing
import threading
import time

from ablator.queue import Queue


def _read_queue_in_child(path, started, connection):
    try:
        started.set()
        connection.send(("ok", Queue(path).read()))
    except Exception as exc:  # pragma: no cover - reported to the parent process
        connection.send(("error", type(exc).__name__, str(exc)))
    finally:
        connection.close()


def _hold_shared_lock(path, acquired, release):
    with open(path) as handle:
        fcntl.flock(handle, fcntl.LOCK_SH)
        acquired.set()
        release.wait(timeout=3)


def test_read_waits_for_a_locked_partial_rewrite(tmp_path):
    path = tmp_path / "queue.jsonl"
    expected = [{"id": "complete", "status": "pending"}]
    path.write_text(json.dumps(expected[0]) + "\n")
    queue = Queue(str(path))
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    started = context.Event()

    with queue._open_locked() as handle:
        handle.seek(0)
        handle.truncate()
        handle.write('{"id": "partial"')
        handle.flush()

        reader = context.Process(
            target=_read_queue_in_child,
            args=(str(path), started, child),
        )
        reader.start()
        child.close()

        assert started.wait(timeout=3)
        time.sleep(0.1)

        assert reader.is_alive(), "read observed the partial JSONL rewrite"
        assert not parent.poll(), "read returned before the writer released its lock"

        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(expected[0]) + "\n")
        handle.flush()

    reader.join(timeout=3)
    assert reader.exitcode == 0
    assert parent.poll()
    assert parent.recv() == ("ok", expected)


def test_read_shares_the_queue_lock_with_another_reader(tmp_path):
    path = tmp_path / "queue.jsonl"
    expected = [{"id": "complete", "status": "pending"}]
    path.write_text(json.dumps(expected[0]) + "\n")
    context = multiprocessing.get_context("spawn")
    acquired = context.Event()
    release = context.Event()
    holder = context.Process(target=_hold_shared_lock, args=(str(path), acquired, release))
    holder.start()
    assert acquired.wait(timeout=3)

    result = []
    done = threading.Event()

    def read():
        result.append(Queue(str(path)).read())
        done.set()

    reader = threading.Thread(target=read)
    reader.start()
    try:
        assert done.wait(timeout=0.5), "read did not share the reader lock"
        assert result == [expected]
    finally:
        release.set()
        reader.join(timeout=3)
        holder.join(timeout=3)
    assert holder.exitcode == 0
