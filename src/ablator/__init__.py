"""ablator — cross-machine ablation/experiment queue orchestrator.

Stdlib-only: one shared flock'd JSONL queue on any shared filesystem,
one host runner per machine, containerized workloads defined entirely
by command templates in a host config file.
"""
__version__ = "0.1.0"
