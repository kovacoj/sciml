"""Worker pool: manages DAFoam residual worker processes.

Uses multiprocessing.Queue for result communication to avoid pipe
buffer deadlocks with concurrent workers.  Commands are still sent
via Pipe, but results go through a shared Queue.
"""
from __future__ import annotations

import multiprocessing
import time
from dataclasses import dataclass
from typing import List

import numpy as np

from .context import PreparedTopology
from .residual_worker import worker_main


@dataclass
class _WorkerEntry:
    topology_id: str
    process: multiprocessing.Process
    pipe: multiprocessing.Pipe
    result_queue: multiprocessing.Queue


class TopologyWorkerPool:
    """Manages transient DAFoam residual worker processes.

    Parameters
    ----------
    prepared_topologies
        All topologies that will ever be evaluated.
    max_concurrent
        Maximum number of workers alive at the same time.
    timeout_seconds
        Per-worker init timeout and per-evaluation-wave timeout.
    """

    def __init__(self, prepared_topologies: List[PreparedTopology],
                 max_concurrent: int = 4,
                 timeout_seconds: float = 1800.0):
        self.topologies = {t.topology_id: t for t in prepared_topologies}
        self.max_concurrent = max_concurrent
        self.timeout = timeout_seconds
        self._workers: dict[str, _WorkerEntry] = {}
        self._ctx = multiprocessing.get_context("spawn")
        self._result_queue: multiprocessing.Queue | None = None

    def start_wave(self, topology_ids: List[str]):
        """Start workers for one wave of topology IDs."""
        self._result_queue = self._ctx.Queue()
        for tid in topology_ids:
            if tid in self._workers:
                continue
            topo = self.topologies[tid]
            parent_conn, child_conn = self._ctx.Pipe(duplex=True)
            p = self._ctx.Process(
                target=worker_main,
                args=(topo.case_dir, tid,
                      topo.u_ids, topo.p_ids, topo.phi_ids,
                      topo.loss_config.gamma_u,
                      topo.loss_config.gamma_p,
                      topo.loss_config.gamma_phi,
                      child_conn,
                      topo.inlet_patches or ["inlet"],
                      topo.outlet_patches or ["outlet"],
                      self._result_queue),
            )
            p.daemon = False
            p.start()

            if not parent_conn.poll(self.timeout):
                if not p.is_alive():
                    raise RuntimeError(
                        f"Worker {tid} exited during initialization "
                        f"with code {p.exitcode}")
                p.terminate()
                p.join(timeout=5)
                raise TimeoutError(
                    f"Worker {tid} did not initialize within "
                    f"{self.timeout} seconds")

            msg = parent_conn.recv()
            if msg["status"] != "ready":
                raise RuntimeError(
                    f"Worker {tid} initialization failed:\n"
                    f"{msg.get('error', '')}\n"
                    f"{msg.get('traceback', '')}")

            self._workers[tid] = _WorkerEntry(tid, p, parent_conn, self._result_queue)
            print(f"[pool] worker {tid} ready (pid={p.pid})", flush=True)

    def evaluate_wave(self, topology_ids: List[str],
                      states: List[np.ndarray]) -> List[dict]:
        """Evaluate residuals for one wave of topologies.

        Sends commands via pipe, receives results via shared Queue.
        """
        import os
        import tempfile

        n = len(topology_ids)
        results = [None] * n
        state_paths = []

        for i, (tid, state) in enumerate(zip(topology_ids, states)):
            tmpdir = tempfile.mkdtemp(prefix=f"wave_{tid}_")
            state_path = os.path.join(tmpdir, "state.npy")
            np.save(state_path, state)
            state_paths.append(state_path)

            self._workers[tid].pipe.send({
                "command": "evaluate",
                "request_id": i,
                "state_path": state_path,
            })

        pending = {i: tid for i in range(n)}
        deadline = time.time() + self.timeout

        while pending:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(
                    f"Worker pool timed out with {len(pending)} pending "
                    f"(timeout={self.timeout}s)")

            try:
                msg = self._result_queue.get(timeout=0.5)
            except Exception:
                for rid, tid in list(pending.items()):
                    entry = self._workers[tid]
                    if not entry.process.is_alive():
                        raise RuntimeError(
                            f"Worker {tid} died (exitcode="
                            f"{entry.process.exitcode})")
                continue

            if msg["status"] == "ok":
                rid = msg["request_id"]
                grad_path = msg.pop("grad_path")
                msg["grad_state"] = np.load(grad_path)
                os.remove(grad_path)
                os.remove(state_paths[rid])
                os.rmdir(os.path.dirname(grad_path))
                results[rid] = msg
                tid = topology_ids[rid]
                del pending[rid]
                print(f"[pool] received result from {tid} "
                      f"(rid={rid})", flush=True)
            elif msg["status"] == "error":
                raise RuntimeError(
                    f"Worker error: {msg.get('error')}\n"
                    f"{msg.get('traceback', '')}")
            else:
                raise RuntimeError(f"Unexpected worker response: {msg}")

        return results

    def simple_step_wave(self, topology_ids: List[str],
                         states: List[np.ndarray]) -> List[np.ndarray]:
        """Run one SIMPLE step for each topology in the wave.

        Returns the updated states.
        """
        import os
        import tempfile

        n = len(topology_ids)
        results = [None] * n
        state_paths = []

        for i, (tid, state) in enumerate(zip(topology_ids, states)):
            tmpdir = tempfile.mkdtemp(prefix=f"simple_{tid}_")
            state_path = os.path.join(tmpdir, "state.npy")
            np.save(state_path, state)
            state_paths.append(state_path)

            self._workers[tid].pipe.send({
                "command": "simple_step",
                "request_id": i,
                "state_path": state_path,
            })

        pending = {i: tid for i in range(n)}
        deadline = time.time() + self.timeout

        while pending:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(
                    f"Simple step wave timed out with {len(pending)} pending")

            try:
                msg = self._result_queue.get(timeout=0.5)
            except Exception:
                for rid, tid in list(pending.items()):
                    entry = self._workers[tid]
                    if not entry.process.is_alive():
                        raise RuntimeError(
                            f"Worker {tid} died (exitcode="
                            f"{entry.process.exitcode})")
                continue

            if msg["status"] == "ok":
                rid = msg["request_id"]
                next_path = msg["next_path"]
                results[rid] = np.load(next_path)
                os.remove(next_path)
                os.remove(state_paths[rid])
                os.rmdir(os.path.dirname(next_path))
                del pending[rid]
            elif msg["status"] == "error":
                raise RuntimeError(
                    f"Worker error: {msg.get('error')}\n"
                    f"{msg.get('traceback', '')}")

        return results

    def close_wave(self):
        for tid, entry in self._workers.items():
            try:
                entry.pipe.send({"command": "close"})
                entry.process.join(timeout=10)
            except Exception:
                pass
            if entry.process.is_alive():
                entry.process.terminate()
                entry.process.join(timeout=5)
        self._workers.clear()
        self._result_queue = None

    def close(self):
        """Shut down all workers (alias for close_wave)."""
        self.close_wave()
