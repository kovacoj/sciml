"""Worker pool: manages DAFoam residual worker processes."""
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


class TopologyWorkerPool:
    """Manages persistent DAFoam residual worker processes."""

    def __init__(self, prepared_topologies: List[PreparedTopology],
                 timeout_seconds: float = 300.0):
        self.topologies = {t.topology_id: t for t in prepared_topologies}
        self.timeout = timeout_seconds
        self._workers: dict[str, _WorkerEntry] = {}
        self._ctx = multiprocessing.get_context("spawn")

    def start(self, topology_ids: List[str]):
        """Start workers for the given topology IDs."""
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
                      child_conn),
            )
            p.daemon = False
            p.start()
            # Wait for ready signal
            msg = parent_conn.recv()
            if msg["status"] != "ready":
                raise RuntimeError(f"Worker {tid} failed to start: {msg}")
            self._workers[tid] = _WorkerEntry(tid, p, parent_conn)
            print(f"[pool] worker {tid} ready (pid={p.pid})")

    def evaluate(self, topology_ids: List[str],
                 states: List[np.ndarray]) -> List[dict]:
        """Send states to workers and collect results."""
        request_id = 0
        results = [None] * len(topology_ids)
        pending = {}

        for i, (tid, state) in enumerate(zip(topology_ids, states)):
            rid = i
            self._workers[tid].pipe.send({
                "command": "evaluate",
                "request_id": rid,
                "state": state,
            })
            pending[rid] = tid

        deadline = time.time() + self.timeout
        while pending:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(
                    f"Worker pool timed out with {len(pending)} pending")

            for rid, tid in list(pending.items()):
                pipe = self._workers[tid].pipe
                if pipe.poll(0.1):
                    msg = pipe.recv()
                    if msg["status"] == "ok":
                        results[msg["request_id"]] = msg
                        del pending[rid]
                    elif msg["status"] == "error":
                        raise RuntimeError(
                            f"Worker {tid} error: {msg.get('error')}\n"
                            f"{msg.get('traceback', '')}")
                    else:
                        raise RuntimeError(f"Unexpected worker response: {msg}")

        return results

    def close(self):
        """Shut down all workers."""
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
