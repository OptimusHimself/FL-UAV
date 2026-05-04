import copy
import time
import random
from queue import Queue
from typing import Dict, List, Any


class SimulatedChannel:
    """Simulates server-client communication using Python queues."""

    def __init__(self, num_clients: int = 5,
                 latency_ms: float = 10.0,
                 packet_loss_rate: float = 0.0):
        self.num_clients = num_clients
        self.latency_ms = latency_ms
        self.packet_loss_rate = packet_loss_rate
        self.download_queues: Dict[int, Queue] = {
            i: Queue() for i in range(num_clients)
        }
        self.upload_queues: Dict[int, Queue] = {
            i: Queue() for i in range(num_clients)
        }

    def server_broadcast(self, data: Any, client_ids: List[int]):
        for cid in client_ids:
            if random.random() > self.packet_loss_rate:
                self.download_queues[cid].put(copy.deepcopy(data))

    def server_send_to_client(self, client_id: int, data: Any):
        if random.random() > self.packet_loss_rate:
            self.download_queues[client_id].put(copy.deepcopy(data))

    def client_upload(self, client_id: int, data: Any):
        if random.random() > self.packet_loss_rate:
            self.upload_queues[client_id].put(copy.deepcopy(data))

    def client_receive(self, client_id: int) -> Any:
        if not self.download_queues[client_id].empty():
            return self.download_queues[client_id].get()
        return None

    def server_receive(self, client_id: int) -> Any:
        if not self.upload_queues[client_id].empty():
            return self.upload_queues[client_id].get()
        return None

    def server_receive_all(self, client_ids: List[int]) -> Dict[int, Any]:
        results = {}
        for cid in client_ids:
            data = self.server_receive(cid)
            if data is not None:
                results[cid] = data
        return results

    def clear(self):
        for q in list(self.download_queues.values()) + list(self.upload_queues.values()):
            while not q.empty():
                q.get()
