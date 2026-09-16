# ============================================================
#  comm_buffer.py  —  Shared Communication Buffer
#  Cost-aware inter-agent communication
# ============================================================
#
#  Design:
#    - A single CommunicationBuffer is owned by GridWorld.
#    - When a drone chooses COMMUNICATE, it calls:
#        buffer.broadcast(drone_id, discovered_positions)
#      which pushes its known survivor positions into the buffer.
#    - All OTHER drones can read from the buffer via:
#        buffer.get_shared_positions(drone_id)
#      which returns positions shared by OTHER drones (not self).
#    - The buffer is cleared at the start of each episode (reset).
#
#  Partial observability guarantee:
#    - A drone only discovers a survivor when it STEPS on that cell.
#    - Shared positions flow only through explicit COMMUNICATE actions.
#    - Default get_obs() returns zero-padded shared slots if no comm.

from collections import defaultdict
from typing import List, Set, Tuple, Dict


class CommunicationBuffer:
    """
    Episode-scoped shared memory for inter-drone communication.

    Stores survivor positions that each drone has chosen to broadcast.
    Other drones can query this buffer to receive shared knowledge.

    Attributes:
        _broadcasts (dict): drone_id → set of (row, col) positions shared
        comm_counts (dict): drone_id → number of COMMUNICATE actions taken
        comm_battery_spent (dict): drone_id → total battery spent on comms
    """

    def __init__(self, num_drones: int):
        self.num_drones = num_drones
        self._broadcasts: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)
        # Metrics — accumulated per episode, read by trainer/main
        self.comm_counts:         Dict[int, int]   = defaultdict(int)
        self.comm_battery_spent:  Dict[int, int]   = defaultdict(int)
        # Track which positions were NEW when broadcast (for useful bonus)
        self._previously_shared: Set[Tuple[int, int]] = set()

    # ──────────────────────────────────────────────────────────
    #  Broadcast (write)
    # ──────────────────────────────────────────────────────────
    def broadcast(
        self,
        sender_id:   int,
        positions:   Set[Tuple[int, int]],
        battery_cost: int,
    ) -> bool:
        """
        Register a drone's discovered survivor positions in the buffer.

        Args:
            sender_id:    ID of the communicating drone
            positions:    set of (row, col) survivor positions drone knows
            battery_cost: how much battery this comm cost (for metrics)

        Returns:
            True if any NEW information was shared (not previously broadcast),
            False if the broadcast was redundant.
        """
        new_positions = positions - self._previously_shared
        self._broadcasts[sender_id].update(positions)
        self._previously_shared.update(positions)
        self.comm_counts[sender_id] += 1
        self.comm_battery_spent[sender_id] += battery_cost
        return len(new_positions) > 0   # useful if new info added

    # ──────────────────────────────────────────────────────────
    #  Query (read)
    # ──────────────────────────────────────────────────────────
    def get_shared_positions(
        self,
        receiver_id: int,
        max_positions: int,
    ) -> List[Tuple[int, int]]:
        """
        Return survivor positions shared by OTHER drones (not self).

        Args:
            receiver_id:   ID of drone requesting shared info
            max_positions: maximum number of positions to return
                           (keeps observation fixed-size)

        Returns:
            List of (row, col) tuples, length <= max_positions.
            Sorted for determinism.
        """
        shared: Set[Tuple[int, int]] = set()
        for sender_id, positions in self._broadcasts.items():
            if sender_id != receiver_id:
                shared.update(positions)
        # Sort for deterministic obs; truncate to max
        return sorted(shared)[:max_positions]

    # ──────────────────────────────────────────────────────────
    #  Episode metrics
    # ──────────────────────────────────────────────────────────
    def total_comm_actions(self) -> int:
        """Total COMMUNICATE actions across ALL drones this episode."""
        return sum(self.comm_counts.values())

    def total_battery_spent_on_comm(self) -> int:
        """Total battery consumed by communication this episode."""
        return sum(self.comm_battery_spent.values())

    # ──────────────────────────────────────────────────────────
    #  Reset
    # ──────────────────────────────────────────────────────────
    def reset(self):
        """Clear all buffered data for a new episode."""
        self._broadcasts.clear()
        self.comm_counts.clear()
        self.comm_battery_spent.clear()
        self._previously_shared.clear()
