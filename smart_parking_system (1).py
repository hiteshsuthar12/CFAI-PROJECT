"""
================================================================================
SMART PARKING ALLOCATION SYSTEM
================================================================================
A comprehensive AI-based system covering:
  CO1: Agent model (PEAS), environment types, problem formulation,
       knowledge representation, Python essentials for AI
  CO2: BFS, DFS, UCS, A*, Greedy search with profiling
  CO3: CSP modeling, backtracking, constraint propagation (MRV/LCV/Degree),
       min-conflicts local search
  CO4: Minimax with alpha-beta pruning, utility functions, evaluation functions
  CO5: Bayesian Networks, probabilistic inference, uncertainty-aware decisions
  CO6: Hybrid architecture combining all above modules, explainability,
       performance engineering, ethics/bias analysis
================================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS & TYPING (CO1: Python essentials, typing hints)
# ─────────────────────────────────────────────────────────────────────────────
import heapq
import time
import tracemalloc
import random
import math
import logging
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Any

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP (CO1: trace logging & step-by-step reasoning)
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("SmartParking")


# ─────────────────────────────────────────────────────────────────────────────
# CO1 — PEAS AGENT MODEL & ENVIRONMENT TYPES
# ─────────────────────────────────────────────────────────────────────────────

class PEASModel:
    """
    CO1: Agent Model using PEAS framework
    Performance Measure / Environment / Actuators / Sensors
    """
    def __init__(self):
        self.performance_measure = [
            "Minimise vehicle wait time",
            "Maximise parking slot utilisation",
            "Minimise total travel distance",
            "Ensure fair allocation across vehicle types",
        ]
        self.environment = {
            "type": "Partially Observable, Stochastic, Sequential, Dynamic, Discrete, Multi-agent",
            "slots": "Multiple parking floors with typed slots (EV/Disabled/Regular)",
            "vehicles": "Cars, EVs, Bikes, Disabled users with varying priorities",
            "sensors": "Entry cameras, slot IR sensors, EV charger status sensors",
        }
        self.actuators = [
            "LED guidance displays",
            "Barrier gates",
            "Dynamic price boards",
            "Mobile app notifications",
        ]
        self.sensors = [
            "Vehicle type detector (camera + ML)",
            "Slot occupancy sensor (IR/ultrasonic)",
            "EV battery level reader",
            "Disability badge scanner",
        ]

    def describe(self):
        print("\n" + "=" * 60)
        print("   PEAS MODEL — Smart Parking Allocation Agent")
        print("=" * 60)
        print("\n📊 PERFORMANCE MEASURES:")
        for p in self.performance_measure:
            print(f"   • {p}")
        print("\n🌍 ENVIRONMENT:", self.environment["type"])
        print("\n🤖 ACTUATORS:")
        for a in self.actuators:
            print(f"   • {a}")
        print("\n📡 SENSORS:")
        for s in self.sensors:
            print(f"   • {s}")
        print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CO1 — STATE / DATACLASSES / KNOWLEDGE REPRESENTATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ParkingSlot:
    """Immutable representation of a parking slot (CO1: dataclasses)"""
    slot_id: str
    floor: int
    slot_type: str          # "EV" | "Disabled" | "Regular"
    distance_from_entry: int  # in metres

@dataclass
class Vehicle:
    """Vehicle state (CO1: classes for state)"""
    vehicle_id: str
    vehicle_type: str       # "EV" | "Disabled" | "Regular"
    entry_time: float = field(default_factory=time.time)
    assigned_slot: Optional[str] = None
    battery_pct: int = 100  # relevant for EVs

@dataclass
class ParkingState:
    """
    CO1: Problem formulation — state representation
    State = snapshot of all slot occupancies
    """
    occupied: Dict[str, Optional[str]]   # slot_id -> vehicle_id or None
    waiting_queue: List[str]             # vehicle_ids waiting
    step: int = 0

    def is_goal(self, vehicle_id: str) -> bool:
        """Goal: vehicle is assigned a slot"""
        return any(v == vehicle_id for v in self.occupied.values())

    def __hash__(self):
        return hash((frozenset(self.occupied.items()), tuple(self.waiting_queue)))

    def __eq__(self, other):
        return self.occupied == other.occupied and self.waiting_queue == other.waiting_queue


# ─────────────────────────────────────────────────────────────────────────────
# CO1 — PARKING LOT GRAPH (Knowledge Representation: Graphs)
# ─────────────────────────────────────────────────────────────────────────────

class ParkingLotGraph:
    """
    CO1: Knowledge representation using graph (adjacency list)
    Nodes = entry/exit + parking slots; Edges = paths with distances
    """
    def __init__(self):
        self.graph: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self.slots: Dict[str, ParkingSlot] = {}
        self._build_lot()

    def _build_lot(self):
        """Build a 3-floor parking lot graph"""
        # Floor 1: slots P1_01 .. P1_06
        # Floor 2: slots P2_01 .. P2_06
        # Floor 3: slots P3_01 .. P3_06
        slot_types = {
            "P1_01": "Disabled", "P1_02": "EV",    "P1_03": "Regular",
            "P1_04": "Regular",  "P1_05": "Regular","P1_06": "Regular",
            "P2_01": "EV",       "P2_02": "EV",    "P2_03": "Regular",
            "P2_04": "Regular",  "P2_05": "Disabled","P2_06": "Regular",
            "P3_01": "Regular",  "P3_02": "Regular","P3_03": "Regular",
            "P3_04": "Regular",  "P3_05": "EV",    "P3_06": "Regular",
        }
        distances = {
            "P1_01": 10, "P1_02": 15, "P1_03": 20, "P1_04": 25, "P1_05": 30, "P1_06": 35,
            "P2_01": 40, "P2_02": 45, "P2_03": 50, "P2_04": 55, "P2_05": 60, "P2_06": 65,
            "P3_01": 70, "P3_02": 75, "P3_03": 80, "P3_04": 85, "P3_05": 90, "P3_06": 95,
        }
        for sid, stype in slot_types.items():
            floor = int(sid[1])
            self.slots[sid] = ParkingSlot(sid, floor, stype, distances[sid])

        # Entry -> Floor 1 connections
        self.add_edge("ENTRY", "P1_01", 10)
        self.add_edge("ENTRY", "P1_02", 15)
        self.add_edge("ENTRY", "P1_03", 20)
        self.add_edge("ENTRY", "P1_04", 25)
        self.add_edge("ENTRY", "P1_05", 30)
        self.add_edge("ENTRY", "P1_06", 35)
        # Floor 1 -> Floor 2 (ramp)
        self.add_edge("P1_06", "P2_01", 10)
        for i in range(1, 6):
            self.add_edge(f"P2_0{i}", f"P2_0{i+1}", 5)
        # Floor 2 -> Floor 3 (ramp)
        self.add_edge("P2_06", "P3_01", 10)
        for i in range(1, 6):
            self.add_edge(f"P3_0{i}", f"P3_0{i+1}", 5)

    def add_edge(self, u: str, v: str, cost: int):
        self.graph[u].append((v, cost))
        self.graph[v].append((u, cost))

    def neighbors(self, node: str) -> List[Tuple[str, int]]:
        return self.graph[node]

    def available_slots(self, occupied: Dict[str, Optional[str]]) -> List[ParkingSlot]:
        return [s for sid, s in self.slots.items() if occupied.get(sid) is None]


# ─────────────────────────────────────────────────────────────────────────────
# CO2 — SEARCH ALGORITHMS with Profiling
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SearchProfile:
    """CO2: Empirical profiling — node expansions, runtime, peak memory"""
    algorithm: str
    nodes_expanded: int = 0
    runtime_ms: float = 0.0
    peak_memory_kb: float = 0.0
    path_cost: int = 0
    path: List[str] = field(default_factory=list)

    def report(self):
        print(f"\n📈 [{self.algorithm}] Search Profile")
        print(f"   Nodes expanded  : {self.nodes_expanded}")
        print(f"   Runtime         : {self.runtime_ms:.2f} ms")
        print(f"   Peak memory     : {self.peak_memory_kb:.2f} KB")
        print(f"   Path cost       : {self.path_cost}")
        print(f"   Path            : {' → '.join(self.path)}")


class SearchAlgorithms:
    """
    CO2: BFS / DFS / UCS / A* / Greedy on the parking lot graph
    """

    def __init__(self, graph: ParkingLotGraph):
        self.graph = graph

    # ── BFS ──────────────────────────────────────────────────────────────────
    def bfs(self, start: str, goal_slots: Set[str]) -> SearchProfile:
        """CO2: BFS — finds shortest path in terms of hops"""
        profile = SearchProfile("BFS")
        tracemalloc.start()
        t0 = time.perf_counter()

        queue = deque([(start, [start], 0)])
        visited: Set[str] = {start}

        while queue:
            node, path, cost = queue.popleft()
            profile.nodes_expanded += 1

            if node in goal_slots:
                profile.path = path
                profile.path_cost = cost
                break

            for neighbor, edge_cost in self.graph.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor], cost + edge_cost))

        profile.runtime_ms = (time.perf_counter() - t0) * 1000
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        profile.peak_memory_kb = peak / 1024
        return profile

    # ── DFS ──────────────────────────────────────────────────────────────────
    def dfs(self, start: str, goal_slots: Set[str]) -> SearchProfile:
        """CO2: DFS — depth-first, not optimal but fast to implement"""
        profile = SearchProfile("DFS")
        tracemalloc.start()
        t0 = time.perf_counter()

        stack = [(start, [start], 0)]
        visited: Set[str] = set()

        while stack:
            node, path, cost = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            profile.nodes_expanded += 1

            if node in goal_slots:
                profile.path = path
                profile.path_cost = cost
                break

            for neighbor, edge_cost in self.graph.neighbors(node):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor], cost + edge_cost))

        profile.runtime_ms = (time.perf_counter() - t0) * 1000
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        profile.peak_memory_kb = peak / 1024
        return profile

    # ── UCS ──────────────────────────────────────────────────────────────────
    def ucs(self, start: str, goal_slots: Set[str]) -> SearchProfile:
        """CO2: Uniform Cost Search — optimal path by cumulative cost"""
        profile = SearchProfile("UCS")
        tracemalloc.start()
        t0 = time.perf_counter()

        # Priority queue: (cost, node, path)
        pq = [(0, start, [start])]
        visited: Set[str] = set()

        while pq:
            cost, node, path = heapq.heappop(pq)
            if node in visited:
                continue
            visited.add(node)
            profile.nodes_expanded += 1

            if node in goal_slots:
                profile.path = path
                profile.path_cost = cost
                break

            for neighbor, edge_cost in self.graph.neighbors(node):
                if neighbor not in visited:
                    heapq.heappush(pq, (cost + edge_cost, neighbor, path + [neighbor]))

        profile.runtime_ms = (time.perf_counter() - t0) * 1000
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        profile.peak_memory_kb = peak / 1024
        return profile

    # ── Heuristic for A* / Greedy ─────────────────────────────────────────
    def heuristic(self, node: str, goal_slots: Set[str]) -> int:
        """
        CO2: Heuristic — admissible & consistent.
        Estimates minimum distance from node to nearest goal slot.
        Never overestimates (admissible).
        """
        if node in self.graph.slots:
            slot = self.graph.slots[node]
            return min(
                abs(slot.distance_from_entry - self.graph.slots[g].distance_from_entry)
                for g in goal_slots if g in self.graph.slots
            ) if goal_slots & self.graph.slots.keys() else 0
        return 0

    # ── A* ───────────────────────────────────────────────────────────────────
    def astar(self, start: str, goal_slots: Set[str]) -> SearchProfile:
        """
        CO2: A* Search — optimal with admissible heuristic
        f(n) = g(n) + h(n)
        Tie-breaking: prefer smaller h(n) when f values equal
        """
        profile = SearchProfile("A*")
        tracemalloc.start()
        t0 = time.perf_counter()

        # (f, h, g, node, path) — h used for tie-breaking (CO2)
        h0 = self.heuristic(start, goal_slots)
        pq = [(h0, h0, 0, start, [start])]
        open_set: Dict[str, int] = {start: h0}  # CO2: open set tracking
        closed_set: Set[str] = set()            # CO2: closed set

        while pq:
            f, h, g, node, path = heapq.heappop(pq)
            if node in closed_set:
                continue
            closed_set.add(node)
            profile.nodes_expanded += 1

            if node in goal_slots:
                profile.path = path
                profile.path_cost = g
                break

            for neighbor, edge_cost in self.graph.neighbors(node):
                if neighbor in closed_set:
                    continue
                new_g = g + edge_cost
                h_n = self.heuristic(neighbor, goal_slots)
                new_f = new_g + h_n
                if neighbor not in open_set or new_f < open_set[neighbor]:
                    open_set[neighbor] = new_f
                    # Tie-break by h (CO2: tie-breaking strategies)
                    heapq.heappush(pq, (new_f, h_n, new_g, neighbor, path + [neighbor]))

        profile.runtime_ms = (time.perf_counter() - t0) * 1000
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        profile.peak_memory_kb = peak / 1024
        return profile

    # ── Greedy Best-First ────────────────────────────────────────────────────
    def greedy(self, start: str, goal_slots: Set[str]) -> SearchProfile:
        """CO2: Greedy Best-First — uses only h(n), not optimal"""
        profile = SearchProfile("Greedy")
        tracemalloc.start()
        t0 = time.perf_counter()

        pq = [(self.heuristic(start, goal_slots), start, [start], 0)]
        visited: Set[str] = set()

        while pq:
            h, node, path, cost = heapq.heappop(pq)
            if node in visited:
                continue
            visited.add(node)
            profile.nodes_expanded += 1

            if node in goal_slots:
                profile.path = path
                profile.path_cost = cost
                break

            for neighbor, edge_cost in self.graph.neighbors(node):
                if neighbor not in visited:
                    heapq.heappush(pq, (
                        self.heuristic(neighbor, goal_slots),
                        neighbor, path + [neighbor], cost + edge_cost
                    ))

        profile.runtime_ms = (time.perf_counter() - t0) * 1000
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        profile.peak_memory_kb = peak / 1024
        return profile

    def compare_all(self, start: str, goal_slots: Set[str]):
        """CO2: Empirical comparison of all search algorithms"""
        print("\n" + "=" * 60)
        print("   CO2: SEARCH ALGORITHM COMPARISON")
        print("=" * 60)
        for algo_fn in [self.bfs, self.dfs, self.ucs, self.astar, self.greedy]:
            profile = algo_fn(start, goal_slots)
            profile.report()
        print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CO3 — CSP MODELING & BACKTRACKING
# ─────────────────────────────────────────────────────────────────────────────

class ParkingCSP:
    """
    CO3: Constraint Satisfaction Problem for parking slot allocation.

    Variables  : Vehicles to assign
    Domain     : Available parking slots per vehicle
    Constraints:
      1. One slot per vehicle
      2. EV vehicle must get EV or Regular slot (prefers EV)
      3. Disabled vehicle must get Disabled slot
      4. No two vehicles share a slot
    """

    def __init__(self, vehicles: List[Vehicle], slots: Dict[str, ParkingSlot]):
        self.vehicles = vehicles
        self.all_slots = slots
        self.assignment: Dict[str, str] = {}   # vehicle_id -> slot_id
        self.failure_reason: Optional[str] = None

    def domain(self, vehicle: Vehicle, occupied: Set[str]) -> List[str]:
        """CO3: Domain — valid slots for a vehicle"""
        result = []
        for sid, slot in self.all_slots.items():
            if sid in occupied:
                continue
            if vehicle.vehicle_type == "Disabled" and slot.slot_type != "Disabled":
                continue
            if vehicle.vehicle_type == "EV" and slot.slot_type not in ("EV", "Regular"):
                continue
            result.append(sid)
        return result

    # ── MRV Heuristic ────────────────────────────────────────────────────────
    def select_unassigned_variable_mrv(self, assigned: Dict[str, str]) -> Optional[Vehicle]:
        """CO3: MRV — pick vehicle with fewest valid slots (Minimum Remaining Values)"""
        unassigned = [v for v in self.vehicles if v.vehicle_id not in assigned]
        if not unassigned:
            return None
        occupied = set(assigned.values())
        return min(unassigned, key=lambda v: len(self.domain(v, occupied)))

    # ── Degree Heuristic ─────────────────────────────────────────────────────
    def degree_heuristic(self, vehicle: Vehicle, assigned: Dict[str, str]) -> int:
        """CO3: Degree — count constraints this vehicle has with unassigned others"""
        occupied = set(assigned.values())
        my_domain = set(self.domain(vehicle, occupied))
        count = 0
        for other in self.vehicles:
            if other.vehicle_id == vehicle.vehicle_id or other.vehicle_id in assigned:
                continue
            other_domain = set(self.domain(other, occupied))
            if my_domain & other_domain:  # shared slots
                count += 1
        return count

    # ── LCV Ordering ─────────────────────────────────────────────────────────
    def order_domain_values_lcv(self, vehicle: Vehicle, assigned: Dict[str, str]) -> List[str]:
        """CO3: LCV — order slots that rule out fewest options for other vehicles"""
        occupied = set(assigned.values())
        domain = self.domain(vehicle, occupied)

        def lcv_score(slot_id: str) -> int:
            count = 0
            for other in self.vehicles:
                if other.vehicle_id in assigned:
                    continue
                occ2 = occupied | {slot_id}
                if slot_id in self.domain(other, occ2):
                    count += 1
            return count

        return sorted(domain, key=lcv_score)

    # ── Forward Checking ─────────────────────────────────────────────────────
    def forward_check(self, vehicle: Vehicle, slot_id: str, assigned: Dict[str, str]) -> bool:
        """CO3: Forward checking — ensure no other vehicle's domain goes empty"""
        occupied = set(assigned.values()) | {slot_id}
        for other in self.vehicles:
            if other.vehicle_id in assigned or other.vehicle_id == vehicle.vehicle_id:
                continue
            if not self.domain(other, occupied):
                logger.debug(f"Forward check failed: assigning {slot_id} to {vehicle.vehicle_id} "
                             f"empties domain of {other.vehicle_id}")
                return False
        return True

    # ── Backtracking Search ───────────────────────────────────────────────────
    def backtrack(self, assigned: Dict[str, str], depth: int = 0) -> Optional[Dict[str, str]]:
        """CO3: Backtracking search with MRV + LCV + forward checking"""
        if len(assigned) == len(self.vehicles):
            return assigned

        vehicle = self.select_unassigned_variable_mrv(assigned)
        if vehicle is None:
            return assigned

        for slot_id in self.order_domain_values_lcv(vehicle, assigned):
            logger.debug(f"{'  ' * depth}Trying {vehicle.vehicle_id} → {slot_id}")
            if self.forward_check(vehicle, slot_id, assigned):
                assigned[vehicle.vehicle_id] = slot_id
                result = self.backtrack(assigned, depth + 1)
                if result is not None:
                    return result
                del assigned[vehicle.vehicle_id]
                logger.debug(f"{'  ' * depth}Backtracking from {slot_id}")

        # CO3: Explainability — why constraint failed
        occupied = set(assigned.values())
        self.failure_reason = (
            f"No valid slot for {vehicle.vehicle_id} (type={vehicle.vehicle_type}). "
            f"Domain was empty. Occupied slots: {occupied}"
        )
        return None

    def solve(self) -> Optional[Dict[str, str]]:
        """Run CSP backtracking solver"""
        print("\n" + "=" * 60)
        print("   CO3: CSP — Backtracking with MRV + LCV + Forward Checking")
        print("=" * 60)
        solution = self.backtrack({})
        if solution:
            print("✅ CSP Solution Found:")
            for vid, sid in solution.items():
                print(f"   {vid} → {sid} ({self.all_slots[sid].slot_type})")
        else:
            print(f"❌ CSP Failed: {self.failure_reason}")
        return solution

    # ── Min-Conflicts Local Search ────────────────────────────────────────────
    def min_conflicts(self, max_steps: int = 1000) -> Optional[Dict[str, str]]:
        """CO3: Local search for CSP using min-conflicts heuristic"""
        print("\n--- CO3: Min-Conflicts Local Search ---")
        # Random initial assignment
        assignment: Dict[str, str] = {}
        slot_ids = list(self.all_slots.keys())
        random.shuffle(slot_ids)
        for i, v in enumerate(self.vehicles):
            assignment[v.vehicle_id] = slot_ids[i % len(slot_ids)]

        for step in range(max_steps):
            # Find conflicted vehicles
            conflicted = []
            for v in self.vehicles:
                sid = assignment[v.vehicle_id]
                occupied = {assignment[other.vehicle_id] for other in self.vehicles
                            if other.vehicle_id != v.vehicle_id}
                if sid in occupied or not self._type_compatible(v, self.all_slots[sid]):
                    conflicted.append(v)

            if not conflicted:
                print(f"✅ Min-Conflicts solved in {step} steps")
                return assignment

            vehicle = random.choice(conflicted)
            occupied = {assignment[other.vehicle_id] for other in self.vehicles
                        if other.vehicle_id != vehicle.vehicle_id}

            def conflict_count(sid: str) -> int:
                c = 1 if sid in occupied else 0
                c += 0 if self._type_compatible(vehicle, self.all_slots[sid]) else 1
                return c

            best_slot = min(slot_ids, key=conflict_count)
            assignment[vehicle.vehicle_id] = best_slot

        print("⚠️  Min-Conflicts: max steps reached without full solution")
        return assignment

    def _type_compatible(self, vehicle: Vehicle, slot: ParkingSlot) -> bool:
        if vehicle.vehicle_type == "Disabled":
            return slot.slot_type == "Disabled"
        if vehicle.vehicle_type == "EV":
            return slot.slot_type in ("EV", "Regular")
        return True


# ─────────────────────────────────────────────────────────────────────────────
# CO4 — MINIMAX WITH ALPHA-BETA PRUNING (Multi-agent reasoning)
# ─────────────────────────────────────────────────────────────────────────────

class ParkingGameTree:
    """
    CO4: Minimax for multi-agent scenario — System (MAX) vs. Adversarial
    Demand Spike (MIN, simulates worst-case arrival).
    
    State: (available_slots, pending_vehicles)
    MAX move: assign best slot
    MIN move: remove the best slot (adversarial demand)
    Utility: total distance saved (lower distance = higher utility)
    """

    def __init__(self, slots: List[ParkingSlot]):
        self.slots = slots
        self.nodes_evaluated = 0
        self.pruned = 0

    def utility(self, available: List[ParkingSlot], vehicles_left: int) -> float:
        """CO4: Utility function — reward for keeping near slots available"""
        if not available:
            return -1000.0 * vehicles_left
        avg_dist = sum(s.distance_from_entry for s in available) / len(available)
        return -avg_dist + (len(available) * 10)

    def is_terminal(self, available: List[ParkingSlot], depth: int) -> bool:
        """CO4: Terminal test — no slots left or depth limit reached"""
        return depth == 0 or not available

    def minimax(
        self,
        available: List[ParkingSlot],
        vehicles_left: int,
        depth: int,
        is_max: bool,
        alpha: float,
        beta: float,
    ) -> Tuple[float, Optional[ParkingSlot]]:
        """
        CO4: Minimax with alpha-beta pruning + depth limit
        alpha = best MAX can guarantee
        beta  = best MIN can guarantee
        """
        self.nodes_evaluated += 1

        if self.is_terminal(available, depth):
            return self.utility(available, vehicles_left), None

        best_slot = None

        if is_max:
            max_val = float("-inf")
            # MAX assigns nearest slot (sorted by distance)
            for slot in sorted(available, key=lambda s: s.distance_from_entry):
                remaining = [s for s in available if s != slot]
                val, _ = self.minimax(remaining, vehicles_left - 1, depth - 1, False, alpha, beta)
                if val > max_val:
                    max_val = val
                    best_slot = slot
                alpha = max(alpha, val)
                if beta <= alpha:
                    self.pruned += 1
                    break   # CO4: Alpha-beta pruning — beta cutoff
            return max_val, best_slot
        else:
            min_val = float("inf")
            # MIN removes most desirable slot (adversarial)
            for slot in sorted(available, key=lambda s: s.distance_from_entry):
                remaining = [s for s in available if s != slot]
                val, _ = self.minimax(remaining, vehicles_left, depth - 1, True, alpha, beta)
                if val < min_val:
                    min_val = val
                    best_slot = slot
                beta = min(beta, val)
                if beta <= alpha:
                    self.pruned += 1
                    break   # CO4: Alpha-beta pruning — alpha cutoff
            return min_val, best_slot

    def recommend_slot(self, vehicles_left: int = 3, depth: int = 4) -> Optional[ParkingSlot]:
        """CO4: Use minimax to recommend best slot under adversarial conditions"""
        print("\n" + "=" * 60)
        print("   CO4: MINIMAX + ALPHA-BETA — Adversarial Slot Selection")
        print("=" * 60)
        val, slot = self.minimax(
            self.slots, vehicles_left, depth,
            is_max=True, alpha=float("-inf"), beta=float("inf")
        )
        print(f"   Minimax value    : {val:.2f}")
        print(f"   Nodes evaluated  : {self.nodes_evaluated}")
        print(f"   Branches pruned  : {self.pruned}")
        if slot:
            print(f"   Recommended slot : {slot.slot_id} (dist={slot.distance_from_entry}m, type={slot.slot_type})")
        print("=" * 60)
        return slot


# ─────────────────────────────────────────────────────────────────────────────
# CO5 — BAYESIAN NETWORK & PROBABILISTIC INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

class BayesianParkingInference:
    """
    CO5: Bayesian Network for parking demand prediction.

    Network structure:
        TimeOfDay → DemandHigh
        IsWeekend → DemandHigh
        DemandHigh → ParkingAvailable
        EVChargerFree → EVSlotAvailable

    Inference: P(ParkingAvailable | TimeOfDay, IsWeekend)
    using Variable Elimination (manual CPTs)
    """

    def __init__(self):
        # CO5: CPTs — Conditional Probability Tables
        # P(DemandHigh | TimeOfDay, IsWeekend)
        self.cpt_demand: Dict[Tuple, float] = {
            ("Peak", True):    0.90,
            ("Peak", False):   0.75,
            ("OffPeak", True): 0.50,
            ("OffPeak", False):0.25,
        }
        # P(ParkingAvailable | DemandHigh)
        self.cpt_parking: Dict[bool, float] = {
            True:  0.20,   # High demand → low availability
            False: 0.85,   # Low demand → high availability
        }
        # P(EVSlotAvailable | EVChargerFree)
        self.cpt_ev: Dict[bool, float] = {
            True:  0.90,
            False: 0.15,
        }
        # Priors
        self.p_weekend = 0.286      # 2/7 days
        self.p_peak_given_weekend = 0.60
        self.p_peak_given_weekday = 0.40
        self.p_charger_free = 0.55

    def p_demand_high(self, time_of_day: str, is_weekend: bool) -> float:
        """CO5: Direct CPT lookup"""
        return self.cpt_demand.get((time_of_day, is_weekend), 0.5)

    def infer_parking_available(self, time_of_day: str, is_weekend: bool) -> Dict[str, float]:
        """
        CO5: Variable Elimination — P(ParkingAvailable | time, weekend)
        Marginalize over DemandHigh.
        """
        p_available = 0.0
        p_not_available = 0.0

        for demand_high in [True, False]:
            p_d = (self.p_demand_high(time_of_day, is_weekend) if demand_high
                   else 1 - self.p_demand_high(time_of_day, is_weekend))
            p_park_given_d = self.cpt_parking[demand_high]
            p_available += p_d * p_park_given_d
            p_not_available += p_d * (1 - p_park_given_d)

        total = p_available + p_not_available
        return {
            "P(Available)":   round(p_available / total, 4),
            "P(Unavailable)": round(p_not_available / total, 4),
        }

    def infer_ev_slot(self, charger_free: bool) -> Dict[str, float]:
        """CO5: Simple Bayesian update for EV slot availability"""
        p_ev = self.cpt_ev[charger_free]
        return {
            "P(EVSlotAvailable)":    round(p_ev, 4),
            "P(EVSlotUnavailable)": round(1 - p_ev, 4),
        }

    def expected_utility_of_waiting(
        self, p_available: float, wait_cost: float = 5.0, park_benefit: float = 20.0
    ) -> float:
        """CO5: Expected utility = Σ P(outcome) * utility(outcome)"""
        eu = p_available * park_benefit + (1 - p_available) * (-wait_cost)
        return round(eu, 4)

    def report(self, time_of_day: str = "Peak", is_weekend: bool = False, charger_free: bool = True):
        print("\n" + "=" * 60)
        print("   CO5: BAYESIAN NETWORK — Probabilistic Inference")
        print("=" * 60)
        print(f"   Conditions  : TimeOfDay={time_of_day}, Weekend={is_weekend}, ChargerFree={charger_free}")

        park_probs = self.infer_parking_available(time_of_day, is_weekend)
        ev_probs = self.infer_ev_slot(charger_free)
        eu = self.expected_utility_of_waiting(park_probs["P(Available)"])

        print(f"\n   P(DemandHigh)        : {self.p_demand_high(time_of_day, is_weekend):.2f}")
        print(f"   P(ParkingAvailable)  : {park_probs['P(Available)']:.4f}")
        print(f"   P(ParkingUnavailable): {park_probs['P(Unavailable)']:.4f}")
        print(f"   P(EVSlotAvailable)   : {ev_probs['P(EVSlotAvailable)']:.4f}")
        print(f"\n   Expected Utility of Arriving Now : {eu:.4f}")
        recommendation = "✅ Recommend arriving now" if eu > 0 else "⚠️  Consider waiting or alternate lot"
        print(f"   Decision             : {recommendation}")
        print("=" * 60)
        return park_probs, ev_probs, eu


# ─────────────────────────────────────────────────────────────────────────────
# CO6 — HYBRID ARCHITECTURE: Combining all modules
# ─────────────────────────────────────────────────────────────────────────────

class SmartParkingSystem:
    """
    CO6: Hybrid Architecture — Rule-based + Search + CSP + Probabilistic + Game-Theory
    Provides explainable reasoning traces and performance analysis.
    """

    def __init__(self):
        # CO1: Build the environment
        self.peas = PEASModel()
        self.lot = ParkingLotGraph()
        self.search = SearchAlgorithms(self.lot)
        self.bayes = BayesianParkingInference()

        # Initial occupancy (some pre-occupied)
        self.occupied: Dict[str, Optional[str]] = {sid: None for sid in self.lot.slots}
        self.occupied["P1_01"] = "V_EXISTING_1"
        self.occupied["P2_03"] = "V_EXISTING_2"

        # Ethics/bias log (CO6)
        self.ethics_log: List[str] = []

    # ── CO6: Rule-based priority filter ──────────────────────────────────────
    def priority_filter(self, vehicle: Vehicle, slots: List[ParkingSlot]) -> List[ParkingSlot]:
        """CO6: Rule + constraint — filter slots by vehicle type priority"""
        if vehicle.vehicle_type == "Disabled":
            preferred = [s for s in slots if s.slot_type == "Disabled"]
            if not preferred:
                self.ethics_log.append(
                    f"⚠️  ETHICS: No Disabled slot for {vehicle.vehicle_id} — bias in allocation!"
                )
                return slots
            return preferred
        if vehicle.vehicle_type == "EV":
            preferred = [s for s in slots if s.slot_type == "EV"]
            return preferred if preferred else [s for s in slots if s.slot_type == "Regular"]
        return [s for s in slots if s.slot_type == "Regular"]

    # ── CO6: Full allocation pipeline ────────────────────────────────────────
    def allocate(self, vehicle: Vehicle, time_of_day: str = "Peak", is_weekend: bool = False) -> Optional[str]:
        """
        CO6: Hybrid reasoning pipeline:
          1. Bayesian check — should vehicle attempt now?
          2. Rule filter — eligible slots
          3. A* search — find shortest path to eligible slot
          4. Return assigned slot with explainability trace
        """
        print(f"\n{'─'*60}")
        print(f"🚗 ALLOCATING: {vehicle.vehicle_id} (Type: {vehicle.vehicle_type})")
        print(f"{'─'*60}")

        # ── Step 1: Probabilistic Check (CO5) ────────────────────────────────
        charger_free = vehicle.vehicle_type == "EV" and random.random() > 0.4
        park_probs, _, eu = self.bayes.report(time_of_day, is_weekend, charger_free)
        if eu < -2:
            print(f"⛔ Probabilistic decision: Low utility ({eu:.2f}). Suggest alternate lot.")
            return None

        # ── Step 2: Rule-based filter (CO6) ──────────────────────────────────
        available = self.lot.available_slots(self.occupied)
        eligible = self.priority_filter(vehicle, available)
        if not eligible:
            print("❌ No eligible slots available.")
            return None

        goal_set = {s.slot_id for s in eligible}
        print(f"\n🎯 Eligible slots: {goal_set}")

        # ── Step 3: A* search (CO2) ───────────────────────────────────────────
        profile = self.search.astar("ENTRY", goal_set)
        profile.report()

        if not profile.path:
            print("❌ No path found to any eligible slot.")
            return None

        assigned_slot_id = profile.path[-1]
        self.occupied[assigned_slot_id] = vehicle.vehicle_id
        vehicle.assigned_slot = assigned_slot_id

        # ── Explainability trace (CO6) ────────────────────────────────────────
        print(f"\n✅ ALLOCATION COMPLETE")
        print(f"   Vehicle  : {vehicle.vehicle_id} ({vehicle.vehicle_type})")
        print(f"   Slot     : {assigned_slot_id} ({self.lot.slots[assigned_slot_id].slot_type})")
        print(f"   Path     : {' → '.join(profile.path)}")
        print(f"   Distance : {profile.path_cost}m")
        print(f"   Reason   : A* search selected nearest eligible slot with admissible heuristic.")

        return assigned_slot_id

    # ── CO6: Full system demonstration ───────────────────────────────────────
    def run_full_demo(self):
        # CO1: PEAS
        self.peas.describe()

        # CO2: Search comparison on a sample query
        ev_slots = {sid for sid, s in self.lot.slots.items() if s.slot_type == "EV"}
        self.search.compare_all("ENTRY", ev_slots)

        # CO3: CSP
        test_vehicles = [
            Vehicle("V001", "EV"),
            Vehicle("V002", "Disabled"),
            Vehicle("V003", "Regular"),
            Vehicle("V004", "Regular"),
        ]
        csp = ParkingCSP(test_vehicles, self.lot.slots)
        csp.solve()
        csp.min_conflicts(max_steps=500)

        # CO4: Minimax
        available_slots = list(self.lot.slots.values())[:8]
        game = ParkingGameTree(available_slots)
        game.recommend_slot(vehicles_left=3, depth=4)

        # CO5 + CO6: Bayesian + Hybrid allocation
        vehicles = [
            Vehicle("V_EV_01", "EV", battery_pct=30),
            Vehicle("V_DIS_01", "Disabled"),
            Vehicle("V_REG_01", "Regular"),
        ]
        for v in vehicles:
            self.allocate(v, time_of_day="Peak", is_weekend=False)

        # CO6: Ethics / bias report
        print("\n" + "=" * 60)
        print("   CO6: ETHICS & BIAS ANALYSIS")
        print("=" * 60)
        if self.ethics_log:
            for entry in self.ethics_log:
                print(f"   {entry}")
        else:
            print("   ✅ No allocation bias detected in this run.")
        print("\n   Known limitations:")
        print("   • Heuristic may under-estimate for multi-floor layouts → suboptimal paths")
        print("   • Bayesian CPTs trained on limited data → uncertainty miscalibration risk")
        print("   • Min-conflicts may get stuck in local optima for large lots")
        print("=" * 60)

        print("\n\n🏁 Smart Parking System demo complete.")
        print("   This system integrates CO1–CO6 concepts:")
        print("   CO1: PEAS, state/dataclasses, graphs, Python essentials")
        print("   CO2: BFS/DFS/UCS/A*/Greedy with empirical profiling")
        print("   CO3: CSP backtracking + MRV/LCV/FC + min-conflicts")
        print("   CO4: Minimax + alpha-beta pruning + utility functions")
        print("   CO5: Bayesian Network + variable elimination + expected utility")
        print("   CO6: Hybrid architecture + explainability + ethics analysis")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    system = SmartParkingSystem()
    system.run_full_demo()
