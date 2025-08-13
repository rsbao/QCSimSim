import math
import random
from collections import defaultdict

class HardwareConfig:
    def __init__(self, 
                num_nodes, 
                mem_bandwidth_GBps,
                mem_latency_ns,
                mem_per_device_GB,
                flops_per_device=1,
                num_devices_per_node=1, # should be square-like/easily divisible!
                inter_node_bandwidth_GBps=0, # per link
                intra_node_bandwidth_GBps=0.0, # per link
                inter_node_latency_ns=0.0, # per link
                intra_node_latency_ns=0.0, # per link
                intra_node_topology="mesh", # mesh, a2a for now
                inter_node_topology="mesh"
                ):
        self.num_nodes = num_nodes
        self.num_devices_per_node = num_devices_per_node
        self.intra_node_bandwidth_GBps = intra_node_bandwidth_GBps
        self.intra_node_latency_ns = intra_node_latency_ns
        self.flops_per_device = flops_per_device
        self.mem_bandwidth_GBps = mem_bandwidth_GBps
        self.inter_node_bandwidth_GBps = inter_node_bandwidth_GBps
        self.mem_latency_ns = mem_latency_ns
        self.inter_node_latency_ns = inter_node_latency_ns
        self.mem_per_device_GB = mem_per_device_GB
        self.intra_node_topology = intra_node_topology
        self.inter_node_topology = inter_node_topology

class QuantumCircuit:
    def __init__(self, num_qubits, gates, num_gates, sim_type="Schrödinger"):
        self.num_qubits = num_qubits
        if(gates == "random"):
            self.gates = []
            for _ in range(num_gates):  # Randomly generate 100 gates
                gate_type = "H" if random.random() < 0.5 else "CNOT"
                target = [random.randint(0,num_qubits-1), random.randint(0,num_qubits-1)] \
                    if gate_type == "CNOT" else [random.randint(0,num_qubits-1)]
                self.gates.append((gate_type, target))
        else:
            self.gates = gates  # List of (gate_type, target(s))
        self.sim_type = sim_type  # Simulation type, e.g., "Schrödinger"

class PerformanceSimulator:
    def __init__(self, circuit: QuantumCircuit, hardware: HardwareConfig):
        self.circuit = circuit
        self.hardware = hardware
        self.state_vector_size = 2 ** circuit.num_qubits  # Number of amplitudes
        self.local_qubits = int(math.log2(hardware.mem_per_device_GB * 1e9 // 16))  # Number of qubits that can be stored in local memory (assuming complex128)   
        self.node_qubits = int(math.log2(hardware.num_devices_per_node)) + self.local_qubits # Total qubits across all nodes
        if circuit.num_qubits > self.node_qubits + int(math.log2(hardware.num_nodes)):
            raise ValueError("Circuit too large for available memory on hardware.")
    
    # ---------------------------
    # Contention helpers (mesh)
    # ---------------------------
    @staticmethod
    def _mesh_dims(P: int):
        """Choose a near-square grid (rows, cols) for P tiles."""
        r = int(math.floor(math.sqrt(P)))
        c = (P + r - 1) // r
        while r * c < P:
            r += 1
        return r, c  # rows, cols

    @staticmethod
    def _id_to_xy(i: int, cols: int):
        return (i // cols, i % cols)  # (row, col)

    @staticmethod
    def _manhattan_path(u_xy, v_xy):
        """Dimension-order routing (x then y). Returns list of directed edges ((r1,c1)->(r2,c2))."""
        (r1, c1), (r2, c2) = u_xy, v_xy
        path = []
        # move in columns (c)
        step = 1 if c2 >= c1 else -1
        for c in range(c1, c2, step):
            path.append(((r1, c), (r1, c + step)))
        # move in rows (r)
        step = 1 if r2 >= r1 else -1
        for r in range(r1, r2, step):
            path.append(((r, c2), (r + step, c2)))
        return path  # list of edges
    
    def _pairwise_xor_partners(self, P: int, bit: int):
        """Generate unique (a,b) pairs where b = a XOR (1<<bit), a<b, for 0<=a<P."""
        mask = 1 << bit
        pairs = []
        for a in range(P):
            b = a ^ mask
            if b < P and a < b:
                pairs.append((a, b))
        return pairs

    def _mesh_contention_time(
        self, P: int, part_bit: int, bytes_per_partner: int, link_bw_Bps: float, per_hop_latency_s: float
    ):
        """
        Estimate time for a global XOR-exchange on a 2D mesh with Manhattan routing:
        - P endpoints on an r x c grid (near square).
        - Each pair exchanges bytes_per_partner in each direction (full-duplex assumed).
        - Time ≈ (max link load) / link_bw + (worst path hops) * per_hop_latency
        """
        rows, cols = self._mesh_dims(P)
        # Map linear id -> coords
        coords = [self._id_to_xy(i, cols) for i in range(P)]

        # Build link loads (treat edges as undirected for load aggregation)
        link_load_bytes = defaultdict(int)
        worst_hops = 0

        for a, b in self._pairwise_xor_partners(P, part_bit):
            path = self._manhattan_path(coords[a], coords[b])
            hops = len(path)
            if hops > worst_hops:
                worst_hops = hops
            # Two-way exchange; count both directions as aggregate load on the link
            exchange_bytes_on_path = 2 * bytes_per_partner
            # Divide equally over edges (store total bytes per edge)
            # More accurately, traffic is streamed; for bottleneck we just add per edge.
            for edge in path:
                # Canonicalize to undirected key
                (u, v) = edge
                key = (u, v) if u < v else (v, u)
                link_load_bytes[key] += exchange_bytes_on_path

        max_link_bytes = max(link_load_bytes.values()) if link_load_bytes else 0
        time_s = (max_link_bytes / link_bw_Bps) + worst_hops * per_hop_latency_s
        return time_s

    # ---------------------------
    # Communication wrappers
    # ---------------------------
    def _intra_node_comm_time(self, distributed_bit: int, bytes_per_partner: int):
        """
        Communication among devices within a node when a qubit lies in the intra-node partition.
        """
        topo = getattr(self.hardware, "intra_node_topology", "all-to-all")
        bw = self.hardware.intra_node_bandwidth_GBps * 1e9
        lat = self.hardware.intra_node_latency_ns * 1e-9

        if topo == "mesh":
            P = self.hardware.num_devices_per_node
            return self._mesh_contention_time(P, distributed_bit, bytes_per_partner, bw, lat)
        else:
            # Non-blocking fabric approximation
            return (bytes_per_partner / bw) + lat

    def _inter_node_comm_time(self, distributed_bit: int, bytes_per_partner: int):
        """
        Communication among nodes when a qubit lies in the inter-node partition.
        """
        topo = getattr(self.hardware, "inter_node_topology", "all-to-all")
        bw = self.hardware.inter_node_bandwidth_GBps * 1e9
        lat = self.hardware.inter_node_latency_ns * 1e-9

        if topo == "mesh":
            P = self.hardware.num_nodes
            return self._mesh_contention_time(P, distributed_bit, bytes_per_partner, bw, lat)
        else:
            # Non-blocking fabric approximation
            return (bytes_per_partner / bw) + lat
        
    # ---------------------------
    # Gate cost with contention
    # ---------------------------
    def gate_cost(self, gate_type, targets):
        """
        Estimate cost of applying a gate.
        For Schrödinger, each gate is applied to the full state vector.
        """
        mem_bandwidth = self.hardware.mem_bandwidth_GBps * 1e9  # B/s
        mem_latency = self.hardware.mem_latency_ns * 1e-9        # s
        amplitude_size = 16  # bytes (complex128)
        
        bytes_per_device = 2**self.local_qubits * amplitude_size
        bytes_per_node = 2**self.node_qubits * amplitude_size

        # Check targets in range
        if any(t >= self.circuit.num_qubits for t in targets):
            raise ValueError(f"Target qubit index out of range: {targets}")
        
        # --- Single-qubit gates ---
        if (gate_type in ("H", "X", "RX")):
            # assuming 28 flops (2 complex multiply + 2 complex add) per single-qubit gate
            compute_time = 14 * 2**(self.circuit.num_qubits) / (self.hardware.flops_per_device * self.hardware.num_nodes)
            t = targets[0]
            if t < self.local_qubits:
                comm_time = bytes_per_device * 2 / mem_bandwidth + mem_latency
                return max(compute_time, comm_time)  
            # Needs communication within the node
            elif t < self.node_qubits:
                # Partner bit within the intra-node partition:
                distributed_bit = t - self.local_qubits
                # Each device exchanges half of its local state with its XOR partner
                bytes_per_partner = bytes_per_device // 2
                comm_time = self._intra_node_comm_time(distributed_bit, bytes_per_partner)
                return max(compute_time, comm_time)
            # Inter-node distributed qubit
            else:
                distributed_bit = t - self.node_qubits  # which network bit at the node level
                bytes_per_partner = bytes_per_node // 2
                comm_time = self._inter_node_comm_time(distributed_bit, bytes_per_partner)
                return max(compute_time, comm_time)

        # # --- Two-qubit gates ---
        # t, c = targets
        # compute_time = 14 * 2**(self.circuit.num_qubits-1) / (self.hardware.flops_per_device * self.hardware.num_nodes)
        # # If qubits are on different nodes (i.e., their bits span partition)
        # if t < self.local_qubits:
        #     if c < self.local_qubits:
        #         # Both qubits are local
        #         return max(compute_time,bytes_per_device / mem_bandwidth + mem_latency)  # Local 2-qubit gate
        #     elif c < self.node_qubits:
        #         # target is local, control is in node
        #         return max(compute_time, bytes_per_device * 2 / mem_bandwidth + mem_latency) + intra_node_latency
        #     else:
        #         # target is local, control is remote
        #         return max(compute_time, bytes_per_device * 2 / mem_bandwidth + mem_latency) + inter_node_latency

        # elif t < self.node_qubits:
        #     if c < self.local_qubits:
        #         # control is local, target is in node
        #         return max(compute_time, bytes_per_device / intra_node_bw + intra_node_latency) + intra_node_latency
        #     elif c < self.node_qubits:
        #         # Both qubits are in node, but not local
        #         return max(compute_time, bytes_per_device * 2 / intra_node_bw + intra_node_latency) + intra_node_latency
        #     else:
        #         # control is remote, target is in node
        #         return max(compute_time, bytes_per_device * 2 / intra_node_bw + intra_node_latency) + intra_node_latency
        # else: 
        #     if c < self.local_qubits:
        #         # control is local, target is remote
        #         return max(compute_time, bytes_per_node / inter_node_bw + inter_node_latency) + inter_node_latency
        #     elif c < self.node_qubits:
        #         # control is in node, target is remote
        #         return max(compute_time, bytes_per_node / inter_node_bw + inter_node_latency) + inter_node_latency
        #     else:
        #         # Both qubits are remote
        #         return max(compute_time, bytes_per_node * 2 / inter_node_bw + inter_node_latency) + inter_node_latency
        
        # --- Two-qubit gates ---
        # model compute roughly half work per amplitude pair (similar scale to your 14*2^(n-1))
        t, c = targets
        compute_time = 14 * 2 ** (self.circuit.num_qubits - 1) / (self.hardware.flops_per_device * self.hardware.num_nodes)

        # Cases break down by which partitions t/c belong to.
        # If a gate *touches* any distributed partition, we model one XOR-exchange on that partition.
        # (This is a common approximation for state-vector sims that use "on-the-fly" exchanges instead of explicit qubit remaps.)
        def level(q):
            if q < self.local_qubits: return "local"
            if q < self.node_qubits:  return "intra"
            return "inter"

        lt, lc = level(t), level(c)

        # Local target
        if lt == "local":
            if lc == "local":
                comm_time = (bytes_per_device * 2 / mem_bandwidth) + mem_latency
                return max(compute_time, comm_time)
            else:
                comm_time = (bytes_per_device / mem_bandwidth) + mem_latency
                return max(compute_time, comm_time)
        # Intra-node target
        elif lt == "intra":
            if lc == "local":
                comm_time = 




    def simulate(self):
        total_time = 0.0
        total_comp_time = 0.0
        total_comm_time = 0.0
        for gate in self.circuit.gates:
            gate_type, targets = gate
            total_time += self.gate_cost(gate_type, targets)
        return total_time


# Example usage
if __name__ == "__main__":
    # 4-qubit GHZ circuit
    circuit = QuantumCircuit(
        num_qubits=25,
        # gates=[ # either a list of gates or "random"
        #     ("H", [0]),
        #     ("CNOT", [0, 1]),
        #     ("CNOT", [1, 2]),
        #     ("CNOT", [2, 3]),
        # ], 
        gates="random",  # Randomly generate gates
        num_gates=100,  # Randomly generate 100 gates
        sim_type="Schrödinger"  # Simulation type can be Schrödinger, Feynman, or tensor network
    )

    # Example hardware configuration from qHiPSTER 2016 paper (TACC Stampede)
    hardware = HardwareConfig(num_nodes=1024, mem_bandwidth_GBps= 40, inter_node_bandwidth_GBps= 5.5, mem_latency_ns=100, inter_node_latency_ns=1000, mem_per_device_GB=32, flops_per_device=8*16*2*(2.7e9))
 
    simulator = PerformanceSimulator(circuit, hardware)
    time_taken = simulator.simulate()

    print(f"Simulated total execution time: {time_taken * 1e3:.2f} ms")
