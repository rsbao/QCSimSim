import math
import random

class HardwareConfig:
    def __init__(self, 
                num_nodes, 
                mem_bandwidth_GBps,
                mem_latency_ns,
                inter_node_latency_ns,
                mem_per_device_GB,
                inter_node_bandwidth_GBps=0,
                flops_per_device=1,
                num_devices_per_node=1,
                intra_node_bandwidth_GBps=0.0,
                intra_node_latency_ns=0.0,
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
    
    def gate_cost(self, gate_type, targets):
        """
        Estimate cost of applying a gate.
        For Schrödinger, each gate is applied to the full state vector.
        """
        mem_bandwidth = self.hardware.mem_bandwidth_GBps * 1e9  # B/s
        mem_latency = self.hardware.mem_latency_ns * 1e-9        # s
        inter_node_bw = self.hardware.inter_node_bandwidth_GBps * 1e9  # B/s
        intra_node_bw = self.hardware.intra_node_bandwidth_GBps * 1e9  # B/s
        inter_node_latency = self.hardware.inter_node_latency_ns * 1e-9
        intra_node_latency = self.hardware.intra_node_latency_ns * 1e-9
        amplitude_size = 16  # bytes (complex128)
        bytes_per_device = 2**self.local_qubits * amplitude_size
        bytes_per_node = 2**self.node_qubits * amplitude_size

        # Check targets in range
        if any(t >= self.circuit.num_qubits for t in targets):
            raise ValueError(f"Target qubit index out of range: {targets}")
        
        # If single-qubit gate on local qubit
        if (gate_type == "H" or gate_type == "X" or gate_type == "RX"):
            # assuming 28 flops (2 complex multiply + 2 complex add) per single-qubit gate
            compute_time = 14 * 2**(self.circuit.num_qubits) / (self.hardware.flops_per_device * self.hardware.num_nodes)
            if targets[0] < self.local_qubits:
                comm_time = bytes_per_device * 2 / mem_bandwidth + mem_latency
                return max(compute_time, comm_time)  
            elif targets[0] < self.node_qubits:
                # Needs communication within the node
                # map communication to network
                comm_time = bytes_per_device * 2 / intra_node_bw + intra_node_latency
                return max(compute_time, comm_time) + intra_node_latency
            else:
                # Needs communication across nodes
                comm_time = bytes_per_node * 2 / inter_node_bw + inter_node_latency
                return max(compute_time, comm_time) + inter_node_latency

        # Two-qubit gate: estimate internode communication
        t, c = targets
        compute_time = 14 * 2**(self.circuit.num_qubits-1) / (self.hardware.flops_per_device * self.hardware.num_nodes)
        # If qubits are on different nodes (i.e., their bits span partition)
        if t < self.local_qubits:
            if c < self.local_qubits:
                # Both qubits are local
                return max(compute_time,bytes_per_device / mem_bandwidth + mem_latency)  # Local 2-qubit gate
            elif c < self.node_qubits:
                # target is local, control is in node
                return max(compute_time, bytes_per_device * 2 / mem_bandwidth + mem_latency) + intra_node_latency
            else:
                # target is local, control is remote
                return max(compute_time, bytes_per_device * 2 / mem_bandwidth + mem_latency) + inter_node_latency

        elif t < self.node_qubits:
            if c < self.local_qubits:
                # control is local, target is in node
                return max(compute_time, bytes_per_device / intra_node_bw + intra_node_latency) + intra_node_latency
            elif c < self.node_qubits:
                # Both qubits are in node, but not local
                return max(compute_time, bytes_per_device * 2 / intra_node_bw + intra_node_latency) + intra_node_latency
            else:
                # control is remote, target is in node
                return max(compute_time, bytes_per_device * 2 / intra_node_bw + intra_node_latency) + intra_node_latency
        else: 
            if c < self.local_qubits:
                # control is local, target is remote
                return max(compute_time, bytes_per_node / inter_node_bw + inter_node_latency) + inter_node_latency
            elif c < self.node_qubits:
                # control is in node, target is remote
                return max(compute_time, bytes_per_node / inter_node_bw + inter_node_latency) + inter_node_latency
            else:
                # Both qubits are remote
                return max(compute_time, bytes_per_node * 2 / inter_node_bw + inter_node_latency) + inter_node_latency
        

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

    hardware = HardwareConfig(num_nodes=1024, mem_bandwidth_GBps= 40, inter_node_bandwidth_GBps= 5.5, mem_latency_ns=100, inter_node_latency_ns=1000, mem_per_device_GB=32, flops_per_device=8*16*2*(2.7e9))

    simulator = PerformanceSimulator(circuit, hardware)
    time_taken = simulator.simulate()

    print(f"Simulated total execution time: {time_taken * 1e3:.2f} ms")
