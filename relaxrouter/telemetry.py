"""Best-effort NVML measurements; unsupported counters remain null."""
import importlib.util
import threading
import time
import statistics


def available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


class Telemetry:
    def __init__(self):
        self.handle = None
        self.idle_power = None
        self.prepared = False
        self.samples = []
        self.thread = None

    def prepare(self, device: str) -> None:
        if self.prepared or not device.startswith("cuda"):
            return
        self.prepared = True
        try:
            import pynvml as nv
            import torch
            nv.nvmlInit()
            # UUID mapping respects CUDA_VISIBLE_DEVICES and avoids sampling the wrong GPU.
            props = torch.cuda.get_device_properties(device)
            uuid = str(props.uuid)
            self.handle = nv.nvmlDeviceGetHandleByUUID(uuid)
            self.nv = nv
            readings = []
            for _ in range(100):
                readings.append(nv.nvmlDeviceGetPowerUsage(self.handle) / 1000)
                time.sleep(0.1)
            self.idle_power = statistics.mean(readings)
        except Exception:
            self.handle = None

    def energy(self):
        try:
            return self.nv.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1000
        except Exception:
            return None

    def start(self) -> None:
        self.samples = []
        self.first = self.energy()
        self.event = threading.Event()
        if self.handle:
            self.thread = threading.Thread(target=self.sample, daemon=True)
            self.thread.start()

    def sample(self) -> None:
        while not self.event.is_set():
            try:
                self.samples.append((self.nv.nvmlDeviceGetUtilizationRates(self.handle).gpu,
                                     self.nv.nvmlDeviceGetPowerUsage(self.handle) / 1000))
            except Exception:
                pass
            self.event.wait(0.1)

    def stop(self) -> dict:
        self.event.set()
        if self.thread:
            self.thread.join()
            self.thread = None
        last = self.energy()
        energy = max(0, last - self.first) if last is not None and self.first is not None else None
        return dict(gpu_energy_j=energy,
                    mean_utilization=statistics.mean(s[0] for s in self.samples) if self.samples else None,
                    mean_power_w=statistics.mean(s[1] for s in self.samples) if self.samples else None)


def summarize(attempts: list, idle_power, intensity: float) -> dict:
    measured = [a for a in attempts if "wall_time" in a]
    energies = [a.get("gpu_energy_j") for a in measured]
    energy = sum(energies) if energies and all(e is not None for e in energies) else None
    wall = sum(a["wall_time"] for a in measured)
    def weighted(key):
        valid = [a for a in measured if a.get(key) is not None]
        duration = sum(a["wall_time"] for a in valid)
        return sum(a[key]*a["wall_time"] for a in valid)/duration if duration else None
    return dict(gpu_energy_j=energy, mean_utilization=weighted("mean_utilization"), mean_power_w=weighted("mean_power_w"),
                idle_power_w=idle_power, idle_share=idle_power*wall/energy if idle_power is not None and energy else None,
                carbon_kg=energy/3_600_000*intensity if energy is not None else None)
