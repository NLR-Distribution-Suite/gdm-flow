"""Run LinDistFlow example using a downloaded demo DistributionSystem model."""

from pathlib import Path

from gdm.systems.distribution import DistributionSystem
from gdm_flow import solve_lindistflow


def main() -> None:
    model_path = Path(__file__).resolve().parent / "models" / "p5r.json"
    system = DistributionSystem.from_json(str(model_path))

    result = solve_lindistflow(system)

    print("=== LinDistFlow Example ===")
    print(f"success: {result.success}")
    print(f"message: {result.message}")
    print(f"source_bus: {result.source_bus}")

    print("sample bus voltages:")
    for key, v in list(result.voltage_v.items())[:5]:
        print(f"  {key}: {v:.3f} V")

    print("sample branch P flows:")
    for key, p in list(result.p_flow_w.items())[:5]:
        print(f"  {key}: {p:.3f} W")


if __name__ == "__main__":
    main()
