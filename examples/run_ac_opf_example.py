"""Run AC OPF example using a downloaded demo DistributionSystem model."""

from pathlib import Path

from gdm.systems.distribution import DistributionSystem
from gdm_flow import optimize_ac_power_flow_from_components


def main() -> None:
    model_path = Path(__file__).resolve().parent / "models" / "p5r.json"
    system = DistributionSystem.from_json(str(model_path))

    result = optimize_ac_power_flow_from_components(
        system,
        include_loads=True,
        include_solar=True,
        include_capacitor=True,
        include_regulator_targets=True,
        include_regulator_limits=True,
    )

    print("=== AC OPF Example ===")
    print(f"success: {result.success}")
    print(f"message: {result.message}")
    print(f"iterations: {result.iterations}")
    print(f"final_objective: {result.final_objective:.6e}")

    print("sample voltages:")
    for label, v in list(zip(result.ybus_result.index_to_label, result.voltage))[:5]:
        print(f"  {label}: {abs(v):.3f} V")


if __name__ == "__main__":
    main()
