"""Run DC OPF example using a downloaded demo DistributionSystem model."""

from pathlib import Path

from gdm.systems.distribution import DistributionSystem
from gdm_flow import solve_dc_opf_from_components


def main() -> None:
    model_path = Path(__file__).resolve().parent / "models" / "p5r.json"
    system = DistributionSystem.from_json(str(model_path))

    result = solve_dc_opf_from_components(
        system,
        include_solar_generators=True,
        include_battery_generators=True,
        include_loads=True,
    )

    print("=== DC OPF Example ===")
    print(f"success: {result.success}")
    print(f"message: {result.message}")
    print(f"iterations: {result.iterations}")
    print(f"objective: {result.objective:.6f}")
    print(f"slack_injection_w: {result.slack_injection_w:.3f}")

    print("sample generator dispatch:")
    for name, p in list(result.generator_dispatch_w.items())[:5]:
        print(f"  {name}: {p:.3f} W")


if __name__ == "__main__":
    main()
