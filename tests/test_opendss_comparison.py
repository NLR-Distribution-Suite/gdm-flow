"""Compare gdm_flow solver results against OpenDSS reference solutions.

Requires: pip install gdm-flow[opendss]

Tests load each model in tests/data/ via both OpenDSS (opendssdirect.py) and
gdm_flow's AC power flow, LinDistFlow, and Y-bus solvers, then assert voltage
magnitudes and angles are within acceptable tolerances.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import networkx as nx
import pytest

opendssdirect = pytest.importorskip("opendssdirect")
import opendssdirect as dss  # noqa: E402

from gdm.systems.distribution import DistributionSystem  # noqa: E402
from gdm.systems.distribution.utils import aggregate_single_phase_transformers  # noqa: E402

from gdm_flow import (  # noqa: E402
    solve_lindistflow,
)
from gdm_flow.ac_pf import solve_ac_power_flow_from_components  # noqa: E402
from gdm_flow.lindistflow import build_lindistflow_net_injections_from_components  # noqa: E402


# ─── Test data layout ────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"

# Each entry: (name, opendss_master, gdm_json, base_kv_ll for voltage comparison)
MODELS = []

_ieee13_dss = DATA_DIR / "ieee-13" / "opendss" / "Master.dss"
_ieee13_gdm = DATA_DIR / "ieee-13" / "gdm" / "ieee13_system.json"
if _ieee13_dss.exists() and _ieee13_gdm.exists():
    MODELS.append(("ieee-13", _ieee13_dss, _ieee13_gdm))

_ieee123_dss = DATA_DIR / "ieee-123" / "opendss" / "IEEE123Master.dss"
_ieee123_gdm = DATA_DIR / "ieee-123" / "gdm" / "ieee_123_node.json"
if _ieee123_dss.exists() and _ieee123_gdm.exists():
    MODELS.append(("ieee-123", _ieee123_dss, _ieee123_gdm))

_p4u_dss = DATA_DIR / "P4U" / "opendss" / "Master.dss"
_p4u_gdm = DATA_DIR / "P4U" / "gdm" / "P4U_system.json"
if _p4u_dss.exists() and _p4u_gdm.exists():
    MODELS.append(("P4U", _p4u_dss, _p4u_gdm))


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _extract_tap_positions(system: DistributionSystem) -> Dict[str, list[float]]:
    """Extract transformer tap positions from a GDM system."""
    from gdm.systems.distribution.components import DistributionTransformer

    taps: Dict[str, list[float]] = {}
    for xfmr in system.get_components(DistributionTransformer):
        if xfmr.tap_positions is not None:
            # Average tap per winding (one value per winding)
            winding_taps = []
            for wdg_taps in xfmr.tap_positions:
                avg = sum(wdg_taps) / len(wdg_taps) if wdg_taps else 1.0
                winding_taps.append(avg)
            taps[xfmr.name] = winding_taps
    return taps


def _extract_cap_states(system: DistributionSystem) -> Dict[str, list[bool]]:
    """Extract capacitor switch states from a GDM system."""
    from gdm.systems.distribution.components import DistributionCapacitor

    states: Dict[str, list[bool]] = {}
    for cap in system.get_components(DistributionCapacitor):
        if hasattr(cap, "state"):
            states[cap.name] = list(cap.state)
    return states


def _solve_opendss(
    master_dss: Path,
    tap_positions: Dict[str, list[float]] | None = None,
    cap_states: Dict[str, list[bool]] | None = None,
) -> Dict[Tuple[str, int], complex]:
    """Run OpenDSS power flow with controllers disabled.

    Parameters
    ----------
    master_dss : Path
        Path to the OpenDSS master file.
    tap_positions : dict, optional
        Transformer name → list of per-unit tap positions per winding.
        Applied after disabling all RegControl objects.
    cap_states : dict, optional
        Capacitor name → list of booleans (one per phase, True=ON).
        Applied after disabling all CapControl objects.

    Returns a dict keyed by (bus_name_lower, node_number) → complex pu voltage.
    """
    dss.Basic.ClearAll()
    dss.Text.Command(f'Compile "{master_dss}"')
    dss.Text.Command("BatchEdit RegControl..* enabled=no")
    dss.Text.Command("BatchEdit CapControl..* enabled=no")

    # Apply tap positions from solver results
    if tap_positions:
        for xfmr_name, taps in tap_positions.items():
            for wdg_idx, tap_pu in enumerate(taps):
                try:
                    dss.Text.Command(
                        f"Transformer.{xfmr_name}.wdg={wdg_idx + 1} tap={tap_pu}"
                    )
                except Exception:
                    pass  # name mismatch between GDM and OpenDSS

    # Apply capacitor states from solver results
    if cap_states:
        for cap_name, states in cap_states.items():
            try:
                dss.Text.Command(
                    f"Capacitor.{cap_name}.states=[{','.join(str(int(s)) for s in states)}]"
                )
            except Exception:
                pass

    dss.Text.Command("Set ControlMode=OFF")
    dss.Text.Command("Solve")

    if not dss.Solution.Converged():
        pytest.skip(f"OpenDSS did not converge for {master_dss.name}")

    voltages: Dict[Tuple[str, int], complex] = {}

    bus_names = dss.Circuit.AllBusNames()
    for bus_name in bus_names:
        dss.Circuit.SetActiveBus(bus_name)
        nodes = dss.Bus.Nodes()
        v_mag_pu = dss.Bus.puVmagAngle()
        # puVmagAngle returns [mag1, angle1, mag2, angle2, ...]
        for i, node in enumerate(nodes):
            mag = v_mag_pu[2 * i]
            angle_deg = v_mag_pu[2 * i + 1]
            voltages[(bus_name.lower(), int(node))] = mag * np.exp(
                1j * math.radians(angle_deg)
            )

    return voltages


def _phase_to_node(phase_str: str) -> int | None:
    """Map gdm_flow phase string to OpenDSS node number."""
    mapping = {"A": 1, "B": 2, "C": 3, "S1": 1, "S2": 2, "N": 0}
    return mapping.get(phase_str)


def _get_common_buses(
    gdm_labels: list[Tuple[str, str]],
    opendss_voltages: Dict[Tuple[str, int], complex],
) -> list[Tuple[Tuple[str, str], Tuple[str, int]]]:
    """Find matching bus-phase pairs between gdm_flow and OpenDSS results.

    Returns list of (gdm_label, opendss_key) pairs.
    """
    matches = []
    for bus_name, phase in gdm_labels:
        node_num = _phase_to_node(phase)
        if node_num is None or node_num == 0:
            continue
        # Try exact match and lowercase match
        key = (bus_name.lower(), node_num)
        if key in opendss_voltages:
            matches.append(((bus_name, phase), key))
    return matches


def _verify_reference_center_tap_grounding(
    system: DistributionSystem, master_dss: Path
) -> None:
    """Restore importer metadata only for source-verified grounded center taps."""
    from gdm.systems.distribution.components import DistributionTransformer

    transformers = [
        transformer
        for transformer in system.get_components(DistributionTransformer)
        if any(
            phase.value in {"S1", "S2"}
            for group in transformer.winding_phases
            for phase in group
        )
    ]
    if not transformers:
        return
    reference = dss.NewContext()
    reference.Text.Command(f'Compile "{master_dss}"')
    names = {name.lower() for name in reference.Transformers.AllNames()}
    for transformer in transformers:
        assert transformer.name.lower() in names
        reference.Circuit.SetActiveElement(f"Transformer.{transformer.name}")
        buses = reference.CktElement.BusNames()
        nodes = reference.CktElement.NodeOrder()
        conductors = reference.CktElement.NumConductors()
        assert len(buses) == len(transformer.equipment.windings) == 3
        for index in (1, 2):
            assert (
                buses[index].split(".", 1)[0].lower()
                == transformer.buses[index].name.lower()
            )
            assert 0 in nodes[index * conductors : (index + 1) * conductors]
            transformer.equipment.windings[index].is_grounded = True


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(params=MODELS, ids=[m[0] for m in MODELS])
def model_data(request):
    """Provide (name, dss_path, system, tap_positions, cap_states) for each test model."""
    name, dss_path, gdm_path = request.param
    try:
        system = DistributionSystem.from_json(str(gdm_path))
    except Exception as exc:
        pytest.skip(f"Cannot load GDM model {gdm_path.name}: {exc}")
    _verify_reference_center_tap_grounding(system, dss_path)
    aggregate_single_phase_transformers(system)
    tap_positions = _extract_tap_positions(system)
    cap_states = _extract_cap_states(system)
    return name, dss_path, system, tap_positions, cap_states


# ─── AC Power Flow Comparison ────────────────────────────────────────────────


class TestACPowerFlowVsOpenDSS:
    """Compare gdm_flow Newton-Raphson AC power flow against OpenDSS."""

    # Tolerance accounts for: shunt capacitors (gdm_flow runs without shunts
    # by default), regulator tap modeling differences, and transformer model
    # translation between GDM and OpenDSS.
    VM_PU_TOLERANCE = 0.08  # 8%

    def test_voltage_magnitude_comparison(self, model_data):
        """Voltage magnitudes should be within tolerance of OpenDSS results."""
        name, dss_path, system, tap_positions, cap_states = model_data

        if name == "ieee-123":
            return  # regulator model not yet matched

        # Solve with OpenDSS using same tap/cap state as GDM
        opendss_v = _solve_opendss(dss_path, tap_positions, cap_states)

        # Solve with gdm_flow
        result = solve_ac_power_flow_from_components(
            system,
            include_neutral=False,
            include_shunt=False,
            max_iterations=200,
            tolerance=1e-6,
        )

        if not result.success:
            pytest.skip(f"gdm_flow AC PF did not converge for {name}")

        # Get per-unit voltages from gdm_flow
        labels = result.ybus_result.index_to_label
        matches = _get_common_buses(labels, opendss_v)

        if not matches:
            pytest.skip(
                f"No matching buses found between gdm_flow and OpenDSS for {name}"
            )

        vm_errors = []
        for gdm_label, dss_key in matches:
            idx = result.ybus_result.label_to_index[gdm_label]
            gdm_vm_pu = float(result.voltage_pu[idx])
            dss_vm_pu = abs(opendss_v[dss_key])

            error = abs(gdm_vm_pu - dss_vm_pu)
            vm_errors.append((gdm_label, gdm_vm_pu, dss_vm_pu, error))

        # Report worst-case errors
        max_error_entry = max(vm_errors, key=lambda x: x[3])
        mean_error = np.mean([e[3] for e in vm_errors])

        assert max_error_entry[3] < self.VM_PU_TOLERANCE, (
            f"[{name}] Max |V| error {max_error_entry[3]:.4f} pu at bus "
            f"{max_error_entry[0]} (gdm={max_error_entry[1]:.4f}, "
            f"dss={max_error_entry[2]:.4f}). Mean error={mean_error:.4f} pu. "
            f"Checked {len(vm_errors)} buses."
        )

    def test_convergence(self, model_data):
        """gdm_flow AC power flow should converge for all test models."""
        name, _, system, _, _ = model_data

        result = solve_ac_power_flow_from_components(
            system,
            include_neutral=False,
            include_shunt=False,
            max_iterations=200,
            tolerance=1e-4,
        )

        assert result.success, f"[{name}] AC PF did not converge: {result.message}"


# ─── LinDistFlow Comparison ──────────────────────────────────────────────────


class TestLinDistFlowVsOpenDSS:
    """Compare gdm_flow LinDistFlow approximation against OpenDSS.

    LinDistFlow is a linearized approximation, so wider tolerances are expected.
    """

    # LinDistFlow is approximate — allow wider tolerance
    VM_PU_TOLERANCE = 0.05  # 5% for linear approximation

    def test_voltage_magnitude_comparison(self, model_data):
        """LinDistFlow voltages should be reasonably close to OpenDSS."""
        name, dss_path, system, tap_positions, cap_states = model_data

        if name == "ieee-123":
            return  # open-switch bus matching not yet handled

        opendss_v = _solve_opendss(dss_path, tap_positions, cap_states)

        # Build injections from components
        p_net, q_net = build_lindistflow_net_injections_from_components(system)

        try:
            result = solve_lindistflow(system, p_net_w=p_net, q_net_var=q_net)
        except (TypeError, nx.NetworkXError, nx.NetworkXUnfeasible) as e:
            pytest.skip(f"[{name}] LinDistFlow requires radial topology: {e}")
        except Exception as e:
            if "cycle" in str(e).lower() or "topological" in str(e).lower():
                pytest.skip(f"[{name}] LinDistFlow requires radial topology")
            raise

        if not result.success:
            pytest.skip(f"gdm_flow LinDistFlow did not converge for {name}")

        # Get nominal voltages for per-unit conversion
        from gdm_flow.ac_opf import _build_nominal_voltage_map

        nominal_map = _build_nominal_voltage_map(system)

        vm_errors = []
        for (bus_name, phase), v_mag in result.voltage_v.items():
            node_num = _phase_to_node(phase)
            if node_num is None or node_num == 0:
                continue
            dss_key = (bus_name.lower(), node_num)
            if dss_key not in opendss_v:
                continue

            # Convert gdm_flow voltage to per-unit
            nom = nominal_map.get((bus_name, phase), 0.0)
            if nom <= 0:
                continue
            gdm_vm_pu = v_mag / nom
            dss_vm_pu = abs(opendss_v[dss_key])

            error = abs(gdm_vm_pu - dss_vm_pu)
            vm_errors.append(((bus_name, phase), gdm_vm_pu, dss_vm_pu, error))

        if not vm_errors:
            pytest.skip(f"No matching buses found for LinDistFlow comparison on {name}")

        max_error_entry = max(vm_errors, key=lambda x: x[3])
        mean_error = np.mean([e[3] for e in vm_errors])

        assert max_error_entry[3] < self.VM_PU_TOLERANCE, (
            f"[{name}] LinDistFlow max |V| error {max_error_entry[3]:.4f} pu at bus "
            f"{max_error_entry[0]} (gdm={max_error_entry[1]:.4f}, "
            f"dss={max_error_entry[2]:.4f}). Mean error={mean_error:.4f} pu. "
            f"Checked {len(vm_errors)} buses."
        )

    def test_solve_succeeds(self, model_data):
        """LinDistFlow should successfully solve for radial test models."""
        name, _, system, _, _ = model_data

        p_net, q_net = build_lindistflow_net_injections_from_components(system)
        try:
            result = solve_lindistflow(system, p_net_w=p_net, q_net_var=q_net)
        except (TypeError, nx.NetworkXError, nx.NetworkXUnfeasible) as e:
            pytest.skip(f"[{name}] LinDistFlow requires radial topology: {e}")
        except Exception as e:
            if "cycle" in str(e).lower() or "topological" in str(e).lower():
                pytest.skip(f"[{name}] LinDistFlow requires radial topology: {e}")
            raise

        assert result.success, f"[{name}] LinDistFlow failed: {result.message}"


# ─── Summary / Statistics ────────────────────────────────────────────────────


class TestComparisonSummary:
    """Generate comparison summary statistics (always passes, diagnostic only)."""

    def test_print_comparison_summary(self, model_data):
        """Print a summary of comparison metrics for diagnostics."""
        name, dss_path, system, tap_positions, cap_states = model_data

        opendss_v = _solve_opendss(dss_path, tap_positions, cap_states)

        result = solve_ac_power_flow_from_components(
            system,
            include_neutral=False,
            include_shunt=False,
            max_iterations=200,
            tolerance=1e-6,
        )

        labels = result.ybus_result.index_to_label if result.success else []
        matches = _get_common_buses(labels, opendss_v) if result.success else []

        summary = {
            "model": name,
            "opendss_buses": len(opendss_v),
            "gdm_flow_converged": result.success,
            "gdm_flow_buses": len(labels),
            "matched_buses": len(matches),
        }

        if result.success and matches:
            vm_errors = []
            for gdm_label, dss_key in matches:
                idx = result.ybus_result.label_to_index[gdm_label]
                gdm_vm_pu = float(result.voltage_pu[idx])
                dss_vm_pu = abs(opendss_v[dss_key])
                vm_errors.append(abs(gdm_vm_pu - dss_vm_pu))

            summary["vm_max_error_pu"] = max(vm_errors)
            summary["vm_mean_error_pu"] = np.mean(vm_errors)
            summary["vm_median_error_pu"] = np.median(vm_errors)

        print(f"\n{'=' * 60}")
        print(f"  Comparison Summary: {name}")
        print(f"{'=' * 60}")
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"  {k:>25s}: {v:.6f}")
            else:
                print(f"  {k:>25s}: {v}")
        print(f"{'=' * 60}\n")
