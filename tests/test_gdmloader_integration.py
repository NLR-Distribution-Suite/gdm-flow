import numpy as np

from gdm.systems.distribution import DistributionSystem


def test_bundled_model_loads_distribution_system(p5r_system):
    assert isinstance(p5r_system, DistributionSystem)
    assert p5r_system.get_source_bus().name


def test_bundled_model_compare_solvers(p5r_system):
    """Run all solvers on the bundled model and compare results."""
    from gdm_flow.ac_opf import optimize_ac_power_flow_from_components
    from gdm_flow.ac_pf import solve_ac_power_flow_from_components
    from gdm_flow.lindistflow import solve_lindistflow

    system = p5r_system

    # Run AC PF
    pf_result = solve_ac_power_flow_from_components(
        system, include_loads=True, include_solar=True, include_capacitor=True
    )
    assert pf_result.success

    # Run AC OPF
    opf_result = optimize_ac_power_flow_from_components(
        system,
        include_loads=True,
        include_solar=True,
        include_capacitor=True,
        include_regulator_targets=True,
        include_regulator_limits=True,
    )
    assert opf_result.success

    # Run LinDistFlow
    ldf_result = solve_lindistflow(system)
    assert ldf_result.success

    # Compare source power between AC OPF and AC PF (should be close)
    src_bus = system.get_source_bus().name
    idx_map = pf_result.ybus_result.index_to_label

    def _source_power(result):
        v = result.voltage
        ybus = result.ybus_result.ybus
        s = v * np.conj(ybus @ v)
        src_idx = [i for i, lbl in enumerate(idx_map) if lbl[0] == src_bus]
        return sum(s[i].real for i in src_idx), sum(s[i].imag for i in src_idx)

    pf_p, pf_q = _source_power(pf_result)
    opf_p, opf_q = _source_power(opf_result)

    # OPF should match PF within 1% for source real power
    assert abs(opf_p - pf_p) / abs(pf_p) < 0.01, (
        f"OPF source P ({opf_p:.1f} W) differs from PF ({pf_p:.1f} W) by more than 1%"
    )

    # Voltage ranges should be reasonable (0.85 to 1.05 pu)
    from gdm_flow.ac_opf import _build_nominal_voltage_map

    nominal_map = _build_nominal_voltage_map(system)
    for name, result in [("PF", pf_result), ("OPF", opf_result)]:
        vm_pu = []
        for idx, label in enumerate(result.ybus_result.index_to_label):
            if label[1] == "N":
                continue
            nom = nominal_map.get(label, 0.0)
            if nom > 0:
                vm_pu.append(float(abs(result.voltage[idx])) / nom)
        assert min(vm_pu) > 0.85, f"{name} min voltage {min(vm_pu):.4f} below 0.85 pu"
        assert max(vm_pu) < 1.05, f"{name} max voltage {max(vm_pu):.4f} above 1.05 pu"
