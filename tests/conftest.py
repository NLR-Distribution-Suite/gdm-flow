import pytest

from gdm.systems.distribution import DistributionSystem


@pytest.fixture(scope="session")
def p5r_system():
    """Download p5r model once per session; skip all tests if download fails."""
    try:
        from gdmloader.constants import GCS_CASE_SOURCE
        from gdmloader.source import SystemLoader

        loader = SystemLoader()
        loader.add_source(GCS_CASE_SOURCE)
        system = loader.load_dataset(
            system_type=DistributionSystem,
            source_name=GCS_CASE_SOURCE.name,
            dataset_name="p5r",
        )
    except Exception as exc:
        pytest.skip(f"Could not download p5r model: {exc}")
    return system
