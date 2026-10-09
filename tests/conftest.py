import pytest

from gdm.systems.distribution import DistributionSystem


@pytest.fixture(scope="session")
def p5r_system():
    """Download p5r once per session; dependency and download failures are errors."""
    from gdmloader.constants import GCS_CASE_SOURCE
    from gdmloader.source import SystemLoader

    loader = SystemLoader()
    loader.add_source(GCS_CASE_SOURCE)
    return loader.load_dataset(
        system_type=DistributionSystem,
        source_name=GCS_CASE_SOURCE.name,
        dataset_name="p5r",
    )
