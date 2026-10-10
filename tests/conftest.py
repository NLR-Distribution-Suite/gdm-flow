from pathlib import Path

import pytest

from gdm.systems.distribution import DistributionSystem


@pytest.fixture(scope="session")
def p5r_system():
    """Load the bundled p5r model and its sidecars without network access."""
    model_path = Path(__file__).resolve().parents[1] / "examples/models/p5r.json"
    return DistributionSystem.from_json(model_path)
