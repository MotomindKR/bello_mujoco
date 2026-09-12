from dataclasses import replace

import mujoco
import pytest

from bello_mujoco.terrain import TerrainConfig, runtime_mjcf_path


@pytest.mark.parametrize("mode", ["plane", "rough"])
def test_runtime_scenes_are_independent_without_modifying_source(tmp_path, mode):
    source = tmp_path / "robot.xml"
    content = '<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>'
    source.write_text(content)
    config = TerrainConfig(mode=mode, hfield_resolution=3)
    first = runtime_mjcf_path(source, config)
    original = first.read_bytes()
    second = runtime_mjcf_path(source, replace(config, friction=(0.7, 0.01, 0.001)))
    assert first != second
    assert first.read_bytes() == original
    assert source.read_text() == content
    model = mujoco.MjModel.from_xml_path(str(second))
    assert model.geom(config.name).friction[0] == pytest.approx(0.7)
    assert runtime_mjcf_path(source, TerrainConfig()) == source
