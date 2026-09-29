"""The GLB adapter preserves source URDFs and resolves temporary mesh paths."""

import struct

import pytest

from bello_mujoco.models.urdf import import_urdf_model


def test_glb_import_uses_stl_companions_and_preserves_mixed_relative_meshes(tmp_path):
    # A closed tetrahedron is sufficient to exercise MuJoCo's real mesh importer.
    vertices = [(0, 0, 0), (100, 0, 0), (0, 100, 0), (0, 0, 100)]
    faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
    mesh = bytearray(80) + struct.pack("<I", len(faces))
    for face in faces:
        values = [0, 0, 0] + [axis for index in face for axis in vertices[index]]
        mesh += struct.pack("<12fH", *values, 0)
    meshes = tmp_path / "meshes"
    meshes.mkdir()
    (meshes / "new.stl").write_bytes(mesh)
    (meshes / "old.stl").write_bytes(mesh)
    source = """<robot name="adapter">
    <mujoco><compiler discardvisual="false"/></mujoco>
    <link name="root"><inertial><mass value="1"/>
    <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><geometry><mesh filename="meshes/new.glb" scale=".001 .001 .001"/></geometry></visual>
    <visual><origin xyz="1 0 0"/><geometry><mesh filename="meshes/old.stl" scale=".001 .001 .001"/></geometry></visual>
    </link></robot>"""
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(source)
    model = import_urdf_model(urdf)
    assert model.nmesh == 2
    assert model.ngeom == 2
    assert urdf.read_text() == source
    (meshes / "new.stl").unlink()
    with pytest.raises(FileNotFoundError, match="STL companion"):
        import_urdf_model(urdf)
