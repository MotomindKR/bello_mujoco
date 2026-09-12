"""Native MJCF serialization with full-precision physical inertias."""

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco


def save_model_xml(path: Path, model: mujoco.MjModel) -> ET.ElementTree:
    """Save the last-loaded XML topology with this model's runtime values.

    MuJoCo's XML writer rounds numbers; retain full precision for small inertias.
    The caller must load the matching source immediately before this call.
    """
    mujoco.mj_saveLastXML(str(path), model)
    tree = ET.parse(path)
    tree.getroot().find("compiler").set("inertiafromgeom", "false")
    for index, body in enumerate(tree.findall(".//body"), start=1):
        inertial = body.find("inertial")
        if inertial is not None:
            inertial.set("mass", f"{model.body_mass[index]:.17g}")
            for attribute, values in (
                ("pos", model.body_ipos[index]),
                ("quat", model.body_iquat[index]),
                ("diaginertia", model.body_inertia[index]),
            ):
                inertial.set(attribute, " ".join(f"{value:.17g}" for value in values))
    return tree
