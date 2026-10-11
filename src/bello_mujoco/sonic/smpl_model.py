"""Extract SONIC's neutral 24-point human FK geometry from licensed SMPL-X."""

import argparse
from pathlib import Path

import numpy as np

from bello_mujoco.sonic.corpus import file_sha256
from bello_mujoco.sonic.smpl import SMPL_GEOMETRY_FORMAT, SMPL_PARENTS


def prepare_model(smplx: Path, output: Path):
    with np.load(smplx, allow_pickle=False) as source:
        vertices = np.asarray(source["v_template"], dtype=np.float32)
        regressor = np.asarray(source["J_regressor"], dtype=np.float32)
        parents = np.asarray(source["kintree_table"])[0].astype(np.int64)
    parents[0] = -1
    if (
        vertices.shape != (10475, 3)
        or regressor.shape != (55, 10475)
        or tuple(parents[:22]) != SMPL_PARENTS[:22]
    ):
        raise ValueError("expected the standard 55-joint SMPL-X body model")
    # compute_human_joints uses 0:22 and 39/54 with neutral finger rotations.
    # Each thumb chain therefore collapses exactly to its wrist.
    for endpoint, wrist in ((39, 20), (54, 21)):
        joint = endpoint
        while joint > wrist:
            parent = int(parents[joint])
            if not 0 <= parent < joint:
                raise ValueError("invalid SMPL-X thumb chain")
            joint = parent
        if joint != wrist:
            raise ValueError("SMPL-X thumb endpoint does not descend from its wrist")
    template = (regressor @ vertices)[[*range(22), 39, 54]]
    if not np.isfinite(template).all():
        raise ValueError("nonfinite converted human geometry")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(
            stream,
            format=SMPL_GEOMETRY_FORMAT,
            v_template=template,
            J_regressor=np.eye(24, dtype=np.float32),
            shapedirs=np.empty((24, 3, 0), dtype=np.float32),
            kintree_table=np.asarray((SMPL_PARENTS, tuple(range(24)))),
            source_smplx_sha256=file_sha256(smplx),
            geometry="SONIC compute_human_joints: neutral SMPL-X body 0:22 and thumb endpoints 39/54; neutral fingers",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smplx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare_model(args.smplx, args.output)


if __name__ == "__main__":
    main()
