"""Render a saved barbell diagnostic trace; never runs policies or hardware."""

import argparse
import json
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from OpenGL import GL


def render(run: Path) -> None:
    model = mujoco.MjModel.from_binary_path(str(run / "scene.mjb"))
    data = mujoco.MjData(model)
    trace = np.load(run / "cycle_trace.npz")
    rows = json.loads((run / "cycle_trace.json").read_text())
    result = json.loads((run / "result.json").read_text())
    model.vis.headlight.ambient[:] = 0.55
    model.vis.headlight.diffuse[:] = 0.8
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 480
    cameras = []
    for azimuth in (135, 90):
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.lookat[:] = [0, 0, 0.7]
        camera.distance = 3.2
        camera.azimuth = azimuth
        camera.elevation = -12
        cameras.append(camera)
    font = ImageFont.load_default(size=20)
    small = ImageFont.load_default(size=16)
    writer = imageio_ffmpeg.write_frames(
        str(run / "cycle.mp4"), (1280, 608), fps=1 / (float(trace["dt"]) * 2)
    )
    writer.send(None)
    snapshots = {}
    highest_bar = -np.inf
    renderer_name = ""
    try:
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            renderer_name = GL.glGetString(GL.GL_RENDERER).decode()
            for index in range(0, len(rows), 2):
                row = rows[index]
                data.qpos[:] = trace["qpos"][index]
                data.qvel[:] = trace["qvel"][index]
                data.ctrl[:] = trace["ctrl"][index]
                data.eq_active[:] = 0
                mujoco.mj_forward(model, data)
                frames = []
                for camera in cameras:
                    camera.lookat[:2] = data.qpos[:2]
                    renderer.update_scene(data, camera=camera)
                    frames.append(renderer.render().copy())
                canvas = Image.new("RGB", (1280, 608), "#f7f7f7")
                canvas.paste(Image.fromarray(np.concatenate(frames, axis=1)), (0, 128))
                draw = ImageDraw.Draw(canvas)
                draw.text(
                    (16, 8),
                    f"BARBELL MODE | {row['time_s']:.2f}s | {row['phase'].replace('_', ' ')}",
                    font=font,
                    fill="#17212b",
                )
                draw.text(
                    (16, 38),
                    f"Standing v2 + trained motion | {result['bar_mass_kg']:g} kg | Robot load {row['load_fraction'] * 100:.0f}% | Motion blend {row['motion_weight'] * 100:.0f}%",
                    font=small,
                    fill="#17212b",
                )
                hand = "CLOSE" if row["finger_closure"] else "OPEN"
                wait = row["awaiting"] or "none"
                draw.text(
                    (16, 64),
                    f"Finger target: {hand} | Confirmation: {wait} | Bar/body contact: {row['barbell/unintended_contact_force_n']:.1f} N",
                    font=small,
                    fill="#205040",
                )
                draw.text(
                    (16, 90),
                    "SIMULATION: real hand gRPC; simulated finger feedback, ideal grips and external holder",
                    font=small,
                    fill="#586674",
                )
                writer.send(np.asarray(canvas))
                bar_height = data.xpos[model.body("barbell").id, 2]
                if row["phase"] != "motion" or bar_height > highest_bar:
                    snapshots[row["phase"]] = canvas.copy()
                    if row["phase"] == "motion":
                        highest_bar = bar_height
            draw.rectangle((0, 0, 1280, 128), fill="#f7f7f7")
            label = (
                "COMPLETE - all handoff gates passed"
                if result["completed"]
                else f"STOP: {result['stop_reason']}"
            )
            draw.text((16, 32), label, font=font, fill="#205040")
            for _ in range(50):
                writer.send(np.asarray(canvas))
            canvas.save(run / "last-frame.png")
    finally:
        writer.close()
    phases = [
        "hands_forward",
        "close_hands",
        "motion",
        "await_support",
        "open_hands",
        "normal_standing_final",
    ]
    sheet = Image.new("RGB", (1280, 912), "white")
    for index, phase in enumerate(phases):
        if phase in snapshots:
            sheet.paste(
                snapshots[phase].resize((640, 304)),
                ((index % 2) * 640, (index // 2) * 304),
            )
    sheet.save(run / "contact-sheet.png")
    (run / "render.json").write_text(
        json.dumps(
            {
                "renderer": renderer_name,
                "frames": len(range(0, len(rows), 2)) + 50,
                "trace_dt": float(trace["dt"]),
                "video": "cycle.mp4",
            },
            indent=2,
        )
    )
    print(run / "cycle.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    render(parser.parse_args().run)
