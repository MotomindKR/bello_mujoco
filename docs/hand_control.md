# Hand control API

`bello-run-hands arm=true calibration_confirmed=true` runs
`bello.hand.v1.HandControl` on `listen=127.0.0.1:50057`. It probes the enabled
Linker O6/L6 hands, then starts in `HOLD`. Startup no longer follows tracking
automatically. `arm=false` remains tracking preflight: no server and no CAN.

The contract is [hand.proto](../src/bello_mujoco/hardware/hand.proto).
The Python client is `bello_mujoco.hardware.hand_server.HandControlClient`.
Other clients can generate bindings from the same schema. The listener is
unauthenticated and defaults to loopback; changing `listen` exposes hardware
control to clients that can reach that address.

## Ownership and commands

Each enabled hand has its own exclusive lease. `Acquire(Hand)` returns its
opaque token. All commands for that hand require this token; a second owner
is rejected. `SetCommand` and `KeepAlive` renew the lease for
`lease_timeout_seconds` (default 1 second). Renew well before expiry, for example
every 0.2 seconds with the default timeout. `Release` relinquishes ownership
and suspends writes. Expired tokens cannot renew or command; acquire again.

`SetCommand` selects one of these sources:

| Source | Behavior |
| --- | --- |
| `HOLD` | Suspend position writes; maintain the lease |
| `TRACKING` | Follow fresh XRoboToolkit samples while the lease remains live |
| `APPLICATION` | Approach the supplied target independently of tracking |

`APPLICATION` accepts exactly one of `open=true`, `close=true`, or six normalized
`positions.values`. Values must be finite and in `[0, 1]`, where 0 is the
calibrated open endpoint and 1 is the calibrated closed endpoint. Axis order is
`thumb_flex, thumb_abduction, index, middle, ring, pinky`. Custom targets serve
as grasp presets; a target does not confirm contact or a secured object.

`max_raw_rate_per_second` optionally lowers the software slew limit for that
command. It must be positive and cannot exceed the server configuration's cap.
Omitting it uses the configured cap. It is a position-command rate, not a
hardware motor-speed or force register. Smoothing and endpoint calibration
apply to both tracking and application targets. Source or rate changes reseed
from measured positions. Delays never accumulate more than one control period
of allowed movement. Ordinary target changes retain the filter state.

Lease expiry, release, `HOLD`, and process shutdown stop further position writes.
They do not open the hand, disable motor torque, or cancel a target already sent
to the device. The device may continue moving toward its last target. There is
no automatic return to tracking. A transport error latches the affected hand
in `HOLD`; inspect status and explicitly select a source again to retry.

Tracking loss suspends only hands currently using `TRACKING`. Recovery requires
a position probe followed by a fresh tracking sample. The server and application
control remain available when the bridge is missing. Both hands share one CAN
writer; gRPC requests change intent and never write to the bus themselves.
A successful `SetCommand` acknowledges accepted intent, not completed movement.

## Python client example

This example selects a partial grasp on the left hand for five seconds. Use
calibrated hardware before running it. Releasing the lease suspends writes;
it does not release the object. Request `open=true` explicitly when appropriate.

```python
import time
import grpc
from bello_mujoco.hardware import hand_pb2 as pb
from bello_mujoco.hardware.hand_server import HandControlClient

with grpc.insecure_channel("127.0.0.1:50057") as channel:
    client = HandControlClient(channel)
    lease = client.acquire(pb.LEFT)
    try:
        client.command(pb.Command(
            side=lease.side,
            token=lease.token,
            source=pb.APPLICATION,
            positions=pb.Positions(values=[0.5] * 6),
            max_raw_rate_per_second=100,
        ))
        until = time.monotonic() + 5
        while time.monotonic() < until:
            print(client.keep_alive(lease))
            time.sleep(0.2)
        # To select headset input under the same lease:
        # client.command(pb.Command(
        #     side=lease.side, token=lease.token, source=pb.TRACKING))
        # Continue renewing while tracking should control this hand.
    finally:
        client.release(lease)
```

## Status and limits

`GetStatus` requires no lease and performs no CAN I/O. It returns enabled sides,
selected source, ownership, remaining lease time, application target, last sent
raw positions, separately measured raw positions, feedback age/freshness,
tracking freshness, suspension reason, transport fault, and selected slew limit.
Read-only feedback polling continues while writes are suspended. Status never
returns another client's token. An expired lease is reported as `HOLD` even
before the next control tick.

`grasp_feedback_supported=false` and `force_control_supported=false` are
explicit capabilities. This service does not yet expose force/torque-limit
register writes, motor fault registers, contact sensing, or grasp detection.
`transport_fault` reports local CAN I/O/protocol failures. Measured position
agreement alone must not advance a barbell handoff to "secured".

Invalid commands return `INVALID_ARGUMENT`; absent/wrong/expired tokens return
`PERMISSION_DENIED`; acquiring an owned hand returns `FAILED_PRECONDITION`.
Invalid commands do not change the target or extend the lease. Server restarts
invalidate every lease and begin with writes suspended.
