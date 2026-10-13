# Hand control API

`bello-run-hands arm=true calibration_confirmed=true` runs
`bello.hand.v1.HandControl` on `listen=127.0.0.1:50057`. It probes the enabled
Linker O6/L6 hands, then starts in `HOLD`. Missing CAN interfaces or hand replies
are retried while the API remains available. Startup no longer follows tracking
automatically. `arm=false` remains tracking preflight: no server and no CAN.

The body runner can own these leases automatically. Set `control.hand_tracking`
in its hardware YAML:

```yaml
control:
  hand_tracking:
    endpoint: http://127.0.0.1:50057
    rpc_timeout_seconds: 0.05
    renew_period_seconds: 0.2
```

With this configured, entering full-body `Tracking` selects hand `TRACKING`.
In `Locomotion`, enabling upper-body tracking selects hand `TRACKING`; turning
the toggle off releases both leases. Stand, Hold, Sit, prerecorded motions,
Barbell, controller loss, and abort suspend automatic hand tracking. Tracking's
return to Stand releases the hands when the return begins. Barbell keeps its
own exclusive application leases. The runner never steals another owner's
leases; partial acquisition is released, and failures are logged and retried
while tracking is requested.

Hand RPCs run independently of body commands. Leases are renewed at the
configured period, not the policy tick rate. Stale optical hand data suspends
writes in the hand server and resumes with fresh data and a measured-position
reseed. Stalled body ticks stop lease renewal and release the hands. A missing
or failed hand service suspends hand tracking without stopping body control.
Use `GetStatus` to inspect ownership, source, and suspension reasons. Set
`control.hand_tracking: null` for robots without this integration.

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
no automatic return to tracking after lease expiry or release.

Temporary CAN I/O failures and missing feedback suspend only the affected hand.
Failed sockets are closed and rebound to the configured interface, with retries
paced by `can_feedback_poll_interval_seconds`. No CAN links are reconfigured;
the arms sharing these buses remain under hardwared's control. Fresh feedback
clears the transport fault, including while in `HOLD`. A still-owned source
resumes after a measured-position probe and filter reseed; tracking additionally
requires a new optical sample after that probe. Outage time never accumulates
extra allowed movement. Lease expiry during an outage leaves the hand in `HOLD`;
recovery does not reacquire ownership or replay expired targets. Malformed
responses still latch the source in `HOLD` and require explicit source selection.

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
