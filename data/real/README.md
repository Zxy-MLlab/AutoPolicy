# Real-scene inputs

Put each new real-world scene in its own directory:

```text
data/real/<scene_id>/
  videos/
    camera_00.mp4
    camera_01.mp4
  metadata.json
  calibration/
    intrinsics.json
    extrinsics.json
  robot/
    joint_states.npz
    gripper_states.npz
```

For a single-camera test, this is the minimum layout:

```text
data/real/<scene_id>/
  videos/camera_00.mp4
  metadata.json
```

`metadata.json` should record the camera name, timestamp convention, FPS, resolution, task description, and
whether robot state/calibration files are available. A single MP4 is enough for a visual/geometry prototype,
but it is not enough to create an executable robot digital twin: robot joint states, camera intrinsics, camera
extrinsics, and a metric scale reference are needed for robot-camera alignment and physics replay.

Example:

```json
{
  "scene_id": "my_pick_place_001",
  "task": "pick the red block and place it in the tray",
  "videos": [{
    "path": "videos/camera_00.mp4",
    "camera_name": "head",
    "fps": 30,
    "timestamp_unit": "seconds"
  }],
  "robot": {
    "joint_states": null,
    "gripper_states": null
  },
  "calibration": {
    "intrinsics": null,
    "extrinsics": null
  }
}
```

Use an ASCII scene id and keep the original MP4 unchanged. Before processing, run:

```bash
ffprobe -v error -show_streams data/real/<scene_id>/videos/camera_00.mp4
```
