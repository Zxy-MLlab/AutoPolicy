# Data layout

- `robotwin_clean_50/`: 50 LeRobot v2.1 task datasets, 2,500 episodes and 551,931 frames. The tree was
  imported from the pre-existing workspace dataset with hard links, so it is directly addressable inside
  AutoPolicy without consuming another 38 GB of data blocks. The files remain valid if the original names are
  removed because each path is an independent hard link.
- `real/`: reserved for synchronized RGB/RGB-D/video and robot state inputs for new reconstruction runs.
- `generated/`: reserved for trajectory and converted dataset output.

Run `scripts/autopolicy.sh audit-dataset data/robotwin_clean_50` after any dataset change.
