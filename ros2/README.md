# Isaac Sim ↔ Nav2 Bridge (Split Decision evader)

This brings up Nav2 in WSL2 and pipes it into the evader through Isaac Sim's
ROS 2 bridge. Defender stays on the local heading-P controller — Nav2 only
handles the evader's go-to-point requests.

```
Windows host                              WSL2 Ubuntu 22.04
─────────────────                         ──────────────────────────────
Isaac Sim 5.x                             ROS 2 Humble + Nav2
  isaacsim.ros2.bridge ext                  map_server (empty_arena)
  /evader/odom + /tf  ──── DDS ─────►       AMCL
  /evader/cmd_vel    ◄──── DDS ─────        controller_server (DWB)
                                            planner_server (NavFn)
  GoalProxyClient ─── TCP :5005 ───►        nav2_goal_proxy.py
                                              ↓ NavigateToPose action
                                            bt_navigator
```

## One-time setup (WSL2)

```bash
# inside WSL2 Ubuntu 22.04
cd /mnt/c/Users/Wayne/Desktop/GMU/PhD/ToM/IsaacToM
bash ros2/install/install_humble_nav2.sh
exec bash       # reload PATH + ROS env

# generate the empty 24x24m occupancy grid
pip install --user numpy pillow
python3 ros2/install/generate_empty_map.py
```

## Per-session runbook

**Terminal 1 — WSL2: Nav2 + goal proxy**
```bash
cd /mnt/c/Users/Wayne/Desktop/GMU/PhD/ToM/IsaacToM
ros2 launch ros2/launch/split_decision_nav2.launch.py
```
You should see `nav2_goal_proxy: goal proxy listening on 0.0.0.0:5005` and
the Nav2 lifecycle nodes activating.

**Terminal 2 — Windows: Isaac Sim with Nav2 mode**
```powershell
conda activate env_isaacsim
cd C:\Users\Wayne\Desktop\GMU\PhD\ToM\IsaacToM
python -m isaac_env.runner -c config/split_decision.yaml --no-vlm --use-nav2
```

Then type goto commands into Isaac Sim's stdin. With `--use-nav2`, evader
commands go to Nav2; defender commands keep using heading-P:
```
evader 0 5         # body-frame target -> world -> Nav2 NavigateToPose
evader B           # goal B -> Nav2
defender -3 4      # defender still uses heading-P (no Nav2 for it)
evader auto        # cancels current Nav2 goal, releases evader to strategy
```

## DDS / network notes

- WSL2's mirrored networking mode (Win11) puts WSL on the same loopback as
  Windows, so DDS over `127.0.0.1` works without extra config. If you're on
  legacy NAT mode, set `WSL_INTEROP_HOST` and use the Windows host's WSL IP.
- We force `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` and `ROS_DOMAIN_ID=42` on
  the WSL side. Set the same on the Windows side BEFORE launching Isaac Sim
  if you have other ROS 2 traffic on the network:
  ```powershell
  $env:RMW_IMPLEMENTATION = "rmw_fastrtps_cpp"
  $env:ROS_DOMAIN_ID = "42"
  ```

## Known caveats / first-run checks

- `isaacsim.ros2.bridge` is the **5.x** extension name. On Isaac Sim 4.x,
  edit `isaac_env/ros2_bridge.py` and change `_BRIDGE_EXT` to
  `omni.isaac.ros2_bridge`. The OmniGraph node names also lose the `isaacsim.`
  namespace on 4.x.
- The DifferentialController in the OmniGraph uses JetBot wheel parameters
  (radius 0.0325 m, wheelbase 0.1185 m). If you swap to Carter, update the
  numbers in `build_evader_graph()`.
- `--use-nav2` skips the local heading-P controller for the evader entirely.
  If Nav2 isn't producing cmd_vel (lifecycle not active, AMCL not localized,
  goal rejected), the evader will sit still — that's expected, not a bug.
- Empty costmap = no obstacle clearance. This is correct for V1 of the
  arena (progress.md says obstacles arrive in V2). When you add obstacles,
  publish a Lidar via Isaac Sim's `IsaacReadLidarPointCloud` + a
  `ROS2PublishLaserScan` node into `/evader/scan` — the Nav2 params are
  already wired to consume that topic.
- Nav2 does its own goal-tolerance check (0.25m); the local heading-P stop
  band (0.3m) is irrelevant when `--use-nav2` is set.
