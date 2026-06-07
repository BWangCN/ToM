#!/usr/bin/env bash
# Run inside WSL2 Ubuntu 22.04. Installs ROS 2 Humble + Nav2 + tools we need.
set -euo pipefail

if [[ "$(lsb_release -cs)" != "jammy" ]]; then
    echo "ERROR: this script targets Ubuntu 22.04 (jammy). Detected $(lsb_release -cs)." >&2
    exit 1
fi

sudo apt update
sudo apt install -y software-properties-common curl gnupg lsb-release
sudo add-apt-repository -y universe

sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install -y \
    ros-humble-desktop \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-slam-toolbox \
    ros-humble-rmw-fastrtps-cpp \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-pip

if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
    sudo rosdep init
fi
rosdep update

# Make ROS auto-source for this user.
if ! grep -q "/opt/ros/humble/setup.bash" "$HOME/.bashrc"; then
    echo 'source /opt/ros/humble/setup.bash' >> "$HOME/.bashrc"
fi

# DDS settings: WSL2's NAT confuses default cyclone discovery. FastDDS over
# localhost is the simplest path that talks to Isaac Sim on the Windows host.
if ! grep -q "RMW_IMPLEMENTATION" "$HOME/.bashrc"; then
    cat >> "$HOME/.bashrc" <<'EOF'

# Isaac Sim ↔ WSL2 ROS 2 bridge
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=42
EOF
fi

echo
echo "Done. Open a new shell, then run:"
echo "  ros2 doctor    # should report no errors"
echo "  ros2 launch nav2_bringup tb3_simulation_launch.py headless:=True   # smoke test"
