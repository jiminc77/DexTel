# Environment Guide (Ubuntu 24.04 + RTX 5090)

## 1. System Prerequisites

### NVIDIA Driver & CUDA

Ensure the latest drivers supporting the Blackwell architecture are installed.

```bash
# Verify Driver (Target: 580.xx+) and CUDA (Target: 12.8+)
nvidia-smi
nvcc --version
```

### System Utilities

Install basic build tools required for ROS and Python extensions.

```bash
sudo apt update
sudo apt install -y build-essential gcc-11 g++-11 git curl python3-pip
```

---

## 2. Install ROS 2 Jazzy

### Installation

```bash
# 1. Enable Ubuntu Universe repository
sudo apt install software-properties-common
sudo add-apt-repository universe

# 2. Add the ROS 2 GPG key
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# 3. Add the repository to sources list
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# 4. Install ROS 2 Jazzy Desktop (includes RViz2, demos)
sudo apt update
sudo apt install -y ros-jazzy-desktop
sudo apt install -y python3-colcon-common-extensions python3-rosdep

# 5. Initialize rosdep
sudo rosdep init
rosdep update

echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
echo "export RMW_IMPLEMENTATION=rmw_fastrtps_cpp" >> ~/.bashrc
echo "export ROS_DOMAIN_ID=0" >> ~/.bashrc
source ~/.bashrc
```

### Environment Setup

```bash
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Install Peripheral Packages (RealSense, etc.)

```bash
# Install RealSense wrappers for Jazzy
sudo apt install -y ros-jazzy-realsense2-camera ros-jazzy-realsense2-description
```

## 3. AI & Vision Environment (PyTorch Nightly)

Create a dedicated Conda environment for the Vision Pipeline to avoid conflicts with system Python.

```bash
# 1. Install Miniconda (if not installed)
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
source ~/.bashrc

# 2. Create Environment
conda create -n isaac python=3.11 -y
conda activate isaac

# 3. Install PyTorch Nightly (Required for RTX 5090 / Blackwell support)
# Use the nightly index for CUDA 12.8 compatibility
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 4. Verify Installation
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"

```

## 4. Isaac Sim 5.0 Installation

Install Isaac Sim within the Conda environment or global pip, ensuring GCC compatibility.

```bash
# Activate your env
conda activate isaac

# Install Isaac Sim 5.0
pip3 install --upgrade pip
pip3 install isaacsim==5.0.0 --extra-index-url https://pypi.nvidia.com

# First Run (Downloads assets and generates cache)
isaacsim
```

---

## 5. Workspace Setup

Set up your development workspace for `isaac`.

```bash
# 1. Create Workspace
mkdir -p ~/workspace/ros2_ws/src
cd ~/workspace/ros2_ws

# 2. Build (Empty workspace test)
colcon build --symlink-install

# 3. Source the workspace
echo "source ~/workspace/ros2_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## 6. Connection & Communication Test (Sim ↔ ROS 2)

### 1. Configuration Check

Ensure both terminals (Sim and ROS) share the same Domain ID.

```bash
export ROS_DOMAIN_ID=0
```

### 2. Configure Isaac Sim (Graph Setup)

1. Launch Isaac Sim: `isaacsim`
2. Enable ROS 2 Bridge:
    - Go to `Window` -> `Extensions`.
    - Search for `isaac.ros2_bridge`.
    - Ensure it is **Enabled**.
3. Create Action Graph (To publish camera data):
    - `Window` $\rightarrow$ `Graph Editors` $\rightarrow$ `Action Graph`.
    - Add the following nodes and connect them:
        
        ![image.png](attachment:da6c4f60-cf7c-42d7-bb82-0884efea01ea:image.png)
        
        - On Playback Tick $\rightarrow$ Isaac Create Render Product $\rightarrow$ ROS2 Camera Helper.
    - Create Camera:
        - `Create` $\rightarrow$`Camera`
    - Configure ROS2 Camera Helper:
        - `Topic Name`: `/camera/color/image_raw`
        - `Type`: `rgb`
    - Configure Isaac Create Render Prodcut:
        - cameraPrim: Camera
4. Add Cube
    - `Create` $\rightarrow$ `Mesh` $\rightarrow$ `Cube` (If you can’t see the cube, press F)
5. Press Play in Isaac Sim.

### 3. Verify in ROS 2 Jazzy (Terminal)

Open a new terminal and verify the topic.

```bash
# Check if the topic exists
ros2 topic list
# Should see: /camera/color/image_raw

# Check data frequency
ros2 topic hz /camera/color/image_raw
```

### 4. Visualize (RViz2)

Since you are on the Host, you can simply run RViz2.

```bash
rviz2
```

1. Click **Add** (bottom left).
2. Select **By Topic**.
3. Select `/camera/color/image_raw` $\rightarrow$ **Image**.
4. If you see the cube, the process was successful.