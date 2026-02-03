# Phase 1: Environment Setup & Vision

**Goal**: Set up a fresh Ubuntu machine for Deep Learning and Robotics, and establish a robust vision pipeline (`HaMeR` + `RealSense`).

---

## Part A: Environment Setup
**Target System**: Ubuntu 24.04 + NVIDIA RTX 5090.

### 1. System Prerequisites

#### NVIDIA Driver & CUDA
Ensure the latest drivers supporting your architecture (e.g., Blackwell/Ada) are installed.
```bash
# Verify Driver (Target: 580.xx+) and CUDA (Target: 12.8+)
nvidia-smi
nvcc --version
```

#### System Utilities
Install build tools required for ROS 2 and Python extensions.
```bash
sudo apt update
sudo apt install -y build-essential gcc-11 g++-11 git curl python3-pip net-tools
```

### 2. Install ROS 2 Jazzy
We use **ROS 2 Jazzy** for low-latency communication.

```bash
# 1. Enable Universe Repo
sudo apt install software-properties-common
sudo add-apt-repository universe

# 2. Add ROS 2 GPG Key
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# 3. Add Repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# 4. Install ROS 2 Jazzy Desktop
sudo apt update
sudo apt install -y ros-jazzy-desktop python3-colcon-common-extensions python3-rosdep

# 5. Initialize rosdep
sudo rosdep init
rosdep update

# 6. Environment Setup
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
echo "export RMW_IMPLEMENTATION=rmw_fastrtps_cpp" >> ~/.bashrc
echo "export ROS_DOMAIN_ID=0" >> ~/.bashrc
source ~/.bashrc
```

#### Install RealSense Drivers
```bash
sudo apt install -y ros-jazzy-realsense2-camera ros-jazzy-realsense2-description
```

### 3. AI & Vision Environment (Conda)
We use a dedicated Conda environment to handle PyTorch and HaMeR dependencies without conflicting with ROS system packages.

```bash
# 1. Install Miniconda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc

# 2. Create 'isaac' Environment
conda create -n isaac python=3.10 -y
conda activate isaac

# 2. Key Libraries
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install mediapipe pyrealsense2 opencv-python numpy scipy
pip install pinocchio
pip install dex-retargeting --no-deps 
# Note: dex-retargeting is installed for util functions, but we use custom logic.

# Verify usage
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 4. Install HaMeR (Hand Mesh Recovery)
HaMeR is used for 3D Keypoint estimation.

```bash
# 1. Clone Dependencies
cd ~/workspace/ros2_ws/src
git clone https://github.com/geopavlakos/hamer.git

# 2. Install HaMeR
cd hamer
pip install -e .
pip install webdataset hydra-core pyrootutils rich smplx==0.1.28 chumpy

# 3. Download Models
mkdir -p _DATA/data/mano
# Download HaMeR Checkpoints
wget https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz
tar -xvf hamer_demo_data.tar.gz
# Download MANO Hand Model
wget -O _DATA/data/mano/MANO_RIGHT.pkl https://huggingface.co/camenduru/HandRefiner/resolve/main/MANO_RIGHT.pkl

# 4. Link Data to DexTel
cd ~/workspace/ros2_ws/src/dextel/dextel
ln -s ../../hamer/_DATA _DATA
```

### 5. Install Isaac Sim 5.0
```bash
pip install isaacsim==5.0.0 --extra-index-url https://pypi.nvidia.com
```

---

## Part B: Vision Pipeline Implementation
**Script**: `dextel/ur3_realsense_hamer.py`

Once the environment is set up, we implement the `RobustTracker` class.

### 1. Key Logic
-   **MediaPipe**: Used for 2D ROI finding (fast).
-   **HaMeR**: Run inference on the ROI to get 3D Mesh.
-   **Depth Fusion**: Use `RealSense` Depth frame to calculate the absolute Z-distance of the wrist.
-   **Smoothing**: Apply `OneEuroFilter` to `(x,y,z)` position and `quaternion` rotation to remove jitter.

### 2. Verification
Test if the vision system is working standalone.

```bash
# 1. Activate Environment
conda activate isaac
source /opt/ros/jazzy/setup.bash

# 2. Run Vision Script
python3 -m dextel.ur3_realsense_hamer
```