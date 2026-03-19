# Setup Guide

## Script Description

- **replay_allegro_grasp_isaacsim.py**: Main script. Runs the Allegro Hand grasping simulation in Isaac Sim and retargets motion to the Shadow Hand in real time. The first 120 frames are throttled to approximately 12 seconds, and fingertip trajectories are written to `fingertip_coords.csv`.
- **plot_fingertip_trajectories.py**: Plotting script. Reads `fingertip_coords.csv`, plots the fingertip trajectory curves for Allegro (4 fingers) and Shadow (5 fingers) in world coordinates, and saves the result as `fingertip_trajectories.png`.

## Dependencies Required by the Main Script

- **NVIDIA Isaac Sim**
  The main script must be run under Isaac Sim's bundled Python environment, which provides the `isaacsim`, `omni.*`, and `pxr` (USD) modules. Please install the version of Isaac Sim that matches the script.

- **Python packages (install inside Isaac Sim's Python environment)**
  If not already bundled with Isaac Sim, install the following packages using Isaac Sim's Python:
  - `numpy`
  - `PyYAML` (`pip install PyYAML`)
  - `anytree` (the main script will attempt to auto-install it if missing: `pip install anytree>=2.12.0`)
  - `pin` (used by dex_retargeting for forward kinematics; must be compatible with the current Python version)
  - `pytransform3d` (used by dex_retargeting)

  Example (using Isaac Sim's Python executable path):
  ```bash
  # Run pip via Isaac Sim's python — adjust the path to match your installation:
  <Isaac_Sim_install_dir>/python.sh -m pip install numpy PyYAML anytree pin pytransform3d
  ```

- **Repository contents**
  - The `src` directory (containing the `dex_retargeting` package) must be on the Python path. The main script adds it via `sys.path`.
  - `assets/robots/robots/hands/` must contain the Allegro and Shadow URDF model files.
  - `src/dex_retargeting/configs/offline/` must contain `allegro_hand_right.yml` and `shadow_hand_right.yml`.
  - Grasp data: placed in the same directory as the main script, or specified via `--yaml`. Defaults to `003.yaml` (must contain a `grasp_1` entry with `cspace_position`).

- **Optional: Shadow Hand model**
  To display the Shadow Hand in the scene, provide a Shadow USD file (e.g. in a `ShadowHand/` directory) or a usable Shadow URDF (e.g. under `assets/robots/robots/hands/shadow_hand/`). Without it, only the Allegro Hand will be displayed, but retargeting computation will still run.

## How to Launch the Main Script (replay_allegro_grasp_isaacsim.py)

Run under **Isaac Sim's bundled Python environment** (do not use the system Python), otherwise Isaac Sim modules will be missing.

1. **From the Isaac Sim installation directory**, use its provided `python.bat` (Windows) or `python.sh` (Linux) to run the script:

   ```bash
   # Windows (run from Isaac Sim's python directory or after adding it to PATH)
   python.bat replay_allegro_grasp_isaacsim.py

   # Linux
   ./python.sh replay_allegro_grasp_isaacsim.py
   ```

2. Alternatively, open and run `replay_allegro_grasp_isaacsim.py` in the **Isaac Sim Script Editor**.

3. Before running, set the working directory to the repository root (the directory containing `replay_allegro_grasp_isaacsim.py`), or ensure the script can correctly resolve paths to `assets`, `src`, and the grasp YAML file (e.g. `003.yaml`).

After launch, the simulation does not start automatically. Click the **Play** button in the Isaac Sim UI to begin.
