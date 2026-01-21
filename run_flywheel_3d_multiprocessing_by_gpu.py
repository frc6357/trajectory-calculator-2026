"""
3D Flywheel Trajectory Calculator with Animation
This module simulates and animates a projectile launched from a moving robot
attempting to hit a target hub. It uses ballistic physics to calculate
optimal shooter angles and speeds, then visualizes the trajectory.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from multiprocessing import Pool, cpu_count
import cupy as cp



# ============================================================
# Physical constants
# ============================================================
# Gravitational acceleration in meters per second squared
G = 9.81  # m/s^2

# ============================================================
# HUB geometry (meters)
# ============================================================
# Hexagonal hub with circumscribed radius (center to vertex)
HUB_RADIUS = 1.06 / 2.0  # circumscribed radius
# Apothem is the distance from center to the middle of a side (for hexagon)
HUB_APOTHEM = HUB_RADIUS * np.cos(np.pi / 6)
# Height of the hub above the field floor
HUB_Z = 1.83  # rim height
# Safety buffer to avoid hitting hub edges
HUB_CLEARANCE = 0.0762  # buffer from hub edges

HUB_X = 4.0213
HUB_Y = 4.6116


# ============================================================
# Shooter constraints
# ============================================================
# Minimum and maximum pitch angles the shooter can achieve
PITCH_MIN = np.deg2rad(45)
PITCH_MAX = np.deg2rad(55)
# Minimum and maximum flywheel speeds in meters per second
FLYWHEEL_MIN_SPEED_M_PER_S = 1.0
FLYWHEEL_MAX_SPEED_M_PER_S = 10.0
# Array of possible flywheel speeds for search optimization
FLYWHEEL_SPEEDS = np.linspace(
    FLYWHEEL_MIN_SPEED_M_PER_S, FLYWHEEL_MAX_SPEED_M_PER_S, 80
)  # m/s
# Number of flywheel speed steps to test (used in some search variants)
FLYWHEEL_STEPS = 20

SHOOTER_OFFSET_R = np.array(
    [0.3, 0.0, 0.65]
)  # Shooter position relative to robot in robot frame (front-right, elevated)

# ============================================================
# Robot constraints
# ============================================================
ROBOT_POS_Z_F = 0.0
ROBOT_VEL_Z_F = 0.0

# ============================================================
# Multiprocessing settings
# ============================================================
NUM_PROCESSES = cpu_count()  # Use all available CPU cores
print("CUDA_PATH:", cp.cuda.get_cuda_path())
print("GPU TARGET:", cp.cuda.get_device_id()) # Make sure you don't try to use the CPU integrated graphics

# ============================================================
# Rotation utilities
# ============================================================
def rotz(yaw):
    """
    Create a 3D rotation matrix for rotation around the Z-axis (yaw).

    Parameters:
        yaw: Rotation angle in radians

    Returns:
        3x3 rotation matrix for yaw rotation in field frame
    """
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# ============================================================
# Ballistic model
# ============================================================
def projectile_position_field(p_field, v_field, t):
    """
    Calculate projectile position at time t under constant gravity.
    Uses kinematic equation: p(t) = p0 + v0*t - 0.5*g*t^2

    Parameters:
        p_field: Initial position [x, y, z] in field frame
        v_field: Initial velocity [vx, vy, vz] in field frame
        t: Time elapsed in seconds

    Returns:
        Position [x, y, z] at time t
    """
    return p_field + v_field * t + np.array([0, 0, -0.5 * G * t**2])


# ============================================================
# HUB intersection test
# ============================================================
def intersect_hub_plane(p_shooter_field, v_field, hub_center_field):
    """
    Check if a projectile trajectory intersects with the hub at height HUB_Z.
    Solves for the time when projectile reaches hub height, then validates
    the horizontal position is within hub boundaries.

    Parameters:
        p_shooter_field: Initial projectile position [x, y, z] in field frame
        v_field: Initial projectile velocity [vx, vy, vz] in field frame
        hub_center_field: Hub center position [x, y, z] in field frame

    Returns:
        (hit_valid, time_to_hit): Tuple of (bool indicating valid hit, time value or None)
    """
    # Solve quadratic equation for intersection with plane z = HUB_Z
    # Using: z(t) = z0 + vz*t - 0.5*g*t^2 = HUB_Z
    a = -0.5 * G
    b = v_field[2]
    c = p_shooter_field[2] - HUB_Z
    disc = b * b - 4 * a * c
    if disc < 0:
        return False, None

    # Take the earlier intersection time (projectile ascending or transitioning)
    t_hit = (-b - np.sqrt(disc)) / (2 * a)
    if t_hit <= 0:
        return False, None

    # Verify projectile is descending at impact (not ascending)
    if v_field[2] - G * t_hit >= 0:
        return False, None

    # Calculate impact point
    p_hit = projectile_position_field(p_shooter_field, v_field, t_hit)

    # Check if impact point is within hexagonal hub boundaries
    d = p_hit[:2] - hub_center_field[:2]
    x, y = abs(d[0]), abs(d[1])
    if (
        x > HUB_RADIUS - HUB_CLEARANCE
        or y > HUB_APOTHEM - HUB_CLEARANCE
        or (x * np.cos(np.pi / 6) + y * np.sin(np.pi / 6) > HUB_RADIUS - HUB_CLEARANCE)
    ):
        return False, None

    return True, t_hit


# ============================================================
# Shot candidate
# ============================================================
class ShotCandidate:
    """
    Represents a single shot configuration with shooter parameters and results.
    Stores yaw, pitch, flywheel speed, impact time, error metrics, and validation flags.
    """

    def __init__(
        self, yaw_rad, pitch_rad, flywheel_speed, time_hit, lateral_error, descending
    ):
        """
        Initialize a shot candidate.

        Parameters:
            yaw_rad: Shooter yaw angle in radians (field frame)
            pitch_rad: Shooter pitch angle in radians
            flywheel_speed: Flywheel speed in m/s
            time_hit: Time to reach hub height in seconds
            lateral_error: Distance from impact point to hub center in meters
            descending: Boolean indicating if projectile is descending at impact
        """
        self.yaw_rad = yaw_rad
        self.pitch_rad = pitch_rad
        self.flywheel_speed = flywheel_speed
        self.time_hit = time_hit
        self.lateral_error = lateral_error
        self.descending = descending

    @property
    def cost(self):
        """
        Calculate a cost metric for shot quality.
        Lower cost is better. Lateral error is primary metric;
        non-descending shots get a penalty.

        Returns:
            Cost value for this shot candidate
        """
        cost = self.lateral_error
        if not self.descending:
            cost += 1.0

        med_fw_speed = (FLYWHEEL_MIN_SPEED_M_PER_S + FLYWHEEL_MAX_SPEED_M_PER_S) / 2
        # Optional: Add small penalty for flywheel speeds far from median to encourage moderate shots
        cost += np.sqrt(abs(self.flywheel_speed - med_fw_speed) / med_fw_speed)
        return cost


# ============================================================
# Convergent solver
# ============================================================
# ============================================================
# Convergent solver
# ============================================================
def solve_shot_planar(
    robot_pos_F,
    robot_vel_F,
    shooter_offset_R,
    robot_yaw_F,
    hub_center_F,
    max_iterations=300,
    tol_lateral=0.01,
):
    """
    Iterative solver for optimal shooter configuration using GPU acceleration with CuPy.
    Searches parameter space (yaw, pitch, speed) and iteratively refines around best candidate.

    All quantities are expressed in the FIELD frame unless suffixed with _R (robot frame).

    Parameters:
        robot_pos_F: Robot position [x, y, z] in field frame
        robot_vel_F: Robot velocity [vx, vy, vz] in field frame
        shooter_offset_R: Shooter position [x, y, z] relative to robot in robot frame
        robot_yaw_F: Robot yaw angle in radians (field frame)
        hub_center_F: Hub center position [x, y, z] in field frame
        max_iterations: Maximum refinement iterations
        tol_lateral: Tolerance for lateral error in meters

    Returns:
        ShotCandidate: Best shot configuration found, or None if no valid shot
    """
        
    # Convert inputs to GPU arrays
    robot_pos_F_gpu = cp.asarray(robot_pos_F, dtype=cp.float32)
    robot_vel_F_gpu = cp.asarray(robot_vel_F, dtype=cp.float32)
    shooter_offset_R_gpu = cp.asarray(shooter_offset_R, dtype=cp.float32)
    hub_center_F_gpu = cp.asarray(hub_center_F, dtype=cp.float32)

    # Step 1: Calculate shooter position in field frame on GPU
    c, s = np.cos(robot_yaw_F), np.sin(robot_yaw_F)
    R_FR = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    shooter_pos_F = robot_pos_F + R_FR @ shooter_offset_R
    shooter_pos_F_gpu = cp.asarray(shooter_pos_F, dtype=cp.float32)

    best = None

    # Step 2: Coarse initial search centered on hub bearing
    hub_bearing_F = float(
        cp.arctan2(
            hub_center_F_gpu[1] - shooter_pos_F_gpu[1],
            hub_center_F_gpu[0] - shooter_pos_F_gpu[0],
        )
    )
    initial_yaw_rel = hub_bearing_F - robot_yaw_F

    yaw_values = cp.linspace(
        initial_yaw_rel - np.deg2rad(30), initial_yaw_rel + np.deg2rad(30), 100
    )
    pitch_values = cp.linspace(PITCH_MIN, PITCH_MAX, 50)
    speed_values = cp.linspace(
        FLYWHEEL_MIN_SPEED_M_PER_S, FLYWHEEL_MAX_SPEED_M_PER_S, 50
    )

    # Step 3: Iterative refinement loop
    for _ in range(max_iterations):
        improved = False

        # Vectorized computation for all combinations
        yaw_grid, pitch_grid, speed_grid = cp.meshgrid(
            yaw_values, pitch_values, speed_values, indexing="ij"
        )

        # Flatten grids for batch processing
        yaw_flat = yaw_grid.flatten()
        pitch_flat = pitch_grid.flatten()
        speed_flat = speed_grid.flatten()

        # Calculate shooter yaw in field frame (vectorized)
        shooter_yaw_F_gpu = robot_yaw_F + yaw_flat

        # Vectorized aim direction
        aim_xy_F_gpu = cp.stack(
            [cp.cos(shooter_yaw_F_gpu), cp.sin(shooter_yaw_F_gpu)], axis=1
        )

        # Vectorized pitch calculations
        cos_p_gpu = cp.cos(pitch_flat)
        sin_p_gpu = cp.sin(pitch_flat)

        # Vectorized velocity calculations
        v0_x = (
            robot_vel_F_gpu[0]
            + speed_flat * cos_p_gpu * aim_xy_F_gpu[:, 0]
        )
        v0_y = (
            robot_vel_F_gpu[1]
            + speed_flat * cos_p_gpu * aim_xy_F_gpu[:, 1]
        )
        v0_z = speed_flat * sin_p_gpu

        # Solve quadratic for plane intersection (vectorized)
        a_gpu = cp.full_like(v0_z, -0.5 * G)
        b_gpu = v0_z
        c_gpu = cp.full_like(v0_z, shooter_pos_F_gpu[2] - HUB_Z)

        disc_gpu = b_gpu * b_gpu - 4 * a_gpu * c_gpu

        # Valid candidates have non-negative discriminant
        valid_mask = disc_gpu >= 0

        # Calculate intersection times
        t_hit_gpu = cp.full_like(disc_gpu, -1.0, dtype=cp.float32)
        t_hit_gpu[valid_mask] = (
            (-b_gpu[valid_mask] - cp.sqrt(disc_gpu[valid_mask]))
            / (2 * a_gpu[valid_mask])
        )

        # Filter: t > 0
        valid_mask &= t_hit_gpu > 0

        # Filter: projectile must be descending
        vz_at_impact = v0_z - G * t_hit_gpu
        valid_mask &= vz_at_impact < 0

        # Calculate impact points (vectorized)
        p_hit_x = shooter_pos_F_gpu[0] + v0_x * t_hit_gpu + 0
        p_hit_y = shooter_pos_F_gpu[1] + v0_y * t_hit_gpu + 0
        p_hit_z = (
            shooter_pos_F_gpu[2]
            + v0_z * t_hit_gpu
            - 0.5 * G * t_hit_gpu**2
        )

        # Calculate lateral errors (vectorized)
        lateral_errors = cp.sqrt(
            (p_hit_x - hub_center_F_gpu[0]) ** 2
            + (p_hit_y - hub_center_F_gpu[1]) ** 2
        )

        # Zero out invalid candidates
        lateral_errors[~valid_mask] = 1e9

        # Find best candidate on GPU
        best_idx = cp.argmin(lateral_errors)
        best_lateral_error = float(lateral_errors[best_idx])

        # Only process if valid
        if best_lateral_error < 1e9:
            candidate = ShotCandidate(
                yaw_rad=float(shooter_yaw_F_gpu[best_idx]),
                pitch_rad=float(pitch_flat[best_idx]),
                flywheel_speed=float(speed_flat[best_idx]),
                time_hit=float(t_hit_gpu[best_idx]),
                lateral_error=best_lateral_error,
                descending=True,
            )

            if best is None or candidate.cost < best.cost:
                best = candidate
                improved = True

        # Step 4: Check convergence
        if best and best.lateral_error <= tol_lateral:
            break

        if not improved:
            break

        # Step 5: Refine search grid on GPU
        best_yaw_rel = best.yaw_rad - robot_yaw_F
        yaw_values = cp.linspace(
            best_yaw_rel - np.deg2rad(5), best_yaw_rel + np.deg2rad(5), 12
        )
        pitch_values = cp.linspace(
            max(PITCH_MIN, best.pitch_rad - 0.05),
            min(PITCH_MAX, best.pitch_rad + 0.05),
            12,
        )
        speed_values = cp.linspace(
            max(FLYWHEEL_MIN_SPEED_M_PER_S, best.flywheel_speed - 0.5),
            min(FLYWHEEL_MAX_SPEED_M_PER_S, best.flywheel_speed + 0.5),
            12,
        )

    return best


# ============================================================
# Animation
# ============================================================
# ============================================================
# Animation
# ============================================================
def animate_candidate(
    candidate, robot_pos_F, robot_yaw_F, robot_vel_F, shooter_offset_R, hub_center_F
):
    """
    Create and display an animation of the projectile and robot motion.
    Shows side view (X-Z) and top view (X-Y) with real-time trajectory updates.

    All quantities are expressed in the FIELD frame unless suffixed with _R (robot frame).

    Parameters:
        candidate: ShotCandidate object with optimal shot parameters
        robot_pos_F: Robot position [x, y, z] in field frame
        robot_yaw_F: Robot yaw angle in radians (field frame)
        robot_vel_F: Robot velocity [vx, vy, vz] in field frame
        shooter_offset_R: Shooter position relative to robot in robot frame
        hub_center_F: Hub center position [x, y, z] in field frame

    Returns:
        FuncAnimation object for the animation
    """

    # ============================================================
    # Step 1: Calculate shooter position in field frame
    # ============================================================
    c, s = np.cos(robot_yaw_F), np.sin(robot_yaw_F)
    R_FR = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    shooter_pos_F0 = robot_pos_F + R_FR @ shooter_offset_R

    # ============================================================
    # Step 2: Calculate initial projectile velocity in field frame
    # ============================================================
    cos_p = np.cos(candidate.pitch_rad)
    sin_p = np.sin(candidate.pitch_rad)

    # Horizontal aim direction
    aim_xy_F = np.array([np.cos(candidate.yaw_rad), np.sin(candidate.yaw_rad)])

    # Initial velocity combines robot motion and flywheel launch
    v0_F = np.array(
        [
            robot_vel_F[0] + candidate.flywheel_speed * cos_p * aim_xy_F[0],
            robot_vel_F[1] + candidate.flywheel_speed * cos_p * aim_xy_F[1],
            candidate.flywheel_speed * sin_p,
        ]
    )

    # ============================================================
    # Step 3: Set up time array for animation
    # ============================================================
    # Calculate time to apex of trajectory
    t_apex = max(0.0, v0_F[2] / G)
    # Extend animation time beyond impact
    t_end = t_apex * 2.0 + 1.0
    # Create smooth time array with 300 frames
    ts = np.linspace(0.0, t_end, 300)

    # ============================================================
    # Step 4: Calculate trajectories for all time points
    # ============================================================
    # Projectile follows ballistic trajectory
    projectile_traj_F = np.array(
        [shooter_pos_F0 + v0_F * t + np.array([0, 0, -0.5 * G * t * t]) for t in ts]
    )

    # Robot moves in straight line at constant velocity
    robot_traj_F = np.array([robot_pos_F + robot_vel_F * t for t in ts])

    # ============================================================
    # Step 5: Create figure with two subplots
    # ============================================================
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    side_ax, top_ax = axs

    # ============================================================
    # Step 6: Configure side view (X-Z plane)
    # ============================================================
    side_ax.set_title("Side View")
    side_ax.set_xlabel("Field X (m)")
    side_ax.set_ylabel("Height Z (m)")
    side_ax.grid(True)

    # Draw hub height line
    side_ax.axhline(HUB_Z, linestyle="--", color="k")
    # Draw hub as a rectangle in side view
    side_ax.add_patch(
        plt.Rectangle(
            (hub_center_F[0] - HUB_RADIUS, HUB_Z - 0.01),
            2 * HUB_RADIUS,
            0.02,
            color="gray",
            alpha=0.3,
        )
    )

    # ============================================================
    # Step 7: Configure top view (X-Y plane)
    # ============================================================
    top_ax.set_title("Top View")
    top_ax.set_xlabel("Field X (m)")
    top_ax.set_ylabel("Field Y (m)")
    top_ax.set_aspect("equal")
    top_ax.grid(True)

    # Draw hub as a circle in top view
    angles = np.linspace(0, 2 * np.pi, 7)
    top_ax.plot(
        hub_center_F[0] + HUB_RADIUS * np.cos(angles),
        hub_center_F[1] + HUB_RADIUS * np.sin(angles),
        linestyle="--",
        color="k",
    )

    # ============================================================
    # Step 8: Set axis limits based on trajectory extents
    # ============================================================
    all_x = np.concatenate([projectile_traj_F[:, 0], robot_traj_F[:, 0]])
    all_y = np.concatenate([projectile_traj_F[:, 1], robot_traj_F[:, 1]])
    all_z = projectile_traj_F[:, 2]

    side_ax.set_xlim(all_x.min() - 0.5, all_x.max() + 0.5)
    side_ax.set_ylim(0.0, max(all_z.max(), HUB_Z) + 0.5)

    top_ax.set_xlim(all_x.min() - 0.5, all_x.max() + 0.5)
    top_ax.set_ylim(all_y.min() - 0.5, all_y.max() + 0.5)

    # ============================================================
    # Step 9: Create animated plot objects
    # ============================================================
    # Side view artists
    (proj_dot_side,) = side_ax.plot([], [], "ro", label="Projectile")
    (robot_dot_side,) = side_ax.plot([], [], "bo", label="Robot")
    (traj_line_side,) = side_ax.plot([], [], "r--")

    # Top view artists
    (proj_dot_top,) = top_ax.plot([], [], "ro", label="Projectile")
    (robot_dot_top,) = top_ax.plot([], [], "bo", label="Robot")
    (traj_line_top,) = top_ax.plot([], [], "r--")

    # Add legends
    side_ax.legend()
    top_ax.legend()

    # ============================================================
    # Step 10: Define animation update function
    # ============================================================
    def update(i):
        """
        Update function called for each animation frame.

        Parameters:
            i: Frame index (0 to len(ts)-1)

        Returns:
            Tuple of updated artist objects for blitting
        """
        # Update side view (X-Z plane)
        proj_dot_side.set_data([projectile_traj_F[i, 0]], [projectile_traj_F[i, 2]])
        robot_dot_side.set_data([robot_traj_F[i, 0]], [robot_traj_F[i, 2]])
        # Draw trajectory line up to current point
        traj_line_side.set_data(projectile_traj_F[:i, 0], projectile_traj_F[:i, 2])

        # Update top view (X-Y plane)
        proj_dot_top.set_data([projectile_traj_F[i, 0]], [projectile_traj_F[i, 1]])
        robot_dot_top.set_data([robot_traj_F[i, 0]], [robot_traj_F[i, 1]])
        # Draw trajectory line up to current point
        traj_line_top.set_data(projectile_traj_F[:i, 0], projectile_traj_F[:i, 1])

        return (
            proj_dot_side,
            robot_dot_side,
            traj_line_side,
            proj_dot_top,
            robot_dot_top,
            traj_line_top,
        )

    # ============================================================
    # Step 11: Create animation object
    # ============================================================
    ani = FuncAnimation(
        fig, update, frames=len(ts), interval=50, blit=True  # 50ms per frame = 20 FPS
    )

    # Display the animation
    plt.tight_layout()
    plt.show()
    return ani


# ============================================================
# Simulation example
# ============================================================
def simulate():
    """
    Main simulation function demonstrating the trajectory calculation
    and animation system with example parameters.
    """
    # Create parameter ranges for scenario simulation
    robot_x_range = np.linspace(1.0, 4.0, 3)
    robot_y_range = np.linspace(0, 8, 3)
    robot_yaw_range = np.linspace(-np.deg2rad(60), np.deg2rad(60), 3)
    robot_vel_x_range = np.linspace(0, 4.0, 3)
    robot_vel_y_range = np.linspace(0, 4.0, 3)

    # Iterate through all scenario combinations
    scenario_count = 0
    successful_shots = 0

    for robot_x in robot_x_range:
        for robot_y in robot_y_range:
            for robot_yaw in robot_yaw_range:
                for robot_vel_x in robot_vel_x_range:
                    for robot_vel_y in robot_vel_y_range:
                        scenario_count += 1

                        robot_pos_field = np.array([robot_x, robot_y, ROBOT_POS_Z_F])
                        robot_yaw_field = robot_yaw
                        robot_velocity_field = np.array(
                            [robot_vel_x, robot_vel_y, ROBOT_VEL_Z_F]
                        )
                        shooter_offset_robot = SHOOTER_OFFSET_R
                        hub_center_field = np.array([HUB_X, HUB_Y, HUB_Z])

                        with cp.cuda.Device(0):
                            candidate = solve_shot_planar(
                                robot_pos_field,
                                robot_velocity_field,
                                shooter_offset_robot,
                                robot_yaw_field,
                                hub_center_field,
                            )

                        if candidate is not None:
                            successful_shots += 1
                            print(
                                f"X={robot_x:.1f}m, Y={robot_y:.1f}m, Yaw={np.rad2deg(robot_yaw):.1f}°, ",
                                f"VelX={robot_vel_x:.1f}m/s, VelY={robot_vel_y:.1f}m/s, ",
                                f"Pitch={np.rad2deg(candidate.pitch_rad):.1f}°, ",
                                f"Speed={candidate.flywheel_speed:.2f}m/s, Error={candidate.lateral_error:.3f}m",
                            )
    
    print(f"\nTotal scenarios: {scenario_count}, Successful: {successful_shots}")

def process_scenarios(chunk):
    """Process a chunk of scenarios and return results."""
    successful = 0
    for robot_x, robot_y, robot_yaw, robot_vel_x, robot_vel_y in chunk:
        robot_pos_field = np.array([robot_x, robot_y, ROBOT_POS_Z_F])
        robot_velocity_field = np.array([robot_vel_x, robot_vel_y, ROBOT_VEL_Z_F])
        hub_center_field = np.array([HUB_X, HUB_Y, HUB_Z])
        
        candidate = solve_shot_planar(
            robot_pos_field,
            robot_velocity_field,
            SHOOTER_OFFSET_R,
            robot_yaw,
            hub_center_field,
        )
        
        if candidate is not None:
            successful += 1
            print(
                f"X={robot_x:.1f}m, Y={robot_y:.1f}m, Yaw={np.rad2deg(robot_yaw):.1f}°, "
                f"VelX={robot_vel_x:.1f}m/s, VelY={robot_vel_y:.1f}m/s, "
                f"Pitch={np.rad2deg(candidate.pitch_rad):.1f}°, "
                f"Speed={candidate.flywheel_speed:.2f}m/s, Error={candidate.lateral_error:.3f}m"
            )
    return successful

# Entry point for script execution
if __name__ == "__main__":
    simulate()
