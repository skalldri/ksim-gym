"""Defines simple task for training a walking policy for the default humanoid."""

import asyncio
import functools
import math
from dataclasses import dataclass
from typing import Collection, Self

import attrs
import chex
import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import ksim
import mujoco
import mujoco_scenes
import mujoco_scenes.mjcf
import optax
import xax
from jaxtyping import Array, PRNGKeyArray

import logging
from enum import IntEnum, auto

logger = logging.getLogger(__name__)

# These are in the order of the neural network outputs.
ZEROS: list[tuple[str, float]] = [
    ("dof_right_shoulder_pitch_03", 0.0),
    ("dof_right_shoulder_roll_03", math.radians(-10.0)),
    ("dof_right_shoulder_yaw_02", 0.0),
    ("dof_right_elbow_02", math.radians(90.0)),
    ("dof_right_wrist_00", 0.0),
    ("dof_left_shoulder_pitch_03", 0.0),
    ("dof_left_shoulder_roll_03", math.radians(10.0)),
    ("dof_left_shoulder_yaw_02", 0.0),
    ("dof_left_elbow_02", math.radians(-90.0)),
    ("dof_left_wrist_00", 0.0),
    ("dof_right_hip_pitch_04", math.radians(-20.0)),
    ("dof_right_hip_roll_03", math.radians(-0.0)),
    ("dof_right_hip_yaw_03", 0.0),
    ("dof_right_knee_04", math.radians(-50.0)),
    ("dof_right_ankle_02", math.radians(30.0)),
    ("dof_left_hip_pitch_04", math.radians(20.0)),
    ("dof_left_hip_roll_03", math.radians(0.0)),
    ("dof_left_hip_yaw_03", 0.0),
    ("dof_left_knee_04", math.radians(50.0)),
    ("dof_left_ankle_02", math.radians(-30.0)),
]

# The total number of joints in the humanoid.
# This number affects the number of input and output dimensions of the NN.
NUM_JOINTS = 20
assert len(ZEROS) == NUM_JOINTS

# The total number of bodies in the humanoid
# Each body will produce a center of mass inertia + velocity observation.
NUM_BODIES = 23

# The number of axes for the IMU sensors. These are optional inputs to the actor NN, and always inputs to the critic NN.
NUM_ACCEL_AXIS = 3
NUM_GYRO_AXIS = 3

# The number of axes for the projected gravity observation.
NUM_PROJECTED_GRAVITY_AXIS = 3

# The number of time observations, which are the sine and cosine of the current timestep.
NUM_TIME_OBSERVATIONS = 2

# The number of ways that the actor and critic NN observes each joint.
# The we observe the joint position and velocity, so each joint is observed twice.
NUM_JOINT_OBSERVATIONS = 2

# 3 control axis: X, Y (linear translation) and Z (rotation)
NUM_CONTROL_AXIS = 3

# The minimum number of inputs to the actor neural network.
# optionally, we will add both the NUM_ACCEL_AXIS and NUM_GYRO_AXIS to the inputs.
# That depends on the value of `config.use_acc_gyro`
NUM_ACTOR_INPUTS_MIN = (
    NUM_TIME_OBSERVATIONS  # Timestep observation (sin + cos)
    + (NUM_JOINTS * NUM_JOINT_OBSERVATIONS)  # Joint positions and velocities
    + NUM_PROJECTED_GRAVITY_AXIS  # Projected gravity
    + NUM_CONTROL_AXIS  # Control vector command (X, Y, Z)
)

NUM_INERTIA_OBSERVATIONS = (
    10  # Number of observations for the intertia of the center of mass of each body
)
NUM_VELOCITY_OBSERVATIONS = (
    6  # Number of observations for the velocity of the center of mass of each body
)
NUM_BASE_POSE_OBSERVATIONS = 3 + 4  # Base position (XYZ) + orientation (WXYZ)

NUM_CRITIC_INPUTS = (
    NUM_TIME_OBSERVATIONS  # Timestep observation (sin + cos)
    + (NUM_JOINTS * NUM_JOINT_OBSERVATIONS)  # Joint positions and velocities
    + (NUM_BODIES * NUM_INERTIA_OBSERVATIONS)  # Center of mass inertia observations
    + (NUM_BODIES * NUM_VELOCITY_OBSERVATIONS)  # Center of mass velocity observations
    + NUM_ACCEL_AXIS  # IMU acceleration
    + NUM_GYRO_AXIS  # IMU gyroscope
    + NUM_PROJECTED_GRAVITY_AXIS
    + NUM_JOINTS  # Actuator forces for each joint
    + NUM_BASE_POSE_OBSERVATIONS  # Base position and orientation
    + NUM_CONTROL_AXIS  # Control vector command (X, Y, Z)
)


@dataclass
class HumanoidWalkingTaskConfig(ksim.PPOConfig):
    """Config for the humanoid walking task."""

    # Model parameters.
    hidden_size: int = xax.field(
        value=128,
        help="The hidden size for the MLPs.",
    )
    depth: int = xax.field(
        value=5,
        help="The depth for the MLPs.",
    )
    num_mixtures: int = xax.field(
        value=5,
        help="The number of mixtures for the actor.",
    )
    var_scale: float = xax.field(
        value=0.5,
        help="The scale for the standard deviations of the actor.",
    )
    use_acc_gyro: bool = xax.field(
        value=True,
        help="Whether to use the IMU acceleration and gyroscope observations.",
    )

    # Optimizer parameters.
    learning_rate: float = xax.field(
        value=3e-4,
        help="Learning rate for PPO.",
    )
    adam_weight_decay: float = xax.field(
        value=1e-5,
        help="Weight decay for the Adam optimizer.",
    )


@attrs.define(frozen=True, kw_only=True)
class JointPositionPenalty(ksim.JointDeviationPenalty):
    @classmethod
    def create_from_names(
        cls,
        names: list[str],
        physics_model: ksim.PhysicsModel,
        scale: float = -1.0,
        scale_by_curriculum: bool = False,
    ) -> Self:
        zeros = {k: v for k, v in ZEROS}
        joint_targets = [zeros[name] for name in names]

        return cls.create(
            physics_model=physics_model,
            joint_names=tuple(names),
            joint_targets=tuple(joint_targets),
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


@attrs.define(frozen=True, kw_only=True)
class BentArmPenalty(JointPositionPenalty):
    @classmethod
    def create_penalty(
        cls,
        physics_model: ksim.PhysicsModel,
        scale: float = -1.0,
        scale_by_curriculum: bool = False,
    ) -> Self:
        return cls.create_from_names(
            names=[
                "dof_right_shoulder_pitch_03",
                "dof_right_shoulder_roll_03",
                "dof_right_shoulder_yaw_02",
                "dof_right_elbow_02",
                "dof_right_wrist_00",
                "dof_left_shoulder_pitch_03",
                "dof_left_shoulder_roll_03",
                "dof_left_shoulder_yaw_02",
                "dof_left_elbow_02",
                "dof_left_wrist_00",
            ],
            physics_model=physics_model,
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


@attrs.define(frozen=True, kw_only=True)
class StraightLegPenalty(JointPositionPenalty):
    @classmethod
    def create_penalty(
        cls,
        physics_model: ksim.PhysicsModel,
        scale: float = -1.0,
        scale_by_curriculum: bool = False,
    ) -> Self:
        return cls.create_from_names(
            names=[
                "dof_left_hip_roll_03",
                "dof_left_hip_yaw_03",
                "dof_right_hip_roll_03",
                "dof_right_hip_yaw_03",
            ],
            physics_model=physics_model,
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


@attrs.define(kw_only=True)
class ControlVectorMarker(ksim.Marker):
    command_name: str = attrs.field(default="control_vector_command")
    radius: float = attrs.field(default=0.05)
    size: float = attrs.field(default=0.03)
    arrow_len: float = attrs.field(default=0.25)
    height: float = attrs.field(default=0.5)

    def update(self, trajectory: ksim.Trajectory) -> None:
        """Update the marker geometry to reflect the active joystick command."""
        cmd_vec = jnp.asarray(trajectory.command[self.command_name])

        # Marker position relative to the target body. We just move it up (above the head ideally)
        self.pos = (0.0, 0.0, self.height)

        # Get the XY components of the command vector
        command_xy = cmd_vec[..., :2]

        # get the magnitude of the command vector
        norm = jnp.linalg.norm(command_xy, axis=-1)

        # If the commanded motion is very small, show a sphere to indicate "stationary"
        if norm < 1e-6:
            self.geom = mujoco.mjtGeom.mjGEOM_SPHERE
            self.scale = (self.radius, self.radius, self.radius)

            # Commanded rotation is also very small. Show a white sphere
            if cmd_vec[..., 3] < 1e-6 and cmd_vec[..., 3] > -1e-6:
                self.rgba = (1.0, 1.0, 1.0, 0.8)
            elif cmd_vec[..., 3] > 1e-6:
                self.rgba = (
                    0.0,  # R
                    0.0,  # G
                    1.0,  # B
                    0.8,  # A
                )  # Show a blue sphere for positive Z rotation
            elif cmd_vec[..., 3] < -1e-6:
                self.rgba = (
                    0.0,  # R
                    1.0,  # G
                    0.0,  # B
                    0.8,  # A
                )  # Show a green sphere for negative Z rotation
            elif cmd_vec[..., 3] < -1e-6:
                self.rgba = (
                    1.0,  # R
                    0.0,  # G
                    0.0,  # B
                    0.8,  # A
                )  # Show a red sphere for unknown state (should never see this)
        # Else show a directional arrow
        else:
            self.geom = mujoco.mjtGeom.mjGEOM_ARROW
            self.scale = (self.size, self.size, self.arrow_len)
            self.orientation = self.quat_from_direction(
                direction=(command_xy[..., 0], command_xy[..., 1], 0.0)
            )
            self.rgba = (
                0.2,  # R
                0.8,  # G
                0.2,  # B
                0.8,  # A
            )  # Show a green-ish arrow for command vector

        # else:
        #     self.geom = mujoco.mjtGeom.mjGEOM_ARROW
        #     self.scale = (self.size, self.size, self.arrow_len)

    #
    #     # Up for left-turn, down for right-turn (like right-hand rule).
    #     direction = (0.0, 0.0, 1.0) if cmd_idx == 3 else (0.0, 0.0, -1.0)
    #     self.orientation = self.quat_from_direction(direction)
    #
    #     # Blue for left-turn, orange for right-turn.
    #     self.rgba = (0.2, 0.2, 1.0, 0.8) if cmd_idx == 3 else (1.0, 0.5, 0.0, 0.8)

    @classmethod
    def get(
        cls,
        command_name: str,
        radius: float = 0.05,
        size: float = 0.03,
        arrow_len: float = 0.25,
        rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.8),
        height: float = 0.5,
        in_robot_frame: bool = False,
    ) -> Self:
        return cls(
            command_name=command_name,
            target_type="root",
            geom=mujoco.mjtGeom.mjGEOM_SPHERE,
            scale=(radius, radius, radius),
            size=size,
            arrow_len=arrow_len,
            radius=radius,
            rgba=rgba,
            height=height,
            track_rotation=in_robot_frame,
        )


@attrs.define(frozen=True, kw_only=True)
class OldControlVectorCommand(ksim.FloatVectorCommand):
    """Command for the humanoid walking task."""

    marker_z_offset: float = attrs.field(default=0.5)
    in_robot_frame: bool = attrs.field(default=False)

    @classmethod
    def create(
        cls,
        forward_backward_range_m_per_s: tuple[float, float] = (-1.0, 1.0),
        left_right_range_m_per_s: tuple[float, float] = (-1.0, 1.0),
        rotation_range_rad_per_s: tuple[float, float] = (-0.5, 0.5),
        switch_prob: float = 0.0,
        in_robot_frame: bool = False,
        marker_z_offset: float = 0.5,
    ) -> Self:
        return cls(
            ranges=(
                forward_backward_range_m_per_s,
                left_right_range_m_per_s,
                rotation_range_rad_per_s,
            ),
            switch_prob=switch_prob,
            marker_z_offset=marker_z_offset,
            in_robot_frame=in_robot_frame,
        )

    def get_markers(self) -> Collection[ksim.Marker]:
        """Get the visualizations for the command.
        Args:
            command: The command to get the visualizations for.
        Returns:
            The visualizations to add to the scene.
        """
        return [
            ControlVectorMarker.get(
                command_name=self.command_name,
                height=self.marker_z_offset,
                in_robot_frame=self.in_robot_frame,
            )
        ]


class Direction(IntEnum):
    STAND_STILL = 0
    FORWARD = 1
    BACKWARD = 2
    STRAFE_LEFT = 3
    STRAFE_RIGHT = 4
    ROTATE_LEFT = 5
    ROTATE_RIGHT = 6


@attrs.define(frozen=True, kw_only=True)
class ControlVectorCommand(ksim.Command):
    """Command for the humanoid walking task."""

    marker_z_offset: float = attrs.field(default=0.5)
    in_robot_frame: bool = attrs.field(default=False)

    forward_backward_range_m_per_s: tuple[float, float] = attrs.field(
        default=(-0.5, 0.5)
    )
    left_right_range_m_per_s: tuple[float, float] = attrs.field(default=(-0.5, 0.5))
    rotation_range_rad_per_s: tuple[float, float] = attrs.field(default=(-0.2, 0.2))

    # Denotes the probability of generating each type of command.
    sample_probs: tuple[float, float, float, float, float, float, float] = attrs.field(
        default=(0.1, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1),
        validator=ksim.utils.validators.sample_probs_validator,
    )

    def _get_direction_array(self) -> Array:
        return jnp.array(
            [
                [0.0, 0.0, 0.0],
                [self.forward_backward_range_m_per_s[1], 0.0, 0.0],
                [self.forward_backward_range_m_per_s[0], 0.0, 0.0],
                [0.0, self.left_right_range_m_per_s[1], 0.0],
                [0.0, self.left_right_range_m_per_s[0], 0.0],
                [0.0, 0.0, self.rotation_range_rad_per_s[1]],
                [0.0, 0.0, self.rotation_range_rad_per_s[0]],
            ]
        )

    def _get_single_direction(self, direction) -> Array:
        if direction == 0:
            # Stand still
            return jnp.array([0.0, 0.0, 0.0])
        elif direction == 1:
            # Forward
            return jnp.array([self.forward_backward_range_m_per_s[1], 0.0, 0.0])
        elif direction == 2:
            # Backward
            return jnp.array([self.forward_backward_range_m_per_s[0], 0.0, 0.0])
        elif direction == 3:
            # Strafe left
            return jnp.array([0.0, self.left_right_range_m_per_s[1], 0.0])
        elif direction == 4:
            # Strafe right
            return jnp.array([0.0, self.left_right_range_m_per_s[0], 0.0])
        elif direction == 5:
            # Rotate left
            return jnp.array([0.0, 0.0, self.rotation_range_rad_per_s[1]])
        elif direction == 6:
            # Rotate right
            return jnp.array([0.0, 0.0, self.rotation_range_rad_per_s[0]])
        else:
            raise ValueError(f"Unknown direction type: {direction}")

    def _pick_random_direction(self, rng: PRNGKeyArray) -> Array:
        return jax.random.choice(
            key=rng,
            a=self._get_direction_array(),
            p=jnp.array(self.sample_probs),
        )

    def initial_command(
        self,
        physics_data: ksim.PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:

        # For now, we only generate commands along a single axis
        # to help the model learn what each axis means.

        # Pick one of the following command types:
        # 0 - stand still
        # 1 - forward
        # 2 - backward
        # 3 - strafe left
        # 4 - strafe right
        # 5 - rotate left
        # 6 - rotate right

        # TODO: scale based on curriculum level:
        # - Vary the speeds that we ask the model to walk at. Continue until mastery.
        # - Start blending multiple axis together. Start with 2 axis blending, continue until mastery.
        # - Then graduate to 3 axis blending, continue until mastery.
        # - Introduce switching between commands during the episode

        # direction = self._pick_random_direction(rng)
        return self._pick_random_direction(rng)  # self._get_single_direction(direction)

    def __call__(
        self,
        prev_command: Array,
        physics_data: ksim.PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        # For now, just return the same command for the entire episode
        return prev_command

    def get_markers(self) -> Collection[ksim.Marker]:
        """Get the visualizations for the command.

        Args:
            command: The command to get the visualizations for.

        Returns:
            The visualizations to add to the scene.
        """
        return [
            ControlVectorMarker.get(
                command_name=self.command_name,
                height=self.marker_z_offset,
                in_robot_frame=self.in_robot_frame,
            )
        ]


@attrs.define(frozen=True, kw_only=True)
class MovementPenalty(ksim.Reward):
    in_robot_frame: bool = attrs.field(default=False)

    lin_vel_penalty_scale: float = attrs.field(default=0.01)
    ang_vel_penalty_scale: float = attrs.field(default=0.01)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        # Extract variables containing the root body linear and angular velocity at each timestep
        # NOTE: qvel is in the world frame
        qvel = trajectory.qvel[..., :6]
        linvel = qvel[..., :3]
        angvel = qvel[..., 3:]

        # If we want to represent commands in the robot frame, we need to rotate the linear velocity vector based on
        # the robot's orientation in the world-frame at each timestep.
        # The world-orientation is stored in the qpos() variable. Each instance of the qpos array contians
        # (Pos(XYZ) Quat(WXYZ)), so we extract the quaternion part of the qpos array by slicing along 3:7.
        if self.in_robot_frame:
            linvel = xax.rotate_vector_by_quat(
                linvel, trajectory.qpos[..., 3:7], inverse=True
            )

        # Penalty to discourage movement in general.
        #
        # Calculate the vector 2-norm == magnitude.
        linvel_norm = jnp.linalg.norm(linvel, axis=-1) * self.lin_vel_penalty_scale
        angvel_norm = jnp.linalg.norm(angvel, axis=-1) * self.ang_vel_penalty_scale

        # Larger velocities == larger vector magnitudes == larger penalties
        vel_norm = linvel_norm + angvel_norm

        return vel_norm


@attrs.define(frozen=True, kw_only=True)
class ControlVectorAngularReward(ksim.Reward):
    command_name: str = attrs.field(default="control_vector_command")
    in_robot_frame: bool = attrs.field(default=False)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        # It's tricky to think about how this function works.
        #
        # Notice that many places in this function access inputs with (..., someNum) or (..., :someNum), or some variation..
        # This is because a "trajectory" is a bunch of individual observations from an episode.
        #
        # Each timestep contains the full simulator state + command state at that moment in time
        # This reward function gets passed all timesteps in a single call.
        #
        # Therefore, all our operations on the trajectory must be performed in a way such that they
        # evaluate correctly across all timesteps.
        #
        # More on this later

        if self.command_name not in trajectory.command:
            raise ValueError(
                f"Command {self.command_name} not found! Ensure that it is in the task."
            )
        control_vector_cmd = trajectory.command[self.command_name]

        # We are saying that, across all timesteps / observations, we expect the control vector command to have 3 dimensions.
        chex.assert_shape(control_vector_cmd, (..., 3))

        # Extract variables containing the root body linear and angular velocity at each timestep
        # NOTE: qvel is in the world frame
        qvel = trajectory.qvel[..., :6]
        angvel = qvel[..., 3:]

        # We need to calculate a new reward.
        # - We provide the robot with an X/Y vector, represnting the desired linear velocity in the robot's frame. X is forward, Y is right(?)
        # - We also provide an angular velocity vector, in the robot's frame. This is Z rotation
        #
        # We essentially want to reward the robot for having a linvel and angvel that is closely matched to the
        # input command.
        #
        # The maximum possible reward == scale
        # We want to compute some value, between 0.0 and 1.0, that represents how close the robot's XY vector is to the command XY vector

        # Extract the X and Y components of the linear velocity for each timestep
        robot_angular = angvel[
            ..., 2:3
        ].flatten()  # Extract the Z component of the angular velocity

        # Extract the joystick command into linear and angular components from the 3D vector (X, Y, ZRot)
        command_angular = control_vector_cmd[..., 2:3].flatten()

        # Compute the difference between the robot's vector and the commanded vector
        angular_diff = command_angular - robot_angular

        # Reward is scale, minus the distance from the commanded vector.
        #
        # TODO this probably needs to be scaled based on the length of the command vector.
        # EX: if we command a very small movement, and we get a very small movement in the other direction,
        # We should subtract a large amount from the reward because we were 100% wrong.

        sigma = 0.2

        # If the command magnitude is 0, then we want to reward the robot for standing still.
        # We can ignore the direction the robot is moving and just consider the magnitude of it's motion
        #
        # Else, if the command mangnitude is greater than 0, we want to reward the robot for moving in the direciton of the
        # commanded vector, and penalize it for moving in the opposite direction.
        angular_reward = jnp.where(
            command_angular > 0.001,  # Less than 1cm/s == stationary
            1.0
            - (angular_diff / command_angular).clip(
                0.0, 2.0
            ),  # Reward when the command magnitude is greater than 0. Allowed range -1.0 to 1.0
            jnp.exp(
                -(robot_angular**2) / (2 * sigma**2)
            ),  # Gaussian reward that provides more reward as the robot speed decreases
        )

        # angular_reward = self.scale - (
        #     jnp.linalg.norm(angular_diff, axis=-1).clip(0.0, self.scale)
        # )

        return angular_reward


def _world_linvel_to_robot_linvel(
    world_linvel: Array, world_orientation: Array
) -> Array:
    return xax.rotate_vector_by_quat(world_linvel, world_orientation, inverse=True)


@attrs.define(frozen=True, kw_only=True)
class ControlVectorLinearReward(ksim.Reward):
    command_name: str = attrs.field(default="control_vector_command")
    in_robot_frame: bool = attrs.field(default=False)
    epsilon: float = attrs.field(default=1e-8)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        # It's tricky to think about how this function works.
        #
        # Notice that many places in this function access inputs with (..., someNum) or (..., :someNum), or some variation..
        # This is because a "trajectory" is a bunch of individual observations from an episode.
        #
        # Each timestep contains the full simulator state + command state at that moment in time
        # This reward function gets passed all timesteps in a single call.
        #
        # Therefore, all our operations on the trajectory must be performed in a way such that they
        # evaluate correctly across all timesteps.
        #
        # More on this later

        if self.command_name not in trajectory.command:
            raise ValueError(
                f"Command {self.command_name} not found! Ensure that it is in the task."
            )
        control_vector_cmd = trajectory.command[self.command_name]

        # We are saying that, across all timesteps / observations, we expect the control vector command to have 3 dimensions.
        chex.assert_shape(control_vector_cmd, (..., 3))

        # Extract variables containing the root body linear and angular velocity at each timestep
        # NOTE: qvel is in the world frame
        qvel = trajectory.qvel[..., :6]
        linvel = qvel[..., :3]

        # If we want to represent commands in the robot frame, we need to rotate the linear velocity vector based on
        # the robot's orientation in the world-frame at each timestep.
        # The world-orientation is stored in the qpos() variable. Each instance of the qpos array contians
        # (Pos(XYZ) Quat(WXYZ)), so we extract the quaternion part of the qpos array by slicing along 3:7.
        if self.in_robot_frame:
            linvel = _world_linvel_to_robot_linvel(linvel, trajectory.qpos[..., 3:7])

        # We need to calculate a new reward.
        # - We provide the robot with an X/Y vector, represnting the desired linear velocity in the robot's frame. X is forward, Y is right(?)
        # - We also provide an angular velocity vector, in the robot's frame. This is Z rotation
        #
        # We essentially want to reward the robot for having a linvel and angvel that is closely matched to the
        # input command.
        #
        # The maximum possible reward == scale
        # We want to compute some value, between 0.0 and 1.0, that represents how close the robot's XY vector is to the command XY vector

        # We want something like the joystick reward function...
        # Joystick reward gives you a score between -1.0 and 1.0 based on the velocity of a given axis
        # The trick is that it uses the OHE vector in the final multiplication to tell if
        # the robot is moving in the commanded direction.
        #
        # Other observations:
        # - No reward for standing still
        # - No penalty for moving in an un-commanded direction
        # - Penalty for moving the wrong way of a commanded direction

        # Extract the X and Y components of the linear velocity for each timestep
        robot_xy_vec = linvel[..., :2]

        # Extract the joystick command into linear and angular components from the 3D vector (X, Y, ZRot)
        command_xy_vec = control_vector_cmd[..., :2]

        # Compute the difference between the robot's vector and the commanded vector
        #
        # Some examples:
        # Say we have command = [1.0, 0.0] and robot = [-4.0, 0.0]
        # This yields a difference of command -  robot = [5.0, 0.0] / [1.0, 0.0] = [5.0, 0.0]
        # A 500% error.
        #
        # Say we have command = [0.1, 0.0] and robot = [-0.1, 0.0]
        # This yields a difference of command -  robot = [0.2, 0.0] / [0.1, 0.0] = [2.0, 0.0]
        # A 200% error.
        xy_diff = command_xy_vec - robot_xy_vec

        error_magnitude = jnp.linalg.norm(
            xy_diff, axis=-1
        )  # Get the magnitude of the error vector

        command_magnitude = jnp.linalg.norm(
            command_xy_vec, axis=-1
        )  # Get the magnitude of the command vector

        robot_magnitude = jnp.linalg.norm(
            robot_xy_vec, axis=-1
        )  # Get the magnitude of the robot's XY vector

        sigma = 0.2

        # If the command magnitude is 0, then we want to reward the robot for standing still.
        # We can ignore the direction the robot is moving and just consider the magnitude of it's motion
        #
        # Else, if the command mangnitude is greater than 0, we want to reward the robot for moving in the direciton of the
        # commanded vector, and penalize it for moving in the opposite direction.
        linear_reward = jnp.where(
            command_magnitude > 0.001,  # Less than 1cm/s == stationary
            1.0
            - (error_magnitude / command_magnitude).clip(
                0.0, 2.0
            ),  # Reward when the command magnitude is greater than 0. Allowed range -1.0 to 1.0
            jnp.exp(
                -(robot_magnitude**2) / (2 * sigma**2)
            ),  # Gaussian reward that provides more reward as the robot speed decreases
        )

        # Reward is scale, minus the distance from the commanded vector.
        # linear_reward = self.scale - jnp.linalg.norm(xy_diff, axis=-1).clip(
        #     0.0, self.scale
        # )

        return linear_reward


@attrs.define(frozen=True, kw_only=True)
class FeetSlipPenalty(ksim.Reward):
    """Penalises COM motion while feet are in contact."""

    scale: float = -1.0
    com_vel_obs_name: str = "center_of_mass_velocity_observation"
    feet_contact_obs_name: str = "feet_contact_observation"

    def get_reward(self, traj: ksim.Trajectory) -> Array:
        vel = jnp.linalg.norm(
            traj.obs[self.com_vel_obs_name][..., :2], axis=-1, keepdims=True
        )
        contact = traj.obs[self.feet_contact_obs_name]
        return jnp.sum(vel * contact, axis=-1)


@attrs.define(frozen=True, kw_only=True)
class ContactForcePenalty(ksim.Reward):
    """Penalises vertical forces above threshold."""

    scale: float = -1.0
    max_contact_force: float = 350.0
    sensor_names: tuple[str, ...]

    def get_reward(self, traj: ksim.Trajectory) -> Array:
        forces = jnp.stack([traj.obs[n] for n in self.sensor_names], axis=-1)
        cost = jnp.clip(jnp.abs(forces[..., 2, :]) - self.max_contact_force, 0)
        return jnp.sum(cost, axis=-1)


@attrs.define(frozen=True, kw_only=True)
class RobotLinearVelocityObservation(ksim.Observation):  #
    def observe(
        self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray
    ) -> Array:
        return _world_linvel_to_robot_linvel(
            world_linvel=state.physics_state.data.qvel[..., :3],
            world_orientation=state.physics_state.data.qpos[..., 3:7],
        )


@attrs.define(frozen=True, kw_only=True)
class FeetContactObservation(ksim.FeetContactObservation):
    """Flattened (4,) contact flags of both feet."""

    def observe(
        self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray
    ) -> Array:
        return super().observe(state, curriculum_level, rng).flatten()


@attrs.define(frozen=True)
class FeetPositionObservation(ksim.Observation):
    foot_left_idx: int
    foot_right_idx: int
    floor_threshold: float = 0.0

    @classmethod
    def create(
        cls,
        *,
        physics_model: ksim.PhysicsModel,
        foot_left_site_name: str,
        foot_right_site_name: str,
        floor_threshold: float = 0.0,
    ) -> Self:
        fl = ksim.get_site_data_idx_from_name(physics_model, foot_left_site_name)
        fr = ksim.get_site_data_idx_from_name(physics_model, foot_right_site_name)
        return cls(foot_left_idx=fl, foot_right_idx=fr, floor_threshold=floor_threshold)

    def observe(
        self, state: ksim.ObservationInput, curriculum_level: Array, rng: PRNGKeyArray
    ) -> Array:
        fl = ksim.get_site_pose(state.physics_state.data, self.foot_left_idx)[
            0
        ] + jnp.array([0.0, 0.0, self.floor_threshold])
        fr = ksim.get_site_pose(state.physics_state.data, self.foot_right_idx)[
            0
        ] + jnp.array([0.0, 0.0, self.floor_threshold])
        return jnp.concatenate([fl, fr], axis=-1)


class Actor(eqx.Module):
    """Actor for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.GRUCell, ...]
    output_proj: eqx.nn.Linear
    num_inputs: int = eqx.static_field()
    num_outputs: int = eqx.static_field()
    num_mixtures: int = eqx.static_field()
    min_std: float = eqx.static_field()
    max_std: float = eqx.static_field()
    var_scale: float = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_inputs: int,
        num_outputs: int,
        min_std: float,
        max_std: float,
        var_scale: float,
        hidden_size: int,
        num_mixtures: int,
        depth: int,
    ) -> None:
        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create RNN layer
        key, rnn_key = jax.random.split(key)
        rnn_keys = jax.random.split(rnn_key, depth)
        self.rnns = tuple(
            [
                eqx.nn.GRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=rnn_key,
                )
                for rnn_key in rnn_keys
            ]
        )

        # Project to output
        self.output_proj = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=num_outputs * 3 * num_mixtures,
            key=key,
        )

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_mixtures = num_mixtures
        self.min_std = min_std
        self.max_std = max_std
        self.var_scale = var_scale

    def forward(self, obs_n: Array, carry: Array) -> tuple[distrax.Distribution, Array]:
        x_n = self.input_proj(obs_n)
        out_carries = []
        for i, rnn in enumerate(self.rnns):
            x_n = rnn(x_n, carry[i])
            out_carries.append(x_n)
        out_n = self.output_proj(x_n)

        # Reshape the output to be a mixture of gaussians.
        slice_len = self.num_outputs * self.num_mixtures
        mean_nm = out_n[..., :slice_len].reshape(self.num_outputs, self.num_mixtures)
        std_nm = out_n[..., slice_len : slice_len * 2].reshape(
            self.num_outputs, self.num_mixtures
        )
        logits_nm = out_n[..., slice_len * 2 :].reshape(
            self.num_outputs, self.num_mixtures
        )

        # Softplus and clip to ensure positive standard deviations.
        std_nm = jnp.clip(
            (jax.nn.softplus(std_nm) + self.min_std) * self.var_scale, max=self.max_std
        )

        # Apply bias to the means.
        mean_nm = mean_nm + jnp.array([v for _, v in ZEROS])[:, None]

        dist_n = ksim.MixtureOfGaussians(
            means_nm=mean_nm, stds_nm=std_nm, logits_nm=logits_nm
        )

        return dist_n, jnp.stack(out_carries, axis=0)


class Critic(eqx.Module):
    """Critic for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.GRUCell, ...]
    output_proj: eqx.nn.Linear
    num_inputs: int = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_inputs: int,
        hidden_size: int,
        depth: int,
    ) -> None:
        num_outputs = 1

        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create RNN layer
        key, rnn_key = jax.random.split(key)
        rnn_keys = jax.random.split(rnn_key, depth)
        self.rnns = tuple(
            [
                eqx.nn.GRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=rnn_key,
                )
                for rnn_key in rnn_keys
            ]
        )

        # Project to output
        self.output_proj = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=num_outputs,
            key=key,
        )

        self.num_inputs = num_inputs

    def forward(self, obs_n: Array, carry: Array) -> tuple[Array, Array]:
        x_n = self.input_proj(obs_n)
        out_carries = []
        for i, rnn in enumerate(self.rnns):
            x_n = rnn(x_n, carry[i])
            out_carries.append(x_n)
        out_n = self.output_proj(x_n)

        return out_n, jnp.stack(out_carries, axis=0)


class Model(eqx.Module):
    actor: Actor
    critic: Critic

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_actor_inputs: int,
        num_actor_outputs: int,
        num_critic_inputs: int,
        min_std: float,
        max_std: float,
        var_scale: float,
        hidden_size: int,
        num_mixtures: int,
        depth: int,
    ) -> None:
        actor_key, critic_key = jax.random.split(key)
        self.actor = Actor(
            actor_key,
            num_inputs=num_actor_inputs,
            num_outputs=num_actor_outputs,
            min_std=min_std,
            max_std=max_std,
            var_scale=var_scale,
            hidden_size=hidden_size,
            num_mixtures=num_mixtures,
            depth=depth,
        )
        self.critic = Critic(
            critic_key,
            hidden_size=hidden_size,
            depth=depth,
            num_inputs=num_critic_inputs,
        )


class HumanoidWalkingTask(ksim.PPOTask[HumanoidWalkingTaskConfig]):
    def get_optimizer(self) -> optax.GradientTransformation:
        return (
            optax.adam(self.config.learning_rate)
            if self.config.adam_weight_decay == 0.0
            else optax.adamw(
                self.config.learning_rate, weight_decay=self.config.adam_weight_decay
            )
        )

    def get_mujoco_model(self) -> mujoco.MjModel:
        mjcf_path = asyncio.run(ksim.get_mujoco_model_path("kbot", name="robot"))
        return mujoco_scenes.mjcf.load_mjmodel(mjcf_path, scene="smooth")

    def get_mujoco_model_metadata(self, mj_model: mujoco.MjModel) -> ksim.Metadata:
        metadata = asyncio.run(ksim.get_mujoco_model_metadata("kbot"))
        if metadata.joint_name_to_metadata is None:
            raise ValueError("Joint metadata is not available")
        if metadata.actuator_type_to_metadata is None:
            raise ValueError("Actuator metadata is not available")
        return metadata

    def get_actuators(
        self,
        physics_model: ksim.PhysicsModel,
        metadata: ksim.Metadata | None = None,
    ) -> ksim.Actuators:
        assert metadata is not None, "Metadata is required"
        return ksim.PositionActuators(
            physics_model=physics_model,
            metadata=metadata,
        )

    def get_physics_randomizers(
        self, physics_model: ksim.PhysicsModel
    ) -> list[ksim.PhysicsRandomizer]:
        return [
            ksim.StaticFrictionRandomizer(),
            ksim.ArmatureRandomizer(),
            ksim.AllBodiesMassMultiplicationRandomizer(
                scale_lower=0.95, scale_upper=1.05
            ),
            ksim.JointDampingRandomizer(),
            ksim.JointZeroPositionRandomizer(
                scale_lower=math.radians(-2), scale_upper=math.radians(2)
            ),
        ]

    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        return [
            ksim.PushEvent(
                x_force=1.0,
                y_force=1.0,
                z_force=0.3,
                force_range=(0.5, 1.0),
                x_angular_force=0.0,
                y_angular_force=0.0,
                z_angular_force=0.0,
                interval_range=(0.5, 4.0),
            ),
        ]

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        return [
            ksim.RandomJointPositionReset.create(
                physics_model, {k: v for k, v in ZEROS}, scale=0.1
            ),
            ksim.RandomJointVelocityReset(),
        ]

    def get_observations(
        self, physics_model: ksim.PhysicsModel
    ) -> list[ksim.Observation]:
        return [
            ksim.TimestepObservation(),
            ksim.JointPositionObservation(noise=math.radians(2)),
            ksim.JointVelocityObservation(noise=math.radians(10)),
            ksim.ActuatorForceObservation(),
            ksim.CenterOfMassInertiaObservation(),
            ksim.CenterOfMassVelocityObservation(),
            ksim.BasePositionObservation(),
            ksim.BaseOrientationObservation(),
            ksim.BaseLinearVelocityObservation(),
            ksim.BaseAngularVelocityObservation(),
            ksim.BaseLinearAccelerationObservation(),
            ksim.BaseAngularAccelerationObservation(),
            ksim.ActuatorAccelerationObservation(),
            ksim.ProjectedGravityObservation.create(
                physics_model=physics_model,
                framequat_name="imu_site_quat",
                lag_range=(0.0, 0.1),
                noise=math.radians(1),
            ),
            ksim.SensorObservation.create(
                physics_model=physics_model,
                sensor_name="imu_acc",
                noise=1.0,
            ),
            ksim.SensorObservation.create(
                physics_model=physics_model,
                sensor_name="imu_gyro",
                noise=math.radians(10),
            ),
            RobotLinearVelocityObservation(),
            FeetContactObservation.create(
                physics_model=physics_model,
                foot_left_geom_names=(
                    "KB_D_501L_L_LEG_FOOT_collision_capsule_0",
                    "KB_D_501L_L_LEG_FOOT_collision_capsule_1",
                ),
                foot_right_geom_names=(
                    "KB_D_501R_R_LEG_FOOT_collision_capsule_0",
                    "KB_D_501R_R_LEG_FOOT_collision_capsule_1",
                ),
                floor_geom_names="floor",
            ),
            FeetPositionObservation.create(
                physics_model=physics_model,
                foot_left_site_name="left_foot",
                foot_right_site_name="right_foot",
                floor_threshold=0.0,
            ),
        ]

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        return [
            ControlVectorCommand(
                in_robot_frame=True,
            )
        ]

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            # Standard rewards.
            # ksim.NaiveForwardReward(clip_max=1.25, in_robot_frame=False, scale=3.0),
            # ksim.NaiveForwardOrientationReward(scale=1.0),
            ksim.StayAliveReward(
                scale=10.0
            ),  # Originall 1.0, 10 is what was used in kbot-joystick
            ksim.UprightReward(scale=0.5),
            ControlVectorLinearReward(scale=3.0, in_robot_frame=True),
            ControlVectorAngularReward(scale=1.0, in_robot_frame=True),
            # Avoid movement penalties.
            ksim.AngularVelocityPenalty(
                index=("x", "y"), scale=-0.1
            ),  # Avoid tilting along the x and y axis
            ksim.LinearVelocityPenalty(
                index=("z"), scale=-0.1
            ),  # Avoid vertical movement
            # Normalization penalties.
            ksim.AvoidLimitsPenalty.create(physics_model, scale=-0.01),
            # Normalization penalties, only activated by curriculum
            ksim.JointAccelerationPenalty(scale=-0.01, scale_by_curriculum=True),
            ksim.JointJerkPenalty(scale=-0.01, scale_by_curriculum=True),
            ksim.LinkAccelerationPenalty(scale=-0.01, scale_by_curriculum=True),
            ksim.LinkJerkPenalty(scale=-0.01, scale_by_curriculum=True),
            ksim.ActionAccelerationPenalty(scale=-0.01, scale_by_curriculum=True),
            # Bespoke rewards.
            BentArmPenalty.create_penalty(physics_model, scale=-0.1),
            StraightLegPenalty.create_penalty(physics_model, scale=-0.2),
            # FeetSlipPenalty(scale=-0.25),
            # Follow command reward
            MovementPenalty(
                scale=-1.5,
                in_robot_frame=True,
                lin_vel_penalty_scale=0.3,
                ang_vel_penalty_scale=0.3,
            ),
        ]

    def get_terminations(
        self, physics_model: ksim.PhysicsModel
    ) -> list[ksim.Termination]:
        return [
            ksim.BadZTermination(unhealthy_z_lower=0.6, unhealthy_z_upper=1.2),
            ksim.FarFromOriginTermination(max_dist=10.0),
        ]

    def get_curriculum(self, physics_model: ksim.PhysicsModel) -> ksim.Curriculum:
        return ksim.DistanceFromOriginCurriculum(
            min_level_steps=5,
            increase_threshold=5,  #
        )

    def get_model(self, key: PRNGKeyArray) -> Model:
        return Model(
            key,
            num_actor_inputs=(
                (NUM_ACTOR_INPUTS_MIN + NUM_ACCEL_AXIS + NUM_GYRO_AXIS)
                if self.config.use_acc_gyro
                else NUM_ACTOR_INPUTS_MIN
            ),
            num_actor_outputs=NUM_JOINTS,  # Output control is the desired joint positions, in radians
            num_critic_inputs=NUM_CRITIC_INPUTS,
            min_std=0.001,
            max_std=1.0,
            var_scale=self.config.var_scale,
            hidden_size=self.config.hidden_size,
            num_mixtures=self.config.num_mixtures,
            depth=self.config.depth,
        )

    def run_actor(
        self,
        model: Actor,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[distrax.Distribution, Array]:
        time_1 = observations["timestep_observation"]
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        proj_grav_3 = observations["projected_gravity_observation"]
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        control_vec_3 = commands["control_vector_command"]

        obs = [
            jnp.sin(time_1),
            jnp.cos(time_1),
            joint_pos_n,  # NUM_JOINTS
            joint_vel_n,  # NUM_JOINTS
            proj_grav_3,  # 3
            control_vec_3,
        ]
        if self.config.use_acc_gyro:
            obs += [
                imu_acc_3,  # 3
                imu_gyro_3,  # 3
            ]

        obs_n = jnp.concatenate(obs, axis=-1)

        # Ensure that the number of inputs matches the model's expectation.
        assert len(obs_n) == model.num_inputs

        action, carry = model.forward(obs_n, carry)

        return action, carry

    def run_critic(
        self,
        model: Critic,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[Array, Array]:
        time_1 = observations["timestep_observation"]
        dh_joint_pos_j = observations["joint_position_observation"]
        dh_joint_vel_j = observations["joint_velocity_observation"]
        com_inertia_n = observations[
            "center_of_mass_inertia_observation"
        ]  # 10x number of bodies, https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#c-frame-variables
        com_vel_n = observations[
            "center_of_mass_velocity_observation"
        ]  # 6x number of bodies , https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjdata
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        proj_grav_3 = observations["projected_gravity_observation"]
        act_frc_obs_n = observations["actuator_force_observation"]
        base_pos_3 = observations["base_position_observation"]
        base_quat_4 = observations["base_orientation_observation"]

        control_vec_3 = commands["control_vector_command"]

        obs_n = jnp.concatenate(
            [
                jnp.sin(time_1),
                jnp.cos(time_1),
                dh_joint_pos_j,  # NUM_JOINTS
                dh_joint_vel_j / 10.0,  # NUM_JOINTS, TODO: why is this divided by 10?
                com_inertia_n,  # NUM_BODIES * NUM_INERTIA_OBSERVATIONS
                com_vel_n,  # NUM_BODIES * NUM_VELOCITY_OBSERVATIONS
                imu_acc_3,  # 3
                imu_gyro_3,  # 3
                proj_grav_3,  # 3
                act_frc_obs_n / 100.0,  # NUM_JOINTS, TODO: why is this divided by 100?
                base_pos_3,  # 3
                base_quat_4,  # 4
                control_vec_3,
            ],
            axis=-1,
        )

        # Ensure that the number of inputs matches the model's expectation.
        assert len(obs_n) == model.num_inputs

        return model.forward(obs_n, carry)

    def _model_scan_fn(
        self,
        actor_critic_carry: tuple[Array, Array],
        xs: tuple[ksim.Trajectory, PRNGKeyArray],
        model: Model,
    ) -> tuple[tuple[Array, Array], ksim.PPOVariables]:
        transition, rng = xs

        actor_carry, critic_carry = actor_critic_carry
        actor_dist, next_actor_carry = self.run_actor(
            model=model.actor,
            observations=transition.obs,
            commands=transition.command,
            carry=actor_carry,
        )

        # Gets the log probabilities of the action.
        log_probs = actor_dist.log_prob(transition.action)
        assert isinstance(log_probs, Array)

        value, next_critic_carry = self.run_critic(
            model=model.critic,
            observations=transition.obs,
            commands=transition.command,
            carry=critic_carry,
        )

        transition_ppo_variables = ksim.PPOVariables(
            log_probs=log_probs,
            values=value.squeeze(-1),
        )

        next_carry = jax.tree.map(
            lambda x, y: jnp.where(transition.done, x, y),
            self.get_initial_model_carry(rng),
            (next_actor_carry, next_critic_carry),
        )

        return next_carry, transition_ppo_variables

    def get_ppo_variables(
        self,
        model: Model,
        trajectory: ksim.Trajectory,
        model_carry: tuple[Array, Array],
        rng: PRNGKeyArray,
    ) -> tuple[ksim.PPOVariables, tuple[Array, Array]]:
        scan_fn = functools.partial(self._model_scan_fn, model=model)
        next_model_carry, ppo_variables = xax.scan(
            scan_fn,
            model_carry,
            (trajectory, jax.random.split(rng, len(trajectory.done))),
            jit_level=4,
        )
        return ppo_variables, next_model_carry

    def get_initial_model_carry(self, rng: PRNGKeyArray) -> tuple[Array, Array]:
        return (
            jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
            jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
        )

    def sample_action(
        self,
        model: Model,
        model_carry: tuple[Array, Array],
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
        argmax: bool,
    ) -> ksim.Action:
        actor_carry_in, critic_carry_in = model_carry
        action_dist_j, actor_carry = self.run_actor(
            model=model.actor,
            observations=observations,
            commands=commands,
            carry=actor_carry_in,
        )
        action_j = action_dist_j.mode() if argmax else action_dist_j.sample(seed=rng)
        return ksim.Action(action=action_j, carry=(actor_carry, critic_carry_in))


if __name__ == "__main__":
    HumanoidWalkingTask.launch(
        HumanoidWalkingTaskConfig(
            # Training parameters.
            num_envs=2048,
            batch_size=256,
            num_passes=4,
            epochs_per_log_step=1,
            rollout_length_seconds=12.0,
            global_grad_clip=2.0,
            # Simulation parameters.
            dt=0.002,
            ctrl_dt=0.02,
            iterations=8,
            ls_iterations=8,
            action_latency_range=(0.003, 0.01),  # Simulate 3-10ms of latency.
            drop_action_prob=0.05,  # Drop 5% of commands.
            # Visualization parameters.
            render_track_body_id=0,
            render_markers=True,
            # Checkpointing parameters.
            save_every_n_seconds=120,
        ),
    )
