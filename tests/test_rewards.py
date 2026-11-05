import jax
import xax
from train import (
    ControlVectorLinearReward,
    ControlVectorAngularReward,
    MovementPenalty,
    _world_linvel_to_robot_linvel,
)

from ksim import Trajectory
import pytest


def is_near(a, b, epsilon=1e-6) -> bool:
    if abs(a - b) < epsilon:
        return True

    return False


def get_command(
    control_vector_command: jax.Array = jax.numpy.array([[0.0, 0.0, 0.0]])
) -> xax.FrozenDict:
    return xax.FrozenDict({"control_vector_command": control_vector_command})


def get_trajectory(
    qvel: jax.Array = jax.numpy.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
    qpos: jax.Array = jax.numpy.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
    command: xax.FrozenDict = get_command(),
) -> Trajectory:
    return Trajectory(
        qpos=qpos,
        qvel=qvel,
        xpos=jax.numpy.array([[0.0, 0.0, 0.0]]),
        xquat=jax.numpy.array([[0.0, 0.0, 0.0]]),
        ctrl=jax.numpy.array([[0.0, 0.0, 0.0]]),
        obs=xax.FrozenDict(),
        command=command,
        event_state=xax.FrozenDict(),
        action=jax.numpy.array([[0.0, 0.0, 0.0]]),
        done=jax.numpy.array([[0.0, 0.0, 0.0]]),
        success=jax.numpy.array([[0.0, 0.0, 0.0]]),
        timestep=jax.numpy.array([[0.0, 0.0, 0.0]]),
        termination_components=xax.FrozenDict(),
        aux_outputs=xax.FrozenDict(),
    )


class TestMovementPenalty:
    def test_movement_penalty_linear(self):
        scale = -1.0
        lin_vel_penalty_scale = 0.1
        ang_vel_penalty_scale = 0.8
        mp = MovementPenalty(
            scale=-1.0,
            in_robot_frame=False,
            ang_vel_penalty_scale=ang_vel_penalty_scale,
            lin_vel_penalty_scale=lin_vel_penalty_scale,
        )

        trajectory = get_trajectory(
            qvel=jax.numpy.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        )

        reward = mp.get_reward(trajectory)

        # Moving @ 1m/s incurs the full lin_vel_penalty
        assert is_near(reward[0], lin_vel_penalty_scale)

    def test_movement_penalty_angular(self):
        scale = -1.0
        lin_vel_penalty_scale = 0.1
        ang_vel_penalty_scale = 0.8
        mp = MovementPenalty(
            scale=-1.0,
            in_robot_frame=False,
            ang_vel_penalty_scale=ang_vel_penalty_scale,
            lin_vel_penalty_scale=lin_vel_penalty_scale,
        )

        trajectory = get_trajectory(
            qvel=jax.numpy.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        )

        reward = mp.get_reward(trajectory)

        # Rotating @ 1 rad/s incurs the full ang_vel_penalty_scale
        assert is_near(reward[0], (ang_vel_penalty_scale))

    def test_movement_penalty_combined(self):
        scale = -1.0
        lin_vel_penalty_scale = 0.1
        ang_vel_penalty_scale = 0.8
        mp = MovementPenalty(
            scale=-1.0,
            in_robot_frame=False,
            ang_vel_penalty_scale=ang_vel_penalty_scale,
            lin_vel_penalty_scale=lin_vel_penalty_scale,
        )

        trajectory = get_trajectory(
            qvel=jax.numpy.array([[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        )

        reward = mp.get_reward(trajectory)

        # Rotating @ 1 rad/s incurs the full ang_vel_penalty
        # Moving @ 1 m/s incurs the full lin_vel_penalty
        assert is_near(reward[0], (ang_vel_penalty_scale + lin_vel_penalty_scale))


class TestControlVectorAngularReward:
    def test_control_vector_angular_reward(self):
        scale = 1.0
        cvr = ControlVectorAngularReward(scale=scale, in_robot_frame=False)

        trajectory = get_trajectory(
            qvel=jax.numpy.array([[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
            command=get_command(jax.numpy.array([[1.0, 0.0, 1.0]])),
        )

        reward = cvr.get_reward(trajectory)

        # Reward should be full scale
        assert is_near(reward[0], scale)

    def test_control_vector_angular_reward_zero_movement(self):
        scale = 1.0
        cvr = ControlVectorAngularReward(scale=scale, in_robot_frame=False)

        trajectory = get_trajectory(
            qvel=jax.numpy.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            command=get_command(jax.numpy.array([[0.0, 0.0, 0.0]])),
        )

        reward = cvr.get_reward(trajectory)

        # Reward should be full scale
        assert is_near(reward[0], scale)


class TestControlVectorLinearReward:
    def test_control_vector_linear_reward(self):
        scale = 1.0
        cvr = ControlVectorLinearReward(scale=scale, in_robot_frame=False)

        trajectory = get_trajectory(
            qvel=jax.numpy.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            command=get_command(jax.numpy.array([[1.0, 0.0, 0.0]])),
        )

        reward = cvr.get_reward(trajectory)

        # Reward should be full scale
        assert is_near(reward[0], scale)

    def test_control_vector_linear_reward_zero_movement(self):
        scale = 1.0
        cvr = ControlVectorLinearReward(scale=scale, in_robot_frame=False)

        trajectory = get_trajectory(
            qvel=jax.numpy.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            command=get_command(jax.numpy.array([[0.0, 0.0, 0.0]])),
        )

        reward = cvr.get_reward(trajectory)

        # Reward should be full scale
        assert is_near(reward[0], scale)

    def test_control_vector_linear_reward_small_movement(self):
        scale = 1.0
        cvr = ControlVectorLinearReward(scale=scale, in_robot_frame=False)

        trajectory = get_trajectory(
            qvel=jax.numpy.array(
                [[0.1, 0.1, 0.0, 0.0, 0.0, 0.0]]
            ),  # Show some small movement along two axis
            command=get_command(
                jax.numpy.array([[0.1, 0.0, 0.0]])
            ),  # Command a small movement forward along one axis
        )

        reward = cvr.get_reward(trajectory)

        # Reward should be full scale
        assert is_near(reward[0], scale), f"{reward[0]} != {scale}"

    def test_control_vector_linear_reward_test_case(self):
        scale = 1.5
        cvr = ControlVectorLinearReward(scale=scale, in_robot_frame=False)

        trajectory = get_trajectory(
            qvel=jax.numpy.array(
                [[-0.5, 1.0, 0.0, 0.0, 0.0, 0.0]]
            ),  # Show some small movement along two axis
            command=get_command(
                jax.numpy.array([[0.0, 1.0, 0.0]])
            ),  # Command a small movement forward along one axis
        )

        reward = cvr.get_reward(trajectory)

        print(f"Reward: {reward}")

        # Reward should be full scale
        # assert is_near(reward[0], scale)


class TestUtils:
    def test_world_linvel_to_robot_linvel(self):
        world_linvel_xyz = jax.numpy.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )

        world_orientation_wxyz = jax.numpy.array(
            [
                [0.707, 0.0, 0.0, 0.707],  # +90 degree rotation around z-axis
                [0.707, 0.0, 0.0, -0.707],  # -90 degree rotation around z-axis
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )

        expected_robot_linvel_xyz = jax.numpy.array(
            [
                [
                    1.0,
                    0.0,
                    0.0,
                ],  # +y world-axis motion with robot rotated +90 degrees == +x robot axis
                [
                    -1.0,
                    0.0,
                    0.0,
                ],  # +y world-axis motion with robot rotated -90 degrees == -x robot axis
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )

        robot_linvel_xyz = _world_linvel_to_robot_linvel(
            world_linvel=world_linvel_xyz,
            world_orientation=world_orientation_wxyz,
        )

        allclose = jax.numpy.allclose(
            a=robot_linvel_xyz,
            b=expected_robot_linvel_xyz,
        )

        print(allclose)

        assert allclose, "Robot linear velocities do not match expected values"
