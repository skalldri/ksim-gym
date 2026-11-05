import pytest
import jax
from train import ControlVectorCommand, OldControlVectorCommand, Direction


class TestOldControlVectorCommand:
    def test_control_vector_command(self):
        cvc = OldControlVectorCommand.create(in_robot_frame=True, switch_prob=0.0)
        curriculum_level = jax.numpy.array([0.0])
        rng = jax.random.key(seed=1)

        initial_command = cvc.initial_command(
            physics_data=None,
            curriculum_level=curriculum_level,
            rng=rng,
        )

        # Verify that all subsequent commands are the same as the initial command
        # since switch_prob is 0.0
        for _ in range(1000):
            command = cvc(
                prev_command=initial_command,
                physics_data=None,
                curriculum_level=curriculum_level,
                rng=rng,
            )
            assert jax.numpy.all(
                command == initial_command
            ), "Subsequent commands should match the initial command"


class TestControlVectorCommand:

    def test_pick_random_direction(self):
        cvc = ControlVectorCommand(in_robot_frame=True)
        rng = jax.random.key(seed=1)

        direction = cvc._pick_random_direction(rng)

    def test_get_single_direction(self):
        cvc = ControlVectorCommand(
            in_robot_frame=True,
            forward_backward_range_m_per_s=(-1.0, 1.0),
            left_right_range_m_per_s=(-1.0, 1.0),
            rotation_range_rad_per_s=(-0.5, 0.5),
        )

        command = cvc._get_single_direction(Direction.STAND_STILL)  # Stand still
        assert jax.numpy.all(
            command == jax.numpy.array([0.0, 0.0, 0.0])
        ), "Stand still command should be [0.0, 0.0, 0.0]"

        command = cvc._get_single_direction(Direction.FORWARD)  # Forward
        assert jax.numpy.all(
            command == jax.numpy.array([1.0, 0.0, 0.0])
        ), "Forward command should be [1.0, 0.0, 0.0]"

        command = cvc._get_single_direction(Direction.BACKWARD)  # Backward
        assert jax.numpy.all(
            command == jax.numpy.array([-1.0, 0.0, 0.0])
        ), "Backward command should be [-1.0, 0.0, 0.0]"

        command = cvc._get_single_direction(Direction.STRAFE_LEFT)  # Strafe left
        assert jax.numpy.all(
            command == jax.numpy.array([0.0, 1.0, 0.0])
        ), "Strafe left command should be [0.0, 1.0, 0.0]"

        command = cvc._get_single_direction(Direction.STRAFE_RIGHT)  # Strafe right
        assert jax.numpy.all(
            command == jax.numpy.array([0.0, -1.0, 0.0])
        ), "Strafe right command should be [0.0, -1.0, 0.0]"

        command = cvc._get_single_direction(Direction.ROTATE_LEFT)  # Rotate left
        assert jax.numpy.all(
            command == jax.numpy.array([0.0, 0.0, 0.5])
        ), "Rotate left command should be [0.0, 0.0, 0.5]"

        command = cvc._get_single_direction(Direction.ROTATE_RIGHT)  # Rotate right
        assert jax.numpy.all(
            command == jax.numpy.array([0.0, 0.0, -0.5])
        ), "Rotate right command should be [0.0, 0.0, -0.5]"
