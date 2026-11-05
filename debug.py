from ksim import TrajectoryDataset
from dpshdl.dataloader import Dataloader

if __name__ == "__main__":

    dataset = TrajectoryDataset(
        path="/home/salldritt/workspace/ksim_playground/ksim-gym/humanoid_walking_task/run_26/dataset_263.npz",
        meta_path="/home/salldritt/workspace/ksim_playground/ksim-gym/humanoid_walking_task/run_26/dataset_263.meta.json",
    )

    dataset.index = 0
    for i in range(dataset.meta["num_samples"]):
        print(f"Sample {i}")
        trajectory, reward_state = dataset.next()
        print(f"Command: {trajectory.command['control_vector_command']}")
        print(f"QVel: {trajectory.qvel}")

    # WARNING: you MUST use num_workers=0 for TrajectoryDataset
    # because it uses CUDA under the hood with MASSIVE memory reservations.
    # TODO fix this sillyness
    # with Dataloader(dataset, num_workers=0) as loader:
    #     for trajectory, reward_state in loader:
    #         print(trajectory)
    #         print(reward_state)

    print(f"Num Samples: {dataset.meta["num_samples"]}")
