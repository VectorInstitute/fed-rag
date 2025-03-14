# torch and flwr
import torch
import torchvision.transforms as transforms
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader

# partition cifar dataset
partitioner = IidPartitioner(num_partitions=2)
fds = FederatedDataset(
    dataset="uoft-cs/cifar10",
    partitioners={"train": partitioner},
)


def get_loaders(partition_id: int) -> tuple[DataLoader, DataLoader]:
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    def apply_transforms(batch: torch.Tensor) -> torch.Tensor:
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(
        apply_transforms
    )
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=32, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader
