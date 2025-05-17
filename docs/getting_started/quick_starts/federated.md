# Centralized to Federated

In this quick start example, we'll demonstrate how to easily transform a centralized
training task into a federated one with just a few additional lines of code.

Let's start by installing the `fed-rag` library by using `pip`:

``` sh
pip install fed-rag
```

## Defining the centralized task

As with any model training endeavour, we define the model, its training loop, and
finally a function for evaluations. Experienced model builders will find this
workflow comfortably familiar, as FedRAG maintains the same essential structure
they're accustomed to while seamlessly introducing federation capabilities (as we
will see shortly in the next sub section).

### Model

We define a simple multi-layer perceptron as our model.

``` py title="the model"
import torch
import torch.nn as nn
import torch.nn.functional as F


# the model
class Net(torch.nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(42, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

!!! note
    FedRAG uses PyTorch as its main deep learning framework and so installing
    `fed-rag` also comes with the `torch` library.

### Training loop

We use a standard training loop that is PyTorch native, making use of the
`~torch.utils.data.DataLoader` class.

``` py title="training loop"
...  # (1)!
from torch.types import Device
from torch.utils.data import DataLoader
from fed_rag.data_structures import TestResult


def train_loop(
    model: torch.nn.Module,
    train_data: DataLoader,
    val_data: DataLoader,
    device: Device,
    num_epochs: int,
    learning_rate: float | None,
) -> TrainResult:
    """My custom train loop."""

    model.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9
    )
    model.train()
    running_loss = 0.0
    for _ in range(num_epochs):
        for batch in train_data:
            features = batch["features"]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(model(features.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(train_data)
    return TrainResult(loss=avg_trainloss)
```

1. Includes the import statements from the previous code block.

### Evaluation function

Finally, a typical evaluation function that computes the accuracy of the model.

``` py title="evaluation function"
...  # (1)!
from fed_rag.data_structures import TestResult


def test(m: torch.nn.Module, test_loader: DataLoader) -> TestResult:
    """My custom tester."""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    m.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)
            outputs = m(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(test_loader.dataset)
    return TestResult(loss=loss, metrics={"accuracy": accuracy})
```

1. Includes the import statements from the two previous code blocks.

### Centralized training

Training the model under the centralized setting is a simple matter of instantiating
a model and invoking the `train()` loop.

``` py title="Centralized training"
train_data = ...  # (1)!
val_data = ...  # (2)!
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Net()
train(
    model=model,
    train_data=train_data,
    val_data=val_data,
    device=device,
    num_epochs=3,
    learning_rate=0.1,
)
```

1. Pass in a train data loader.
2. Pass in a validation data loader.

## Federating the centralized task

In this section, we demonstrate how we can take the centralized task above,
sensibly represented by the triple (model, trainer, tester) where trainer is the
training loop, and tester is the evaluation function.

### Defining the FL task

The code block below shows how to define a [`PyTorchFLTask`](../../api_reference/fl_tasks/pytorch.md)
from the centralized trainer and tester functions, but not before automatically
performing some required inspection on them.

``` py title="Getting an FL Task"
from fed_rag.decorators import federate
from fed_rag.fl_tasks.pytorch import PyTorchFLTask


# apply decorators on the previously established trainer
train_loop = federate.trainer.pytorch(train_loop)  # (1)!
test = federate.tester.pytorch(test)  # (2)!

# define the fl task
fl_task = PyTorchFLTask.from_trainer_and_tester(
    trainer=train_loop, tester=test
)
```

1. `train_loop` as defined in the **training loop** code block.
2. `test` as defined in the **evaluation function** code block.

!!! note
    [`federate.trainer.pytorch`](../../api_reference/decorators/index.md) and
    [`federate.tester.pytorch`](../../api_reference/decorators/index.md) are both
    decorators and could have been incorporated in the **training loop** and
    **evaluation function** code blocks, respectively. We separated them here to
    clearly demonstrate the progression from centralized to federated implementation,
    making the transformation process more explicit. In typical usage, you would
    apply these decorators directly to your functions when defining them.

### Getting a server and clients

With the `FLTask` in hand, we can create a server and some clients in order to
establish a federated network.

``` py title="Getting a server and two clients"
# the server
model = Net()
server = fl_task.server(model=model)

# defining two clients
clients = []
for i in range(2):
    train_data, val_data = get_loaders(partition_id=i)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_epochs = 1
    learning_rate = 0.1

    client = fl_task.client(
        # train params
        model=model,
        train_data=train_data,
        val_data=val_data,
        device=device,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
    )
    clients.append(client)
```

### Federated training

To perform the training, we simply need to start the servers and clients!

``` py title="Starting the server and clients"
import flwr as fl

# the below commands are blocking and would need to be run in separate processes
fl.server.start_server(server=server, server_address="[::]:8080")
fl.client.start_client(client=clients[0], server_address="[::]:8080")
fl.client.start_client(client=clients[1], server_address="[::]:8080")
```

!!! note
    FedRAG uses the `flwr` library as its backend federated learning framework,
    and like `torch`, comes bundled with installation of `fed-rag`.
