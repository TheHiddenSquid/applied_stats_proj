from dataclasses import dataclass
import torch.nn as nn # type: ignore

@dataclass
class MlpArguments():
    input_size: int
    hidden_size: int = 4
    output_size: int = 1
    loss_fn: nn.modules.loss = nn.CrossEntropyLoss()

@dataclass
class TrainingArguments():
    epochs: int = 1000
    lr: float = 0.1
    zo_lr: float = 0.1 #Mezo paper uses 1e−5,1e−6,1e−7 TODO: also, when running huggingface seems to use an adaptive learning rate
    weight_decay: float = 0.0 #MeZO paper uses 0 and 0.1
    epsilon: float = 1e-3
    zo_epochs: int = 10000