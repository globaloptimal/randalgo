import flwr as fl
from transformers import AutoModelForSequenceClassification
from typing import List
import numpy as np
import torch
from flwr.common import Parameters, ndarray_to_bytes

def pytorch_params_to_flwr_params(model_params: torch.nn.parameter.Parameter) -> Parameters:
    tensors = [
        ndarray_to_bytes(np.asarray(param.detach().cpu().numpy()))
        for param in model_params.values()
    ]
    return Parameters(tensors, tensor_type='np')

def main():
    # Load model and get initial parameters
    model = AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2
    )
    initial_params = model.state_dict()

    # Convert initial parameters to Flower's format
    flwr_initial_params = pytorch_params_to_flwr_params(initial_params)

    # Define strategy
    strategy = fl.server.strategy.FedAdam(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        initial_parameters=flwr_initial_params,
    )

    # Start server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()

