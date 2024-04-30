import torch

def aggregate_weights(local_weights_list):
    """
    Aggregate the weights from multiple local models.
    Args:
        local_weights_list: List of state_dicts from local models.
    Returns:
        A state_dict representing the aggregated weights.
    """

    # Initialize the aggregated weights with the structure of the first model's weights
    aggregated_weights = {key: torch.zeros_like(value) for key, value in local_weights_list[0].items()}

    # Sum the weights from all local models
    for local_weights in local_weights_list:
        for key in aggregated_weights.keys():
            aggregated_weights[key] += local_weights[key]

    # Take the average
    for key in aggregated_weights.keys():
        aggregated_weights[key] = aggregated_weights[key].float()  # Convert to float tensor
        aggregated_weights[key] /= len(local_weights_list)

    return aggregated_weights