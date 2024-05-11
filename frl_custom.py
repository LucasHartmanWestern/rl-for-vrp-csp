import torch

def get_global_weights(zone_weights, ev_info, city_multiplier, zone_multiplier, model_multiplier):
    max_model_index = 0

    for info in ev_info:
        max_model_index = max(info['model_indices']) if max(info['model_indices']) > max_model_index else max_model_index

    print(f"Num of Models: {max_model_index + 1} - Num of Zones: {len(zone_weights)}")

    # Get aggregated weights for each zone
    zone_aggregates = [aggregate_weights(weights) for weights in zone_weights]

    city_weights = []

    # Get aggregated weights for each model
    model_aggregates = []

    for model_index in range(max_model_index + 1):
        model_weights = []

        for zone_ind, weights in enumerate(zone_weights):
            model_indices = ev_info[zone_ind]['model_indices']

            for nn_ind, w in enumerate(weights):
                city_weights.append(w)
                if model_indices[nn_ind] == model_index:
                    model_weights.append(w)

        model_aggregates.append(aggregate_weights(model_weights))

    city_aggregates = aggregate_weights(city_weights)

    # ZxM matrix of weights where Z is the number of zones and M is the number of models
    global_weights = [[{} for _ in range(len(model_aggregates))] for _ in range(len(zone_aggregates))]

    # Calculate the combined weights for each zone and model pair
    for z in range(len(zone_aggregates)):
        for m in range(len(model_aggregates)):
            combined_weights = {}
            for key in zone_aggregates[z].keys():
                # Create each entry of global weights by doing a weighted average of the zone and model
                # (0.75 x Zone Weight z) + (0.25 x Model Weight m) = Entry (z, n) of global_weights
                combined_weights[key] = city_multiplier * city_aggregates[key] + zone_multiplier * zone_aggregates[z][key] + model_multiplier * model_aggregates[m][key]
            global_weights[z][m] = combined_weights

    return global_weights

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