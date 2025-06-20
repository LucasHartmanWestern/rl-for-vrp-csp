# SURE-DM: Sustainable Urban Routing Evaluation of Decision Makers 

## Overview

This paper presents a federated learning-based framework to tackle sustainable and dynamic electric vehicle routing, integrating multiple decision-making (DM) paradigms within a realistic simulation environment. Our environment models real-world conditions, including variable traffic patterns, seasonal temperature effects on battery performance, and authentic charging station distributions and constraints. In this setting, we compare the performance and environmental impact of reinforcement learning and transformer-based decision-makers, analyzing their ability to minimize travel distance, reduce peak charging station loads, and conserve energy. We use a federated learning approach which allows fine-grained per-vehicle DM, while aggregating experience at city, zone, and car model levels to accelerate learning and enhance scalability. By sharing policies among agents, the system adapts more rapidly to new conditions and achieves robust solutions that improve upon centralized methods. We evaluate the training overhead and estimate CO$_2$ emissions, demonstrating that federated learning can reduce environmental costs and training durations. Our findings offer actionable insights for electric vehicle fleet operators, urban planners, and policymakers, illustrating the promise of federated, data-driven DM methods in promoting environmentally responsible and grid-friendly electric vehicle routing.

## Authors

- Lucas Hartman ([lhartma8@uwo.ca](mailto:lhartma8@uwo.ca))
- Santiago Gomez-Rosero ([sgomezro@uwo.ca](mailto:sgomezro@uwo.ca))
- Ethan Pigou ([epigou@uwo.ca](mailto:epigou@uwo.ca))
- Miriam A. M. Capretz ([mcapretz@uwo.ca](mailto:mcapretz@uwo.ca))

Department of Electrical and Computer Engineering, Western University, London, Ontario, Canada, N6A 5B9


## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/LucasHartmanWestern/rl-for-vrp-csp.git
    cd rl-for-vrp-csp
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Configuration

- **Environment Configuration**: `environment_config.yaml` contains parameters for the custom environment.
- **Agent Configuration**: `neural_network_config.yaml` contains hyperparameters for the reinforcement learning agent.
- **Algorithm Configuration**: `algorithm_config.yaml` contains algorithm to use.
- **Evaluation Configuration**: `evaluation_config.yaml` contains values to change how much information is displayed during training.

## Usage

### Training the Model

Run the main script to train the reinforcement learning agent using federated learning:
```sh
python app.py
```

### Evaluating the Model

Evaluate the trained models created:
```sh
python ?.py
```

## Diagrams
### Process Diagram
![Process Diagram](images/process_diagram.png)
