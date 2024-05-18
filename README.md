# MERL: Multi-factor Edge-weighting with Reinforcement Learning for EV Routing

## Overview

With Electric Vehicle (EV) adoption rates continuing to grow, it is becoming increasingly critical to find solutions to mitigate the risk of overloading the electrical grid. This project examines the effectiveness of a deployed EV routing application powered by the Multi-factor Edge-weighting with Reinforcement Learning (MERL) algorithm. The algorithm is designed to optimize EV routing by minimizing traffic clustering and travel duration. By combining real-time data on traffic patterns, charging station availability, and grid load, the application directs drivers efficiently, aiming to alleviate congestion and balance grid demands.

The application operates through a client-server framework on real-world data, but training is conducted using a realistic simulation environment to find routes that optimize travel efficiency and grid sustainability. Our findings demonstrate that this advanced algorithmic approach significantly reduces peak charging station loads and enhances electrical grid efficiency. This study highlights the potential of integrating sophisticated algorithms with practical applications to mitigate the challenges of urban EV integration, marking a pivotal step towards sustainable urban mobility.

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