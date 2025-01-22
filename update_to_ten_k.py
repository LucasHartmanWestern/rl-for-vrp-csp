import os
import shutil
import yaml

starting_exp_index = 4000
ending_exp_index = 4071

chunk_gap = 1000

for i in range(starting_exp_index, ending_exp_index + 1):
    print(f"Updating experiment {i}")
    
    # Load the config file
    with open(f"experiments/Exp_{i}/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    initial_num_aggs = config["federated_learning_settings"]["aggregation_count"]
    algorithm = config["algorithm_settings"]["algorithm"]

    seed = config["environment_settings"]["seed"]

    for offset in range(0, 3):

        # Update the num of episodes or aggregation count to have 10k total episodes
        if initial_num_aggs == 6:
            config["federated_learning_settings"]["aggregation_count"] = 10
        elif initial_num_aggs == 30:
            config["federated_learning_settings"]["aggregation_count"] = 50
        elif initial_num_aggs == 1:
            config["nn_hyperparameters"]["num_episodes"] = 10000

        config["environment_settings"]["seed"] = seed * (offset + 1)

        # Save the config file
        with open(f"experiments/Exp_{i + (offset * chunk_gap)}/config.yaml", "w") as file:
            yaml.dump(config, file)

        # Load the description file
        with open(f"experiments/Exp_{i}/description.txt", "r") as file:
            description = file.read()

        description = description.replace(f"Experiment {i}", f"Experiment {i + (offset * chunk_gap)}")
        description = description.replace(f"Seed: {seed}", f"Seed: {seed * (offset + 1)}")

        # Update the description file
        if initial_num_aggs == 6:
            description = description.replace("Number of aggregations: 6", "Number of aggregations: 10")
        elif initial_num_aggs == 30:
            description = description.replace("Number of aggregations: 30", "Number of aggregations: 50")
        elif initial_num_aggs == 1:
            description = description.replace("Number of episodes: 6000", "Number of episodes: 10000")

        # Update the description file
        with open(f"experiments/Exp_{i + (offset * chunk_gap)}/description.txt", "w") as file:
            file.write(description)

        # Load the train_job.sh file
        with open(f"experiments/Exp_{i}/train_job.sh", "r") as file:
            train_job = file.read()

        # Load the eval_job.sh file
        with open(f"experiments/Exp_{i}/eval_job.sh", "r") as file:
            eval_job = file.read()

        # Update the exp_num in the train_job.sh and eval_job.sh files
        train_job = train_job.replace(f"Exp_{i}", f"Exp_{i + (offset * chunk_gap)}")
        eval_job = eval_job.replace(f"Exp_{i}", f"Exp_{i + (offset * chunk_gap)}")

        train_job = train_job.replace("--mem=64G", "--mem=32G")
        eval_job = eval_job.replace("--mem=64G", "--mem=32G")

        train_job = train_job.replace(f"-e {i}", f"-e {i + (offset * chunk_gap)}")
        eval_job = eval_job.replace(f"-e {i}", f"-e {i + (offset * chunk_gap)}")

        train_job = train_job.replace(f"experiment {i}", f"experiment {i + (offset * chunk_gap)}")
        eval_job = eval_job.replace(f"experiment {i}", f"experiment {i + (offset * chunk_gap)}")

        eval_job = eval_job.replace(f"--mem=24G", "--mem=32G")

        # Update time estimates
        if algorithm == "DQN":
            train_job = train_job.replace("--time=45:00:00", "--time=90:00:00")
            eval_job = eval_job.replace("--time=45:00:00", "--time=90:00:00")
        elif algorithm == "PPO":
            train_job = train_job.replace("--time=80:00:00", "--time=133:00:00")
            eval_job = eval_job.replace("--time=80:00:00", "--time=133:00:00")

        # Save the train_job.sh file
        with open(f"experiments/Exp_{i + (offset * chunk_gap)}/train_job.sh", "w") as file:
            file.write(train_job)
        with open(f"experiments/Exp_{i + (offset * chunk_gap)}/eval_job.sh", "w") as file:
            file.write(eval_job)
