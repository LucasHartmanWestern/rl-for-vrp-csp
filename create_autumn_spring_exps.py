import os
import shutil

# Get all the experiment directories
experiments = os.listdir("experiments")

# Function to modify the season in the config.yaml and description.txt files
def modify_season(file_path, old_season, new_season):
    with open(file_path, 'r') as file:
        filedata = file.read()
    filedata = filedata.replace(old_season, new_season)
    with open(file_path, 'w') as file:
        file.write(filedata)

# Iterate over the existing experiments
for exp in experiments:
    if exp.startswith("Exp_"):
        exp_num = int(exp.split('_')[1])
        if (4000 <= exp_num <= 4071 or 5000 <= exp_num <= 5071 or 6000 <= exp_num <= 6071):
            new_exp_num = exp_num + 3000
            new_exp_dir = f"experiments/Exp_{new_exp_num}"
            shutil.copytree(f"experiments/{exp}", new_exp_dir)

            # Find and remove any ipynb files
            for file in os.listdir(new_exp_dir):
                if file.endswith(".ipynb"):
                    os.remove(os.path.join(new_exp_dir, file))
            
            # Modify the config.yaml file
            config_path = os.path.join(new_exp_dir, "config.yaml")
            with open(config_path, 'r') as file:
                config_data = file.read()
            if "summer" in config_data:
                modify_season(config_path, "summer", "autumn")
            elif "winter" in config_data:
                modify_season(config_path, "winter", "spring")
            
            # Modify the description.txt file
            description_path = os.path.join(new_exp_dir, "description.txt")
            with open(description_path, 'r') as file:
                description_data = file.read()
            if "summer" in description_data:
                modify_season(description_path, "summer", "autumn")
            elif "winter" in description_data:
                modify_season(description_path, "winter", "spring")

            # Update the experiment number in the description.txt file
            with open(description_path, 'r') as file:
                desc_content = file.read()
            
            # Replace the experiment number in the header
            desc_content = desc_content.replace(f"### Experiment {exp_num}: ###", f"### Experiment {new_exp_num}: ###")
            
            with open(description_path, 'w') as file:
                file.write(desc_content)

            # Update the experiment number in the train_job.sh and eval_job.sh files
            train_job_path = os.path.join(new_exp_dir, "train_job.sh")
            eval_job_path = os.path.join(new_exp_dir, "eval_job.sh")

            # Replace every instance of old experiment number with new experiment number
            with open(train_job_path, 'r') as file:
                train_job_data = file.read()
            train_job_data = train_job_data.replace(f"{exp_num}", f"{new_exp_num}")
            with open(train_job_path, 'w') as file:
                file.write(train_job_data)

            with open(eval_job_path, 'r') as file:
                eval_job_data = file.read()
            eval_job_data = eval_job_data.replace(f"{exp_num}", f"{new_exp_num}")
            with open(eval_job_path, 'w') as file:
                file.write(eval_job_data)

            print(f"Modified experiment {new_exp_num}")