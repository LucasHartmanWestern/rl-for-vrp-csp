import os

# Loop through all Exp_XXXX folders in this directory
for folder in os.listdir(os.path.dirname(__file__)):
    if os.path.isdir(folder):
        
        # Remove "Experiment Analysis.ipynb" file if it exists
        if os.path.exists(os.path.join(folder, "Experiment Analysis.ipynb")):
            os.remove(os.path.join(folder, "Experiment Analysis.ipynb"))
        
        # Remove "Experiment Analysis.ipynb" file if it exists
        if os.path.exists(os.path.join(folder, "Experiment Analysis.ipynb")):
            os.remove(os.path.join(folder, "Experiment Analysis.ipynb"))
        
