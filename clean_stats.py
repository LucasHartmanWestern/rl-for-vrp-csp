import argparse
import pandas as pd

# Get directory using -rd
parser = argparse.ArgumentParser(description="Clean the sstat csv file")
parser.add_argument('-rd', type=str, help="Directory containing the sstat csv file")
parser.add_argument('-j', type=str, help="Job ID")
args = parser.parse_args()


# Load the sstat csv file
sstat_df = pd.read_csv(f"{args.rd}/job_{args.j}_stats.csv", delimiter='|')

# Save data in a new cleaned csv file that is human-readable
sstat_df.to_csv(f"./job_{args.j}_cleaned_stats.csv", index=False, encoding='utf-8-sig', sep=',')
