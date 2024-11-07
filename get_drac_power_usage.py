import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
from datetime import datetime, timedelta
import urllib.request
import os
from dotenv import load_dotenv
import argparse

load_dotenv()

def save_job_power_usage(job, experiment_num, username, base_path):

    url = f"https://portail.beluga.calculquebec.ca/secure/jobstats/{username}/{job['id_job']}/graph/power.json"
    cookie = f"sessionid={os.getenv('BELUGA_COOKIE')}"
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:92.0) Gecko/20100101 Firefox/92.0',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-CA,en-US;q=0.7,en;q=0.3', 'Connection': 'keep-alive',
                'Cookie': cookie,
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document','Sec-Fetch-Mode': 'navigate', 'Sec-Fetch-Site': 'none', 'Sec-Fetch-User': '?1', \
                'Cache-Control': 'max-age=0'}

    get = urllib.request.urlopen(urllib.request.Request(url, headers=headers))
    data = json.loads(get.read())
    save_path = f"{base_path}/Exp_{experiment_num}/power_metrics.csv"

    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if len(data['data'][0]['x']) < 0:
        raise Exception(f"No data found for job {job['id']}")
    
    with open(save_path, 'w') as f:
        f.write('Time,Power (W)\n')
        for x, y in zip(data['data'][0]['x'], data['data'][0]['y']):
            f.write(f'{x},{y}\n')

def get_jobs_per_experiments(experiment_list, username, base_path):
    url = f"https://portail.beluga.calculquebec.ca/api/jobs/?format=datatables&username={username}"

    print(f"Making request to {url}")

    cookie = f"sessionid={os.getenv('BELUGA_COOKIE')}"
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:92.0) Gecko/20100101 Firefox/92.0',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-CA,en-US;q=0.7,en;q=0.3', 'Connection': 'keep-alive',
                'Cookie': cookie,
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document','Sec-Fetch-Mode': 'navigate', 'Sec-Fetch-Site': 'none', 'Sec-Fetch-User': '?1', \
                'Cache-Control': 'max-age=0'}

    get = urllib.request.urlopen(urllib.request.Request(url, headers=headers))
    data = json.loads(get.read())

    for experiment in experiment_list:
        # Find most recent id_job for this experiment that has power data
        for job in data['data']:
            if job['job_name'] == f"Exp_{experiment}_train":
                try:
                    save_job_power_usage(job, experiment, username, base_path)
                    print(f"Saved data for job {job['id_job']}")
                    break
                except Exception as e:
                    print(f"No data found for job {job['id_job']}")

if __name__ == "__main__":

    if os.getenv('BELUGA_COOKIE') is None:
        raise Exception("BELUGA_COOKIE not set in .env file!")

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', type=str, nargs='*', default=[])
    parser.add_argument('-u', type=str, default='hartman')
    parser.add_argument('-p', type=str, default='./metrics')
    args = parser.parse_args()

    get_jobs_per_experiments(args.e, args.u, args.p)
