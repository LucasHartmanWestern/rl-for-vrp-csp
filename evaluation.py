import numpy as np
import matplotlib as plt

def evaluate(ev_info, metrics):
    print(f"Number of Zones: {len(ev_info)}")
    print(f"Number of EVs: {len(ev_info[0])}")
    print(f"Number of aggregations: {int(len(metrics) / len(ev_info))}")

    

def evaluate_distance():
    print("Evaluating Distance Metrics")

    # TODO:
    # - Evaluate average distance travelled to reach destination
    # - Evaluate average distance travelled to reach destination by zone
    # - Evaluate average distance travelled to reach destination by car model
    # - Evaluate average distance travelled to reach destination per episode of training
    # - Evaluate average distance travelled to reach destination per episode of training by zone
    # - Evaluate average distance travelled to reach destination per episode of training by car model

def evaluate_time():
    print("Evaluating Time Metrics")

    # TODO:
    # - Evaluate average time to reach destination
    # - Evaluate average time to reach destination by zone
    # - Evaluate average time to reach destination by car model
    # - Evaluate average time to reach destination per episode of training
    # - Evaluate average time to reach destination per episode of training by zone
    # - Evaluate average time to reach destination per episode of training by car model

def evaluate_reward():
    print("Evaluating Reward Metrics")

    # TODO:
    # - Evaluate average reward plateau
    # - Evaluate average reward plateau by zone
    # - Evaluate average reward plateau by car model
    # - Evaluate average reward per episode of training
    # - Evaluate average reward per episode of training by zone
    # - Evaluate average reward per episode of training by car model

def evaluate_training_duration():
    print("Evaluating Training Time Metrics")

    # TODO:
    # - Evaluate how long it takes to plateau to reward
    # - Evaluate how long it takes to retrain after defining base models

def evaluate_traffic_levels():
    print("Evaluating Traffic Metrics")

    # TODO:
    # - Add peak traffic by charger
    # - Add average traffic levels
    # - Add traffic levels across seeds
    # - Add traffic levels across zones
    # - Add average traffic per episode of training
    # - Add average traffic per episode of training by zone
