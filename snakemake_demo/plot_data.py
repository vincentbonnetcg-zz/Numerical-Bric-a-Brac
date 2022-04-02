import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_time_histogram(data, output_file):
    dates = np.asarray([datetime.strptime(d.replace('24:00', '23:59'), '%m/%d/%Y %H:%M') for d in data['datetime']])
    times = [d.hour+d.minute/60.0 for d in dates]

    plt.figure(figsize=(8, 5), dpi=80)
    ufo_shape = data['shape'][0]
    plt.suptitle(f'Time of day: {ufo_shape}')
    plt.ylabel('UFOs')
    plt.xlabel('Time')
    n, bins, patches = plt.hist(times, 24, facecolor='green', alpha=0.75)
    plt.savefig(output_file)

def plot_duration_histogram(data, output_file):
    durations = np.asarray([float(i) for i in data['duration']])
    durations = durations[durations < 1200]

    plt.figure(figsize=(8, 5), dpi=80)
    ufo_shape = data['shape'][0]
    plt.suptitle(f'Event duration : {ufo_shape}')
    plt.ylabel('UFOs')
    plt.xlabel('Duration in seconds')
    n, bins, patches = plt.hist(durations, 20, facecolor='blue', alpha=0.75)
    plt.savefig(output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_file")
    parser.add_argument("-output_file")
    parser.add_argument("-attribute")
    args = parser.parse_args()   
    
    # Load file
    data = None
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    # Prepare file data
    values = None
    if args.attribute == 'time':
        plot_time_histogram(data, args.output_file)
    elif args.attribute == 'duration':
        plot_duration_histogram(data, args.output_file)
