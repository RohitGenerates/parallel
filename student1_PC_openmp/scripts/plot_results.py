import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_results():
    if not os.path.exists("results.csv"):
        print("results.csv not found! Run the simulation first.")
        return

    df = pd.read_csv("results.csv")

    queue_data = df[df['Type'].str.startswith('Queue')]
    sched_data = df[df['Type'].str.startswith('Schedule')]

    plt.figure(figsize=(12, 5))

    # Plot 1: Queue based Producer/Consumer
    plt.subplot(1, 2, 1)
    labels = queue_data['Type'].tolist()
    times = queue_data['Time(s)'].tolist()

    plt.bar(labels, times, color='skyblue')
    plt.title('Execution Time: Queue P/C Configurations')
    plt.ylabel('Time (seconds)')
    plt.xlabel('Configuration')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Plot 2: Loop Scheduling
    plt.subplot(1, 2, 2)
    schedules = ["static", "dynamic", "guided"]
    colors = ['r', 'g', 'b']

    for sched, col in zip(schedules, colors):
        subset = sched_data[sched_data['Type'] == f"Schedule_{sched}"]
        plt.plot(subset['Threads'], subset['Time(s)'], marker='o', label=sched, color=col)

    plt.title('Scheduling Over uneven Workload')
    plt.ylabel('Time (seconds)')
    plt.xlabel('Number of Threads')
    plt.xticks([2, 4, 8, 16])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('performance_analysis.png')
    print("Graph saved as performance_analysis.png")

if __name__ == "__main__":
    plot_results()
