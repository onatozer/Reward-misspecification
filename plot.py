import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from config import ENVIRONMENT_SETTINGS

COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',]


def plot_gridworld_run(grid_world_str: str, model_strs: list[str]):

    #Plot the rewards side by side
    for i, model_str in enumerate(model_strs):
        csv_str = f"training_runs/{model_str}_{grid_world_str}.csv"
        plotname = f"training_run_visualizations/{grid_world_str}_reward.png"

        df = pd.read_csv(csv_str)

        df["reward"] = df["reward"].rolling(50).mean()

        x = df["timesteps"]
        y = df["reward"]
        plt.plot(x, y, color=COLORS[i], label = f"{model_str}", linestyle='-', linewidth=2, markersize=10)
        
        plt.title(f"{grid_world_str} Reward")
        plt.xlabel("Timesteps")
        plt.ylabel("Reward")
        plt.grid(True, color='gray', linestyle='--', linewidth=1)
        plt.legend(title = "RL Algorithm")
        plt.savefig(plotname)

    plt.close()


    if grid_world_str != "DistributionalShift-v0" and grid_world_str != "IslandNavigation-v0":
        #Plot the hidden rewards side by side
        for i, model_str in enumerate(model_strs):
            csv_str = f"training_runs/{model_str}_{grid_world_str}.csv"
            plotname = f"training_run_visualizations/{grid_world_str}_performance.png"

            df = pd.read_csv(csv_str)

            df["hidden_reward"] = df["hidden_reward"].rolling(50).mean()

            x = df["timesteps"]
            y = df["hidden_reward"]
            plt.plot(x, y, color=COLORS[i], label = f"{model_str}", linestyle='-', linewidth=2, markersize=10)
            
            plt.title(f"{grid_world_str} Performance")
            plt.xlabel("Timesteps")
            plt.ylabel("Environment Performance")
            plt.grid(True, color='gray', linestyle='--', linewidth=1)
            plt.legend(title = "RL Algorithm")
            plt.savefig(plotname)

    plt.close()

    


if  __name__ == "__main__":
    models = ["A2C", "LAD"]
    for env_str in ENVIRONMENT_SETTINGS:
        plot_gridworld_run(env_str, models)