import matplotlib.pyplot as plt

# Step 1: Load the data from the text file
with open("performance_Qiteration_0.40.66.txt", "r") as file:
    lines = file.readlines()

# Step 2: Parse the data into a structured format
episodes = []
rewards = []

for line in lines[1:2000]:  # Skip the header
    episode, reward, _, _ = line.split(",")
    episodes.append(int(episode))
    rewards.append(float(reward))

# Step 3: Plot the data
plt.plot(episodes, rewards, label="Q-Iteration")
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Performance of Q-Iteration on MountainCar")
plt.legend()
plt.grid(True)

# Save the plot as a .png
plt.savefig("q_iteration_performance_0.40.66.png", dpi=300)

plt.show()
