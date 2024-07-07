import random
import matplotlib.pyplot as plt
fig, ax = plt.subplots()

heads_tails = [0, 0]

for coin in range(100): 
    heads_tails[random.randint(0, 1)] += 1
    plt.bar(['Heads', 'Tails'], heads_tails, color=['cyan', 'magenta'], edgecolor='black', linewidth=2)
    plt.pause(0.01)

plt.show()
