import numpy as np

data = np.loadtxt("./data/medium.csv", delimiter=',', dtype=[("s","u8"), ("a","u1"), ("r","i8"), ("sp","u8")], skiprows=1)
with open("./data/medium-unravelled.csv", "w", encoding="utf-8") as file:
    file.write(f"s,a,r,sp\n")
    for row in data:
        state, action, reward, next_state = row
        state = np.unravel_index(state, (99, 499))
        next_state = np.unravel_index(next_state, (99, 499))
        file.write(f'"({int(state[0])},{int(state[1])})",{action},{reward},"({int(next_state[0])},{int(next_state[1])})"\n')

