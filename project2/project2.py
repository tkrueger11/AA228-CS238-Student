import math
import numpy as np
from typing import Any
from collections.abc import Callable
import heapq
import time

TRACKING = 1

class Defaults:
    LEARNING_RATE = 0.1
    DISCOUNT_FACTOR = 0.95
    CONVERGENCE_LIMIT = 0
    UNSTYLE_STATE = lambda s: s - 1
    STYLE_ACTION = lambda a: a + 1
    UNSTYLE_ACTION = lambda a: a - 1
    COL_TYPES = [("s","u8"), ("a","u8"), ("r","f8"), ("sp","u8")]

class ReinforcementLearning:
    def __init__(self, discount_factor: float, n_states: int, unstyle_state: Callable[[Any], int], style_action: Callable[[int], Any], unstyle_action: Callable[[Any], int], col_types: list[str]):
        self.discount_factor = discount_factor
        self.n_states = n_states
        self.unstyle_state = unstyle_state
        self.style_action = style_action
        self.unstyle_action = unstyle_action
        self.col_types = col_types

    def styled_policy(self, state: int) -> Any:
        action = self.pi(state)
        return self.style_action(action)
    
    def pi(self, state: int) -> int:
        raise NotImplementedError("Subclasses must implement pi(state)")

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        raise NotImplementedError("Subclasses must implement update(state, action, reward, next_state)")

    def learn(self, csv: str) -> None:
        data = np.loadtxt(csv, delimiter=',', dtype=self.col_types, skiprows=1)
        count = 0
        for row in data:
            state, action, reward, next_state = row
            action = self.unstyle_action(action)
            state = self.unstyle_state(state)
            next_state = self.unstyle_state(next_state)

            self.update(state, action, reward, next_state)
            count += 1
            if TRACKING == 1 and count % 1000 == 0:
                print(f"Completed {count} updates")

    def export_policy(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as file:
            for state in range(self.n_states):
                action = self.styled_policy(state)
                file.write(str(action) + "\n") 

class SolveableMaxLikelihoodMDP(ReinforcementLearning):
    def __init__(self, discount_factor: float, convergence_limit: float, n_states: int, n_actions: int, unstyle_state: Callable[[Any], int], style_action: Callable[[int], Any], unstyle_action: Callable[[Any], int], col_types: list[str]):
        super().__init__(discount_factor, n_states, unstyle_state, style_action, unstyle_action, col_types)

        self.convergence_limit = convergence_limit

        self.N = np.zeros((n_states,n_actions,n_states))
        self.Rho = np.zeros((n_states,n_actions))

        self.T = None
        self.R = None
        self.policy = None

    def pi(self, state: int) -> int:
        if self.policy is None:
            raise Exception("Must solve before accessing optimal policy")
        return self.policy[state]

    def setup(self) -> None:
        n = np.sum(self.N, axis=2)
        n[n == 0] = math.inf

        self.T = self.N / n[:, :, None]
        self.R = self.Rho / n

    def solve(self):
        self.setup()

        U = np.full(self.n_states, math.inf)
        next_U = np.zeros(self.n_states)
        convergence = math.inf
        while convergence > self.convergence_limit:
            U = next_U
            next_U = self.backup(U)
            convergence = np.max(np.abs(U - next_U))

        self.policy = self.get_policy(next_U)

    def backup(self, U: np.ndarray) -> np.ndarray:
        Q = self.R + self.discount_factor * (self.T @ U.T)
        return np.max(Q, axis=1)

    def get_policy(self, U: np.ndarray) -> np.ndarray:
        Q = self.R + self.discount_factor * (self.T @ U.T)
        return np.argmax(Q, axis=1)

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        self.N[state][action][next_state] += 1
        self.Rho[state][action] += reward

class SweepableMaxLikelihoodMDP(ReinforcementLearning):
    def __init__(self, discount_factor: float, n_sweeps: int, n_states: int, n_actions: int, unstyle_state: Callable[[Any], int], style_action: Callable[[int], Any], unstyle_action: Callable[[Any], int], col_types: list[str]):
        super().__init__(discount_factor, n_states, unstyle_state, style_action, unstyle_action, col_types)

        self.n_sweeps = n_sweeps

        self.N = np.zeros((n_states,n_actions,n_states))
        self.N_sa = np.full((n_states,n_actions), math.inf)
        self.Rho = np.zeros((n_states,n_actions))

        self.U = np.zeros(n_states)
        self.policy = np.full(n_states, -1)

        self.predecessors = [set() for _ in range(n_states)]

        self.priority_queue = []
        self.priority_cache = {}

    def pi(self, state: int) -> int:
        action = -1
        increment = -1 if state > self.n_states / 2 else 1
        while action == -1 and state >= 0 and state < self.n_states:
            action = self.policy[state]
            state += increment
        return action

    def backup(self, state: int) -> None:
        n = self.N_sa[state]

        T = self.N[state] / n[:, None]
        R = self.Rho[state] / n

        Q = R + self.discount_factor * (T @ self.U)
        Q[self.N_sa[state] == 0] = -math.inf

        U = np.max(Q)
        U = 0 if math.isinf(U) else U

        self.policy[state] = np.argmax(Q)
        self.U[state] = np.max(Q)

    def set_priority(self, state: int, priority: float) -> None:
        heapq.heappush_max(self.priority_queue, (priority, state))
        self.priority_cache[state] = priority

    def run_sweep(self, state: int) -> None:
        prev_U = self.U[state]
        self.backup(state)

        delta_U = abs(prev_U - self.U[state])

        for prev_state, prev_action in self.predecessors[state]:
            T = self.N[prev_state][prev_action][state] / self.N_sa[prev_state][prev_action]
            previous_priority = self.priority_cache[prev_state] if prev_state in self.priority_cache else 0
            priority = max(T * delta_U, previous_priority)
            if priority > previous_priority:
                self.set_priority(prev_state, priority)

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        self.N[state][action][next_state] += 1
        self.Rho[state][action] += reward

        if math.isinf(self.N_sa[state][action]):
            self.N_sa[state][action] = 1
        else:
            self.N_sa[state][action] += 1

        self.predecessors[next_state].add((state, action))

        self.set_priority(state, math.inf)

        iterations = self.n_sweeps
        while iterations > 0:
            if not self.priority_queue:
                break

            priority, state_p = heapq.heappop_max(self.priority_queue)

            if state_p in self.priority_cache and self.priority_cache[state_p] == priority:
                iterations -= 1
                del self.priority_cache[state_p]
                self.run_sweep(state_p)

class QLearning(ReinforcementLearning):
    def __init__(self, discount_factor: float, n_states: int, n_actions: int, unstyle_state: Callable[[Any], int], style_action: Callable[[int], Any], unstyle_action: Callable[[Any], int], col_types: list[str]):
        super().__init__(discount_factor, n_states, unstyle_state, style_action, unstyle_action, col_types)
        self.Q = np.full((n_states, n_actions), -math.inf)
        self.N = np.zeros((n_states, n_actions))

    def pi(self, state: int) -> int:
        action = -1
        increment = -1 if state > self.n_states / 2 else 1
        while action == -1 and state >= 0 and state < self.n_states:
            action = np.argmax(self.Q[state])
            state += increment
        return action

    def value(self, state: int) -> int:
        return np.max(self.Q[state])

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        self.N[state][action] += 1
        bootstrapped_value = self.value(next_state)
        if math.isinf(bootstrapped_value): bootstrapped_value = 0
        if math.isinf(self.Q[state][action]): self.Q[state][action] = 0
        target_value = self.Q[state][action]
        learning_rate = 1 / self.N[state][action]
        self.Q[state][action] += learning_rate * (reward + self.discount_factor * bootstrapped_value - target_value)

class SARSA(ReinforcementLearning):
    def __init__(self, discount_factor: float, n_states: int, n_actions: int, unstyle_state: Callable[[Any], int], style_action: Callable[[int], Any], unstyle_action: Callable[[Any], int], col_types: list[str]):
        super().__init__(discount_factor, n_states, unstyle_state, style_action, unstyle_action, col_types)
        self.Q = np.zeros((n_states, n_actions))
        self.N = np.zeros((n_states, n_actions))

    def pi(self, state: int) -> int:
        return np.argmax(self.Q[state])

    def update(self, state: int, action: int, reward: float, next_state: int, next_action: int) -> None:
        self.N[state][action] += 1
        bootstrapped_value = self.Q[next_state][next_action]
        target_value = self.Q[state][action]
        learning_rate = 1 / self.N[state][action]
        self.Q[state][action] += learning_rate * (reward + self.discount_factor * bootstrapped_value - target_value)

    def learn(self, csv: str) -> None:
        data = np.loadtxt(csv, delimiter=',', dtype=self.col_types, skiprows=1)
        max_idx = data.shape[0] - 1
        for idx, row in enumerate(data):
            state, action, reward, next_state = row
            action = self.unstyle_action(action)
            state = self.unstyle_state(state)
            next_state = self.unstyle_state(next_state)
            next_action = self.unstyle_action(data[idx + 1][1]) if idx < max_idx else self.pi(state)

            self.update(state, action, reward, next_state, next_action)


def run_solveable_model_based(n_states: int, n_actions: int, learn_csv: str, export_path: str) -> None:
    agent = SolveableMaxLikelihoodMDP(
        discount_factor=Defaults.DISCOUNT_FACTOR, 
        convergence_limit=Defaults.CONVERGENCE_LIMIT,
        n_states=n_states, 
        n_actions=n_actions, 
        unstyle_state=Defaults.UNSTYLE_STATE,
        style_action=Defaults.STYLE_ACTION, 
        unstyle_action=Defaults.UNSTYLE_ACTION,
        col_types=Defaults.COL_TYPES
    )

    agent.learn(learn_csv)
    agent.solve()
    agent.export_policy(export_path)

def run_sweep_model_based(n_states: int, n_actions: int, learn_csv: str, export_path: str, n_sweeps: int, discount_factor: float = Defaults.DISCOUNT_FACTOR) -> None:
    agent = SweepableMaxLikelihoodMDP(
        discount_factor=discount_factor, 
        n_sweeps=n_sweeps,
        n_states=n_states, 
        n_actions=n_actions, 
        unstyle_state=Defaults.UNSTYLE_STATE,
        style_action=Defaults.STYLE_ACTION, 
        unstyle_action=Defaults.UNSTYLE_ACTION,
        col_types=Defaults.COL_TYPES
    )

    agent.learn(learn_csv)
    agent.export_policy(export_path)

def run_q_learning(n_states: int, n_actions: int, learn_csv: str, export_path: str, discount_factor: float = Defaults.DISCOUNT_FACTOR) -> None:
    agent = QLearning(
        discount_factor=discount_factor, 
        n_states=n_states, 
        n_actions=n_actions, 
        unstyle_state=Defaults.UNSTYLE_STATE,
        style_action=Defaults.STYLE_ACTION, 
        unstyle_action=Defaults.UNSTYLE_ACTION,
        col_types=Defaults.COL_TYPES
    )

    agent.learn(learn_csv)
    agent.export_policy(export_path)

def run_sarsa_learning(n_states: int, n_actions: int, learn_csv: str, export_path: str) -> None:
    agent = SARSA(
        discount_factor=Defaults.DISCOUNT_FACTOR, 
        n_states=n_states, 
        n_actions=n_actions, 
        unstyle_state=Defaults.UNSTYLE_STATE,
        style_action=Defaults.STYLE_ACTION, 
        unstyle_action=Defaults.UNSTYLE_ACTION,
        col_types=Defaults.COL_TYPES
    )

    agent.learn(learn_csv)
    agent.export_policy(export_path)

def main() -> None:
    start = time.time()

    # small.csv
    print(f"Running Small Training")
    run_solveable_model_based(
        n_states=100, 
        n_actions=4, 
        learn_csv="./data/small.csv", 
        export_path="./small.policy"
    )
    print(f"Took {time.time() - start}ms to run small training")
    start = time.time()

    # medium.csv
    print(f"Running Medium Training")
    run_sweep_model_based(
        n_states=50_000, 
        n_actions=7, 
        learn_csv="./data/medium.csv", 
        export_path="./medium.policy", 
        n_sweeps=200, 
        discount_factor=1
    )
    print(f"Took {time.time() - start}ms to run medium training")
    start = time.time()

    # large.csv
    print(f"Running Large Training")
    run_sweep_model_based(
        n_states=302_020, 
        n_actions=9, 
        learn_csv="./data/large.csv", 
        export_path="./large.policy", 
        n_sweeps=50, 
        discount_factor=1
    )
    print(f"Took {time.time() - start}ms to run large training")

    print("Success!")

if __name__ == "__main__":
    main()