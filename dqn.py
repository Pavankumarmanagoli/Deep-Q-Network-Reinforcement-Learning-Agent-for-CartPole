import gymnasium as gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32


class ReplayBuffer:
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return (
            torch.tensor(s_lst, dtype=torch.float),
            torch.tensor(a_lst),
            torch.tensor(r_lst, dtype=torch.float),
            torch.tensor(s_prime_lst, dtype=torch.float),
            torch.tensor(done_mask_lst, dtype=torch.float),
        )

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        if random.random() < epsilon:
            return random.randint(0, 1)
        return out.argmax().item()


def train(q, q_target, memory, optimizer):
    for _ in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask

        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    env = gym.make("CartPole-v1")
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    print_interval = 20
    score = 0.0
    episode_rewards = []

    for n_epi in range(10000):
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))
        s, _ = env.reset() # env.reset() in Gymnasium returns (observation, info_dict)
        terminated = False # Initialize terminated variable for Gymnasium API
        truncated = False # Initialize truncated variable for Gymnasium API
        episode_score = 0.0

        # The loop condition should use both terminated and truncated to check if the episode has ended.
        while not (terminated or truncated):
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            # env.step in Gymnasium returns (observation, reward, terminated, truncated, info)
            s_prime, r, terminated, truncated, info = env.step(a)

            # done_mask should be 0.0 if the episode terminated due to reaching a terminal state.
            # Truncated episodes are often treated as non-terminal for Q-value updates in DQN.
            done_mask = 0.0 if terminated else 1.0

            memory.put((s, a, r / 100.0, s_prime, done_mask))

            s = s_prime
            score += r
            episode_score += r

        episode_rewards.append(episode_score)

        if memory.size() > 2000:
            train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print(
                f"n_episode: {n_epi}, score: {score / print_interval:.1f}, "
                f"n_buffer: {memory.size()}, eps: {epsilon * 100:.1f}%"
            )
            score = 0.0

    torch.save(q.state_dict(), "dqn_cartpole_model.pth")
    print("Model saved as dqn_cartpole_model.pth")
    env.close()


if __name__ == "__main__":
    main()
