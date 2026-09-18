# DQN, Explained Simply

This document explains how the Deep Q-Network (DQN) used in this project works, from start to end, in plain language. It uses the hawker-stall placement problem as the running example.

---

## 1. The problem in plain terms

We have a grid of Mumbai (G/North + F/North wards). Some cells are legal for a stall, some aren't. Some cells have more foot traffic than others. We want an agent that looks at this grid and picks good spots to place a fixed number of stalls (say, 25), one at a time, without repeating a spot or picking an illegal one.

This is a sequential decision problem: every choice affects what's still available for the next choice. That's exactly what reinforcement learning (RL) is built for.

---

## 2. The core RL vocabulary

- **Agent** — the thing making decisions. Here, the DQN.
- **Environment** — the world the agent acts in. Here, `HawkerZoneEnv`.
- **State / observation** — what the agent currently sees. Here, a 3-layer grid: footfall, legal mask, occupied mask.
- **Action** — a choice the agent can make. Here, "place a stall at cell X."
- **Reward** — a number telling the agent how good that action was. Here, roughly `footfall score − spread penalty`, or `-1` if the action was illegal/duplicate.
- **Episode** — one full run, start to finish. Here, one episode = placing all stalls (or hitting the step limit).

The agent's job: learn which actions lead to the most total reward over an episode, not just the next step.

---

## 3. What is a "Q-value"?

For any given state, and any action you could take from it, there's a number called the **Q-value**:

```text
Q(state, action) = "how much total future reward do I expect
                     if I take this action now, and then act well afterward?"
```

If the agent knew the true Q-value for every (state, action) pair, it would have a trivial strategy: always pick the action with the highest Q-value. The entire learning problem is: **we don't know these values ahead of time — we have to estimate them from experience.**

---

## 4. Why "Deep"? The neural network's job

The grid here is huge — 100+ rows × 90+ columns of possible cells, and the "state" changes every time a stall is placed (the occupied layer updates). There's no way to store a lookup table of Q-values for every possible state; there are too many.

So instead of a table, we use a neural network (a CNN, since our state is grid-shaped like an image) to *approximate* the Q-value function:

```text
CNN(state) → one Q-value estimate per possible action (per grid cell)
```

Feed it the 3-channel grid (footfall, legal, occupied), and it outputs a number for every cell: "how good do I think placing a stall here is, right now?" That's the "Deep" in Deep Q-Network — deep learning is used to approximate Q-values instead of memorizing them.

---

## 5. How the network learns: the Bellman idea

The network starts out with random weights, so its first guesses are garbage. It improves using a simple consistency rule, called the **Bellman equation**:

```text
Q(state, action) should ≈ reward_now + discount × max(Q(next_state, all possible next actions))
```

In words: *"the value of doing this action now should equal the immediate reward, plus the best value achievable from wherever I end up next."*

Every training step, the network:
1. Takes a state and an action it (or a past version of itself) tried.
2. Looks at what reward it actually got, and what state it landed in.
3. Computes what the Q-value *should have been* (right side of the equation above).
4. Nudges its weights so its *prediction* gets closer to that target.

Do this thousands of times, over many episodes, and the network's Q-value estimates get more and more accurate.

---

## 6. Exploration vs. exploitation

If the agent always picks the action it currently thinks is best, it can get stuck on a mediocre strategy it found early and never discover something better. So during training, DQN uses an **epsilon-greedy** policy:

- With probability epsilon (ε), take a completely random action (explore).
- Otherwise, take the action with the highest predicted Q-value (exploit).

Epsilon typically starts high (mostly random, to see lots of the grid) and decays over training (mostly exploiting what it's learned).

---

## 7. Two tricks that make DQN actually stable

Plain Q-learning with a neural network is notoriously unstable. DQN adds two fixes:

### Experience replay
Instead of learning only from the most recent action, the agent stores past experiences — `(state, action, reward, next_state)` — in a **replay buffer**. During training it samples random batches from this buffer instead of learning strictly in the order things happened. This breaks harmful correlations between consecutive steps and reuses data efficiently.

In this project: `buffer_size=500` — the last 500 experiences are kept and sampled from.

### Target network
The "target" side of the Bellman equation (`max Q(next_state, ...)`) is computed using a second, slower-updating copy of the network, not the one being actively trained. If you used the same network for both the prediction and the target, you'd be chasing a moving target that shifts every single update — training becomes unstable. The separate, more slowly updated target network keeps the goalposts still long enough for learning to converge.

---

## 8. Putting it together: the training loop

This is what `model.learn(total_timesteps=50000)` is doing under the hood, every step:

```text
1. Look at current state (footfall / legal / occupied grid).
2. Pick an action (epsilon-greedy: random or best-predicted).
3. Take that action in the environment (HawkerZoneEnv.step).
4. Observe the reward and the new state.
5. Store (state, action, reward, next_state) in the replay buffer.
6. Sample a random batch from the buffer.
7. Compute target Q-values using the target network.
8. Update the main network's weights to reduce the gap between
   its predictions and those targets.
9. Occasionally sync the target network to match the main network.
10. Repeat until the episode ends (all stalls placed, or step limit hit),
    then start a new episode. Keep going until total_timesteps is reached.
```

---

## 9. What happens after training

Once trained, the network is just used to *predict*, no more learning:

```text
1. Reset the environment (fresh grid, no stalls placed yet).
2. Feed the current state into the network → get a Q-value for every cell.
3. Mask out illegal or already-occupied cells (set their Q-value to -infinity)
   so the agent never picks them.
4. Pick the cell with the highest remaining Q-value.
5. Place the stall there, update the state.
6. Repeat until all stalls are placed.
```

This is exactly what `visualize_real.py` does: load the trained model, run this loop, and plot the resulting stall locations on the real map.

---

## 10. Quick mental model, summarized

```text
Q-value          = "how good is this move, considering everything that follows?"
Neural network    = a tool to estimate Q-values when there are too many states to memorize
Bellman equation  = the rule used to correct the network's estimates over time
Replay buffer     = a memory of past experience, so learning isn't just from the latest step
Target network     = a stabilizer, so the training target doesn't shift under the network's feet
Epsilon-greedy     = the balance between trying new things and using what's already learned
```

Training is the network getting progressively better at answering: *"if I place a stall here right now, how good is that, really?"* — and after enough episodes, its answers are good enough to produce sensible stall placements.
