# mlds-assignment1
---
## Author & Repository Information
- **Student Name:** Hsin Yu Chen
- **GitHub Repository:** [mlds-assignment1](https://github.com/shellychen1221/mlds-assignment1)

---

## Project Structure
```
mlds-assignment1/
├── value_iteration/
│   └── shuttle_mdp.py          # Part 2: Shuttle dispatching problem (Enumeration, Value & Policy Iteration)
├── policy_gradient/
│   ├── cartpole_pg.py          # Part 3a: Policy Gradient on CartPole-v1
│   ├── pong_pg.py              # Part 3b: Policy Gradient on Pong-v5 (with frame preprocessing)
│   └── cartpole_baseline.py    # Part 2: Policy Gradient with Baseline (Variance Reduction)
├── reports/
│   ├── shuttle_dispatch_results.png   # Value function & optimal policy plots
│   ├── cartpole_training.png          # CartPole training reward curve & moving average
│   ├── pong_training.png              # Pong training reward curve & moving average
│   └── cartpole_baseline_training.png # CartPole with baseline training curve
└── README.md
```
---
## Prerequisites & Environment Setup
Ensure you are using **Python 3.10+** and macOS/Linux. Set up your virtual environment and install the required dependencies:

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install core machine learning and reinforcement learning libraries
pip install torch numpy matplotlib gymnasium "gymnasium[atari,accept-rom-license]" ale-py shimmy
```
---
## Instructions on How to Run the Code

### 1. Part 2: Value and Policy Iteration (Shuttle Dispatching Problem)

Solves the shuttle dispatching MDP ($K=15, C_f=100, C_h=2, A_t \sim Unif\{1,2,3,4,5\}$) using Enumeration ($T=500$), Value Iteration ($T=\infty$), and Policy Iteration.

```bash
cd value_iteration
python shuttle_mdp.py
```

* **Output:** Generates and saves comparison plots to `reports/shuttle_dispatch_results.png`.

### 2. Part 3a: Policy Gradient on CartPole-v1

Trains a feedforward policy network using REINFORCE with Adam ($lr=0.01, \gamma=0.95$).

```bash
cd policy_gradient
python cartpole_pg.py
```

* **Output:** Saves the training reward curve and 100-episode moving average to `reports/cartpole_training.png`.

### 3. Part 3b: Policy Gradient on Pong-v5

Trains a policy network on Atari Pong using specified frame preprocessing (cropping, downsampling, and background erasing) with $\gamma=0.99$.

```bash
cd policy_gradient
python pong_pg.py
```

* **Output:** Saves the training performance plot to `reports/pong_training.png`.

### 4. Part 2: Policy Gradient with Baseline (Actor-Critic)

Improves training stability by incorporating an Actor-Critic baseline to reduce gradient variance.

```bash
cd policy_gradient
python cartpole_baseline.py
```

* **Output:** Saves the baseline-enhanced training plot to `reports/cartpole_baseline_training.png`.

## Discussion Highlights & Methodology

### Edge Cases Handled

1. In the Shuttle MDP, customer queues are strictly capped at 200 using boundary clipping to prevent array index overflow.
2. In Policy Gradient returns/advantages normalization, a small epsilon ($1e-5$) is added to the denominator to prevent division-by-zero errors when episode rewards have zero variance.

### Correctness Validation Tests

* Verified that finite-horizon Enumeration ($T=500$) and infinite-horizon Value Iteration yield identical expected cost curves.
* Tested shuttle capacity shifts (e.g., changing $K$ from 15 to 10) and confirmed that the optimal policy threshold adjusts logically.

### Future Work (Extended 3-Month Period)

1. Implement **Proximal Policy Optimization (PPO)** and Generalized Advantage Estimation (GAE) to address sample inefficiency in Atari environments.
2. Explore **Model-Based RL (MCTS)** for complex state transitions in dispatching optimization.