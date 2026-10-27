import numpy as np
import matplotlib.pyplot as plt

CAPACITY_K = 15          # Shuttle capacity
COST_DISPATCH = 100      # Cost to send a shuttle
COST_WAITING = 2         # Cost per waiting customer per step
MAX_CUSTOMERS = 200      # Max people allowed at the station
GAMMA = 0.95             # Discount factor
ARRIVAL_DIST = [1, 2, 3, 4, 5]  # Possible customer arrivals
ARRIVAL_PROB = 0.2       # Probability for each arrival (1/5)


def get_expected_cost_and_next_states(state, action):
    """
    Calculate the immediate cost and the chances of reaching different next states.
    action: 0 = Don't send shuttle, 1 = Send shuttle
    """
    # Cost for sending the shuttle
    dispatch_cost = COST_DISPATCH if action == 1 else 0
    
    # Customers left at the station right after our decision
    customers_after_action = max(0, state - CAPACITY_K) if action == 1 else state
    
    # Waiting cost for the customers left behind
    waiting_cost = COST_WAITING * customers_after_action
    immediate_cost = dispatch_cost + waiting_cost
    
    # Figure out next states based on random new arrivals
    next_state_probs = {}
    for arrival in ARRIVAL_DIST:
        next_s = min(MAX_CUSTOMERS, customers_after_action + arrival)
        next_state_probs[next_s] = next_state_probs.get(next_s, 0.0) + ARRIVAL_PROB
        
    return immediate_cost, next_state_probs


def solve_enumeration(T=500):
    """
    a) Enumeration (Finite time T = 500)
    """
    states = range(MAX_CUSTOMERS + 1)
    V = np.zeros((T + 1, MAX_CUSTOMERS + 1))
    
    # Loop backward in time
    for t in range(T - 1, -1, -1):
        for s in states:
            best_val = float('inf')
            for action in [0, 1]:
                imm_cost, next_probs = get_expected_cost_and_next_states(s, action)
                expected_future_cost = sum(prob * V[t + 1, next_s] for next_s, prob in next_probs.items())
                total_cost = imm_cost + GAMMA * expected_future_cost
                if total_cost < best_val:
                    best_val = total_cost
            V[t, s] = best_val
            
    return V[0, :]


def solve_value_iteration(max_iter=1000, tol=1e-5):
    """
    b) Value Iteration (Infinite time T = infinity)
    """
    states = range(MAX_CUSTOMERS + 1)
    V = np.zeros(MAX_CUSTOMERS + 1)
    policy = np.zeros(MAX_CUSTOMERS + 1, dtype=int)
    
    for iteration in range(max_iter):
        V_new = np.zeros_like(V)
        for s in states:
            best_val = float('inf')
            best_action = 0
            for action in [0, 1]:
                imm_cost, next_probs = get_expected_cost_and_next_states(s, action)
                expected_future_cost = sum(prob * V[next_s] for next_s, prob in next_probs.items())
                total_cost = imm_cost + GAMMA * expected_future_cost
                if total_cost < best_val:
                    best_val = total_cost
                    best_action = action
            V_new[s] = best_val
            policy[s] = best_action
            
        # Stop early if values barely change
        if np.max(np.abs(V_new - V)) < tol:
            print(f"Value Iteration finished in {iteration + 1} steps.")
            break
        V = V_new
        
    return V, policy


def solve_policy_iteration(max_iter=100):
    """
    c) Policy Iteration
    """
    states = range(MAX_CUSTOMERS + 1)
    policy = np.zeros(MAX_CUSTOMERS + 1, dtype=int)
    V = np.zeros(MAX_CUSTOMERS + 1)
    
    for iteration in range(max_iter):
        # Evaluate current policy using linear equations
        A = np.zeros((MAX_CUSTOMERS + 1, MAX_CUSTOMERS + 1))
        b = np.zeros(MAX_CUSTOMERS + 1)
        
        for s in states:
            action = policy[s]
            imm_cost, next_probs = get_expected_cost_and_next_states(s, action)
            A[s, s] = 1.0
            b[s] = imm_cost
            for next_s, prob in next_probs.items():
                A[s, next_s] -= GAMMA * prob
                
        V = np.linalg.solve(A, b)
        
        # Improve policy based on updated values
        policy_stable = True
        for s in states:
            old_action = policy[s]
            best_val = float('inf')
            best_action = old_action
            
            for action in [0, 1]:
                imm_cost, next_probs = get_expected_cost_and_next_states(s, action)
                expected_future_cost = sum(prob * V[next_s] for next_s, prob in next_probs.items())
                total_cost = imm_cost + GAMMA * expected_future_cost
                if total_cost < best_val:
                    best_val = total_cost
                    best_action = action
                    
            if best_action != old_action:
                policy_stable = False
            policy[s] = best_action
            
        if policy_stable:
            print(f"Policy Iteration finished in {iteration + 1} steps.")
            break
            
    return V, policy


if __name__ == "__main__":
    states = np.arange(MAX_CUSTOMERS + 1)
    
    print("Running Enumeration...")
    v_enum = solve_enumeration(T=500)
    
    print("Running Value Iteration...")
    v_vi, policy_vi = solve_value_iteration()
    
    print("Running Policy Iteration...")
    v_pi, policy_pi = solve_policy_iteration()
    
    plt.figure(figsize=(12, 10))
    
    # Plot Value Functions
    plt.subplot(2, 1, 1)
    plt.plot(states, v_enum, label="Enumeration (T=500)", color="blue", linewidth=2)
    plt.plot(states, v_vi, label="Value Iteration (T=inf)", color="orange", linestyle="dashed", linewidth=2)
    plt.xlabel("Number of Customers Waiting")
    plt.ylabel("Expected Cost (Value)")
    plt.title("Optimal Value Functions")
    plt.legend()
    plt.grid(True)
    
    # Plot Optimal Policy
    plt.subplot(2, 1, 2)
    plt.step(states, policy_vi, label="Policy (0: Don't dispatch, 1: Dispatch)", color="green", where="mid", linewidth=2)
    plt.xlabel("Number of Customers Waiting")
    plt.ylabel("Action")
    plt.title("Optimal Dispatch Policy")
    plt.yticks([0, 1])
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("../reports/shuttle_dispatch_results.png", dpi=300)
    print("Plot saved to reports/shuttle_dispatch_results.png")
    plt.show()