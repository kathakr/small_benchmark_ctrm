# value_iteration_mod.py
import math
from collections import deque

class ValueIteration:
    def __init__(self, environment=None, ctrm=None):
        self.V = {}  # Value table: V[(env_state, ctrm_state, time_index)] = probability
        self.env = environment
        self.ctrm = ctrm
        self.states = None
        self.num_steps = 0
        self.delta = 0.0

    def doVI(self, T, epsilon):
        """
        Main function to run Time-Bounded Reachability Value Iteration.
        T: Total time horizon
        epsilon: Approximation parameter
        Returns: Probability of success from initial state with full time budget
        """
        print(f"\n{'='*60}")
        print("STARTING TIME-BOUNDED VALUE ITERATION")
        print(f"{'='*60}")
        print(f"Time Bound (T): {T}")
        print(f"Epsilon (ε): {epsilon}")

        # 1. Find maximum rate from CTRM (lambda_max)
        print("\n1. Calculating maximum exit rate (λ_max) from CTRM...")
        lambda_max = 0
        for u in self.ctrm.states:
            for s in self.env.states:
                for a in self.env.actions:
                    rate = self.ctrm.get_rate_counterfactual(u, s, a)
                    if rate > lambda_max:
                        lambda_max = rate
                        print(f"   New λ_max: {lambda_max} found at (CTRM:{u}, Env:{s}, Action:{a})")
        print(f"   Final λ_max: {lambda_max}")

        # 2. Calculate discretization step delta (Theorem 1)
        self.delta = (2 * epsilon) / (lambda_max * T)
        print(f"\n2. Calculated discretization step δ = (2 * ε) / (λ_max * T) = {self.delta:.6f}")

        # 3. Calculate number of discrete time steps
        self.num_steps = int(T / self.delta) + 1
        print(f"   Number of discrete time steps: {self.num_steps} (0 to {self.num_steps-1})")

        # 4. Discover all environment states
        print(f"\n3. Discovering all environment states...")
        self.states = self.fill_states()
        print(f"   Discovered states: {self.states}")

        # 5. Initialize value table V
        print(f"\n4. Initializing value table V[s, u, k]...")
        self.fill_vtable(T, self.delta)
        print("   Initialization complete.")

        # 6. Run the backwards induction value iteration
        print(f"\n5. Starting backwards induction value iteration...")
        success_prob = self.value_iteration(T, self.delta)
        
        print(f"\n{'='*60}")
        print("VALUE ITERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Optimal probability of success within {T} seconds: {success_prob:.6f}")
        
        return success_prob

    def fill_states(self):
        """Discover all reachable states in the environment using BFS."""
        initial_state = self.env.initstate
        state_set = {initial_state}
        queue = deque([initial_state])
        
        print(f"   Starting from initial state: {initial_state}")
        
        while queue:
            current_state = queue.popleft()
            next_state_dict = self.env.next_state(current_state, self.env.actions[0])
            
            for next_state in next_state_dict:
                if next_state not in state_set:
                    state_set.add(next_state)
                    queue.append(next_state)
                    print(f"   Discovered new state: {next_state}")
        return list(state_set)

    def fill_vtable(self, T, delta):
        """Initialize the value table V."""
        # V[(env_state, ctrm_state, time_index)] = probability
        for state in self.states:
            for ctrm_state in self.ctrm.states:
                for time_index in range(self.num_steps):
                    product_state = (state, ctrm_state, time_index)
                    # Initialize value: 1.0 if goal state, 0.0 otherwise
                    if self.ctrm.is_accepting(ctrm_state):
                        self.V[product_state] = 1.0
                    else:
                        self.V[product_state] = 0.0

    def value_iteration(self, T, delta):
        """Perform backwards induction value iteration."""
        print(f"   Beginning backwards induction from time step 0 to {self.num_steps-1}...")
        
        # Backwards induction: from time step 1 to max step
        for time_index in range(1, self.num_steps):
            if time_index % 10 == 0:  # Print progress every 10 steps
                print(f"   Processing time step {time_index}/{self.num_steps-1}")
            
            for state in self.states:
                for ctrm_state in self.ctrm.states:
                    product_state = (state, ctrm_state, time_index)
                    
                    # Skip if this is already a goal state (probability is 1)
                    if self.ctrm.is_accepting(ctrm_state):
                        continue
                    
                    max_value = 0.0
                    best_action = None
                    
                    # Loop over all actions to find the best one
                    for action in self.env.actions:
                        action_value = 0.0
                        rate = self.ctrm.get_rate_counterfactual(ctrm_state, state, action)
                        
                        if rate == 0:
                            continue
                            
                        # Get transition probabilities
                        next_states = self.env.next_state(state, action)
                        
                        # For each possible next state
                        for next_env_state, prob in next_states.items():
                            # Get the label of the CURRENT state (for CTRM transition)
                            label = self.env.get_label(state)
                            next_ctrm_state = self.ctrm.delta_u(ctrm_state, label)
                            
                            # Time progresses by delta
                            new_time_index = time_index - 1
                            next_product_state = (next_env_state, next_ctrm_state, new_time_index)
                            
                            # Probability of transition happening in time delta
                            trans_prob = 1 - math.exp(-rate * delta)
                            future_value = self.V.get(next_product_state, 0.0)
                            
                            # Probability of NO transition happening
                            no_trans_prob = math.exp(-rate * delta)
                            same_product_state = (state, ctrm_state, new_time_index)
                            same_future_value = self.V.get(same_product_state, 0.0)
                            
                            # Add to action value
                            contribution = prob * (trans_prob * future_value + no_trans_prob * same_future_value)
                            action_value += contribution
                            
                            if time_index == self.num_steps - 1:  # Debug print for first iteration
                                print(f"     Action '{action}': {prob:.2f} * [{trans_prob:.3f}*{future_value:.3f} + {no_trans_prob:.3f}*{same_future_value:.3f}] = {contribution:.4f}")
                        
                        if action_value > max_value:
                            max_value = action_value
                            best_action = action
                    
                    # Update value for this state
                    old_value = self.V[product_state]
                    self.V[product_state] = max_value
                    
                    if time_index == self.num_steps - 1:  # Debug print for first iteration
                        print(f"   V{product_state} = {max_value:.4f} (best action: '{best_action}')")

        # Return value of initial state with full time budget
        init_state = (self.env.initstate, self.ctrm.initstate, self.num_steps - 1)
        return self.V.get(init_state, 0.0)

    def print_value_table(self, max_states=5):
        """Print a subset of the value table for debugging."""
        print(f"\n{'='*60}")
        print("VALUE TABLE SAMPLE (First few states)")
        print(f"{'='*60}")
        
        count = 0
        for key, value in self.V.items():
            if count >= max_states:
                break
            print(f"V{key} = {value:.6f}")
            count += 1