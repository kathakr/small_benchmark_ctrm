# benchmark_simple.py
import math
from value_iteration_time_b import ValueIteration

class SimpleGridEnv:
    def __init__(self):
        self.states = ['S0', 'S1', 'S2']
        self.actions = ['move']
        self.initstate = 'S0'
        
    def next_states(self, state):
        return self.states
        
    def next_state(self, state, action):
        if state == 'S0':
            return {'S1': 1.0}
        elif state == 'S1':
            return {'S0': 0.1, 'S2': 0.9}
        elif state == 'S2':
            return {'S2': 1.0}
        return {}
    
    def get_label(self, state):
        if state == 'S0':
            return set()
        elif state == 'S1':
            return {"package"}
        elif state == 'S2':
            return {"goal"}
        return set()

class SimpleCTRM:
    def __init__(self):
        self.states = ['U0', 'U1', 'U2']
        self.initstate = 'U0'
        self.accepting_states = {'U2'}
        
    def delta_u(self, ctrm_state, label):
        if ctrm_state == 'U0':
            if "package" in label:
                return 'U1'
            else:
                return 'U0'
        elif ctrm_state == 'U1':
            if "goal" in label:
                return 'U2'
            else:
                return 'U1'
        elif ctrm_state == 'U2':
            return 'U2'
        return ctrm_state
        
    def get_rate_counterfactual(self, ctrm_state, env_state, action):
        if env_state == 'S1':  # Congested area
            return 0.5
        if ctrm_state == 'U1' and env_state == 'S2':  # Fast path to goal
            return 2.0
        return 1.0  # Default rate
        
    def is_accepting(self, state):
        return state in self.accepting_states

def run_benchmark():
    """Run the complete benchmark test."""
    print("CREATING BENCHMARK: 'The Hurried Delivery Robot'")
    print("Environment States: S0 (Start), S1 (Package+Congested), S2 (Goal)")
    print("CTRM States: U0 (Looking), U1 (Has Package), U2 (Goal Reached)")
    print("Special Rates: S1 is congested (rate=0.5), S2 with package is fast (rate=2.0)")
    print("Transitions: From S1, 90% chance to S2, 10% chance back to S0")
    
    # Create environment and CTRM
    env = SimpleGridEnv()
    ctrm = SimpleCTRM()
    
    # Create and run Value Iteration
    vi = ValueIteration(environment=env, ctrm=ctrm)
    
    # Test parameters
    T = 8.0  # Time bound
    epsilon = 0.1  # Accuracy parameter
    
    print(f"\n{'='*60}")
    print("RUNNING VALUE ITERATION")
    print(f"{'='*60}")
    
    success_prob = vi.doVI(T, epsilon)
    
    # Print final results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Time Bound: {T} seconds")
    print(f"Approximation Parameter: ε = {epsilon}")
    print(f"Discretization Step: δ = {vi.delta:.6f}")
    print(f"Number of Time Steps: {vi.num_steps}")
    print(f"Optimal Success Probability: {success_prob:.6f}")
    
    # Print a sample of the value table
    vi.print_value_table(max_states=10)
    
    # Test with different time bounds to show the effect
    print(f"\n{'='*60}")
    print("TESTING WITH DIFFERENT TIME BOUNDS")
    print(f"{'='*60}")
    
    for test_T in [4.0, 8.0, 16.0, 32.0]:
        test_vi = ValueIteration(environment=env, ctrm=ctrm)
        test_prob = test_vi.doVI(test_T, epsilon=0.1)
        print(f"T={test_T:.1f}s -> Success Probability: {test_prob:.6f}")

if __name__ == "__main__":
    run_benchmark()