import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Global GA parameters
POP_SIZE = 40
N_GENERATIONS =15
TOURNAMENT_SIZE = 3
CX_PROB = 0.7
MUT_PROB = 0.2

class GeneticFuzzyControllerCartPole:
    def __init__(self, n_mfs=3):
        self.n_inputs = 4  # CartPole state: [position, velocity, angle, angular velocity]
        self.n_mfs = n_mfs
        self.mf_bounds = [
            (-4.8, 4.8),    # cart position
            (-3.0, 3.0),    # cart velocity (approx)
            (-0.418, 0.418),# pole angle (radians)
            (-3.0, 3.0)     # pole angular velocity (approx)
        ]
        # Boundaries for consequent parameters in each fuzzy rule
        self.rule_bounds = (-2, 2)
        # Number of fuzzy rules: (n_mfs)^n_inputs. Here, 3^4 = 81.
        self.n_rules = n_mfs ** self.n_inputs
        # For a first-order Takagi-Sugeno model: linear function of inputs plus bias → (n_inputs + 1) parameters per rule.
        self.n_rule_params = self.n_inputs + 1
        # Total number of rule parameters
        self.total_rule_params = self.n_rules * self.n_rule_params

    def triangular_mf(self, x, params):
        a, b, c = params
        if x <= a or x >= c:
            return 0.0
        if x <= b:
            return (x - a) / (b - a + 1e-6)
        else:
            return (c - x) / (c - b + 1e-6)

    def fuzzy_inference(self, state, individual):
        # Extract centers for each input.
        # For each of the 4 inputs, we have n_mfs centers.
        centers = []
        start = 0
        for i in range(self.n_inputs):
            centers_i = np.sort(individual[start:start+self.n_mfs])
            centers.append(centers_i)
            start += self.n_mfs
        # The remainder of the individual encodes the rule parameters.
        rule_params = individual[start:]
        
        # Build membership functions for each input.
        # Each input gets 3 MFs: left shoulder, middle, right shoulder.
        mfs = []
        for i in range(self.n_inputs):
            lower, upper = self.mf_bounds[i]
            c = centers[i]
            mf_i = [
                (lower, c[0], c[1]),   # left shoulder
                (c[0], c[1], c[2]),    # middle
                (c[1], c[2], upper)    # right shoulder
            ]
            mfs.append(mf_i)
        
        # Compute membership values for each input.
        membership_values = []
        for i in range(self.n_inputs):
            x_val = state[i]
            mf_vals = [self.triangular_mf(x_val, mf) for mf in mfs[i]]
            membership_values.append(mf_vals)
        
        # For each fuzzy rule (each combination of MFs), compute the activation.
        activations = []
        outputs = []
        rule_index = 0
        # Iterate over all combinations using np.ndindex.
        for idx in np.ndindex(*(self.n_mfs,)*self.n_inputs):
            # Activation is the minimum of the membership values across inputs.
            act = min(membership_values[i][idx[i]] for i in range(self.n_inputs))
            activations.append(act)
            # Extract corresponding rule parameters (each rule has n_inputs+1 parameters)
            params = rule_params[rule_index*self.n_rule_params:(rule_index+1)*self.n_rule_params]
            # Rule output: linear combination of inputs plus bias.
            output = np.dot(params[:self.n_inputs], state) + params[self.n_inputs]
            outputs.append(output)
            rule_index += 1
        
        total_activation = sum(activations) + 1e-6
        # Defuzzify using the weighted average of rule outputs.
        fuzzy_output = sum(act * out for act, out in zip(activations, outputs)) / total_activation
        return fuzzy_output

    def initialize_individual(self):
        # Generate centers for each input.
        centers = []
        for i in range(self.n_inputs):
            lower, upper = self.mf_bounds[i]
            centers_i = np.sort(np.random.uniform(lower, upper, self.n_mfs))
            centers.append(centers_i)
        centers = np.concatenate(centers)
        # Generate consequent (rule) parameters.
        rule_params = np.random.uniform(self.rule_bounds[0], self.rule_bounds[1], self.total_rule_params)
        return np.concatenate([centers, rule_params])

    def fitness(self, individual, render=False):
        # Evaluate the fuzzy controller on CartPole.
        # We run five episodes and average the rewards.
        env = gym.make("CartPole-v1", render_mode="human" if render else "rgb_array")
        total_reward = 0
        for _ in range(5):
            observation, info = env.reset()
            done = False
            while not done:
                # Compute fuzzy controller output given the state.
                fuzzy_output = self.fuzzy_inference(observation, individual)
                # Map the continuous output to a discrete action.
                # For example, if output > 0, push right; otherwise, push left.
                action = 1 if fuzzy_output > 0 else 0
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
        env.close()
        
        return total_reward / 5


    def tournament_selection(self, population, fitnesses):
        selected = []
        for _ in range(len(population)):
            candidates = np.random.choice(len(population), TOURNAMENT_SIZE, replace=False)
            best_idx = candidates[np.argmax(fitnesses[candidates])]
            selected.append(population[best_idx])
        return selected

    def crossover(self, parent1, parent2):
        child = np.empty_like(parent1)
        # Crossover for centers for each input.
        for i in range(self.n_inputs):
            start = i * self.n_mfs
            end = start + self.n_mfs
            split = np.random.randint(1, self.n_mfs)  # random split point (for n_mfs=3, split is 1 or 2)
            child[start:start+split] = parent1[start:start+split]
            child[start+split:end] = parent2[start+split:end]
            child[start:end] = np.sort(child[start:end])
        # Arithmetic crossover for rule parameters.
        rule_start = self.n_inputs * self.n_mfs
        p1_rules = parent1[rule_start:]
        p2_rules = parent2[rule_start:]
        alpha = np.random.rand(len(p1_rules))
        child[rule_start:] = alpha * p1_rules + (1 - alpha) * p2_rules
        return child

    def mutate(self, individual):
        mut_ind = individual.copy()
        # Mutate centers.
        for i in range(self.n_inputs):
            start = i * self.n_mfs
            end = start + self.n_mfs
            if np.random.rand() < MUT_PROB:
                idx = np.random.randint(self.n_mfs)
                lower, upper = self.mf_bounds[i]
                mut_ind[start+idx] += np.random.normal(0, 0.1)
                mut_ind[start:end] = np.sort(mut_ind[start:end])
                mut_ind[start:end] = np.clip(mut_ind[start:end], lower, upper)
        # Mutate rule parameters.
        rule_start = self.n_inputs * self.n_mfs
        rule_params = mut_ind[rule_start:]
        mask = np.random.rand(len(rule_params)) < MUT_PROB
        noise = np.random.normal(0, 0.1, len(rule_params))
        rule_params = np.clip(rule_params + mask * noise, self.rule_bounds[0], self.rule_bounds[1])
        mut_ind[rule_start:] = rule_params
        return mut_ind

    def run_evolution(self):
        population = [self.initialize_individual() for _ in range(POP_SIZE)]
        best_fitness_history = []
        for gen in range(N_GENERATIONS):
            fitnesses = np.array([self.fitness(individual) for individual in population])
            best_idx = np.argmax(fitnesses)
            best_fitness_history.append(fitnesses[best_idx])
            print(f"Gen {gen+1}: Best Fitness (reward) = {best_fitness_history[-1]:.4f}")
            selected = self.tournament_selection(population, fitnesses)
            offspring = []
            for i in range(0, len(population), 2):
                p1 = selected[i]
                p2 = selected[(i+1) % len(population)]
                if np.random.rand() < CX_PROB:
                    c1 = self.crossover(p1, p2)
                    c2 = self.crossover(p2, p1)
                    offspring.extend([c1, c2])
                else:
                    offspring.extend([p1, p2])
            population = [self.mutate(ind) for ind in offspring]
        # Demonstrate the best individual in a rendered episode.
        best_ind = population[np.argmax([self.fitness(ind) for ind in population])]
        demo_reward = self.fitness(best_ind, render=True)
        print(f"\nDemo run reward: {demo_reward}")
        # Plot evolution of best fitness over generations.
        plt.plot(best_fitness_history)
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness (reward)')
        plt.title('Evolution Progress')
        plt.show()

# To run the evolution:
if __name__ == '__main__':
    controller = GeneticFuzzyControllerCartPole(n_mfs=3)
    controller.run_evolution()
