import numpy as np

class DIO:
    def __init__(self, objective_function, search_space, n_dholes, max_iterations):
        self.objective_function = objective_function
        self.search_space = np.array(search_space)
        self.n_dholes = n_dholes
        self.max_iterations = max_iterations
        self.n_dim = len(search_space)

        # Initialize the pack of dholes
        self.dholes = np.random.uniform(self.search_space[:, 0], self.search_space[:, 1], (self.n_dholes, self.n_dim))

        self.fitness = np.full(self.n_dholes, np.inf)
        self.alpha_dhole = np.zeros(self.n_dim)
        self.alpha_fitness = np.inf
        
        # Calculate initial fitness for each dhole
        for i in range(self.n_dholes):
            self.fitness[i] = self.objective_function(self.dholes[i])
        
        # Find the alpha dhole
        self.alpha_fitness = np.min(self.fitness)
        self.alpha_dhole = self.dholes[np.argmin(self.fitness)].copy()

    def optimize(self):
        for iteration in range(self.max_iterations):
            for i in range(self.n_dholes):
                # Update dhole's position using the DIO logic from the paper
                
                # Randomly select three dholes (can be the same as the current one)
                a, b, c = np.random.choice(self.n_dholes, 3)

                # Strategy 1: Chasing the alpha dhole
                r1 = np.random.random(self.n_dim)
                r2 = np.random.random(self.n_dim)
                A1 = 2 * r1 - 1
                C1 = 2 * r2
                D_alpha = np.abs(C1 * self.alpha_dhole - self.dholes[i])
                X1 = self.alpha_dhole - A1 * D_alpha

                # Strategy 2: Chasing a random dhole from the pack
                r3 = np.random.random(self.n_dim)
                r4 = np.random.random(self.n_dim)
                A2 = 2 * r3 - 1
                C2 = 2 * r4
                D_beta = np.abs(C2 * self.dholes[a] - self.dholes[i])
                X2 = self.dholes[a] - A2 * D_beta

                # Strategy 3: Pack cooperation
                m = (self.dholes[a] + self.dholes[b] + self.dholes[c]) / 3
                r5 = np.random.random(self.n_dim)
                r6 = np.random.random(self.n_dim)
                A3 = 2 * r5 - 1
                C3 = 2 * r6
                D_gamma = np.abs(C3 * m - self.dholes[i])
                X3 = m - A3 * D_gamma

                # Update position
                self.dholes[i] = (X1 + X2 + X3) / 3

                # Boundary checking
                self.dholes[i] = np.clip(self.dholes[i], self.search_space[:, 0], self.search_space[:, 1])

                # Fitness evaluation
                new_fitness = self.objective_function(self.dholes[i])
                if new_fitness < self.fitness[i]:
                    self.fitness[i] = new_fitness
                    if new_fitness < self.alpha_fitness:
                        self.alpha_fitness = new_fitness
                        self.alpha_dhole = self.dholes[i].copy()

            print(f"Iteration {iteration+1}/{self.max_iterations}, Best Fitness: {self.alpha_fitness}")

        return self.alpha_dhole, self.alpha_fitness
