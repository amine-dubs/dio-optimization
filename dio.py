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
        # History trackers
        convergence_curve = np.zeros(self.max_iterations)
        
        for t in range(self.max_iterations):
            # First, update fitness and leader based on current positions
            for i in range(self.n_dholes):
                # Boundary checking
                self.dholes[i] = np.clip(self.dholes[i], self.search_space[:, 0], self.search_space[:, 1])
                
                # Evaluate fitness
                fitness = self.objective_function(self.dholes[i])
                self.fitness[i] = fitness
                
                # Update alpha dhole (leader)
                if fitness < self.alpha_fitness:
                    self.alpha_fitness = fitness
                    self.alpha_dhole = self.dholes[i].copy()

            # Vocalization Influence
            V = 2 - t * (2 / self.max_iterations)
            
            # Determine phase based on MATLAB implementation (alternates every quarter)
            phase = 0 if (t // (self.max_iterations / 4)) % 2 == 0 else 1

            # Update positions based on phase
            for i in range(self.n_dholes):
                if phase == 0:  # Exploration phase
                    if np.random.rand() < 0.5:
                        # Randomly re-initialize position
                        self.dholes[i,:] = self.search_space[:, 0] + np.random.rand(self.n_dim) * (self.search_space[:, 1] - self.search_space[:, 0])
                    else:
                        # Follow lead vocalizer with added noise and neighbor influence
                        r = np.random.rand()
                        B = V * r**2 - V
                        C = r + np.sin(r * np.pi)
                        
                        D_lead = np.abs(C * self.alpha_dhole**2 - self.dholes[i,:]**2)
                        X_lead = self.alpha_dhole - B * np.sqrt(np.abs(D_lead))
                        
                        neighbors_influence = np.mean(self.dholes, axis=0) - self.dholes[i,:]
                        
                        w1 = np.random.rand()
                        w2 = 1 - w1
                        
                        noise = np.random.randn(self.n_dim) * (self.search_space[:, 1] - self.search_space[:, 0]) * 0.1
                        self.dholes[i,:] = w1 * X_lead + w2 * neighbors_influence + noise
                
                else:  # Exploitation phase
                    r = np.random.rand()
                    B = V * r**2 - V
                    C = r + np.sin(r * np.pi)
                    
                    D_lead = np.abs(C * self.alpha_dhole**2 - self.dholes[i,:]**2)
                    X_lead = self.alpha_dhole - B * np.sqrt(np.abs(D_lead))
                    
                    neighbors_influence = np.mean(self.dholes, axis=0) - self.dholes[i,:]
                    
                    w1 = np.random.rand()
                    w2 = 1 - w1
                    
                    self.dholes[i,:] = w1 * X_lead + w2 * neighbors_influence

            convergence_curve[t] = self.alpha_fitness
            print(f"Iteration {t+1}/{self.max_iterations}, Best Fitness: {self.alpha_fitness}")

        return self.alpha_dhole, self.alpha_fitness
