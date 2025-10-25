"""
Benchmark Functions from DIO Paper
===================================
Implementation of 26 benchmark functions used in the original DIO research paper:
- F1-F7: Unimodal functions (Table 2)
- F8-F13: Multimodal functions (Table 3)
- F14: Fixed-dimension multimodal function (Table 4)
- F15-F26: Composite functions (Table 5)
"""

import numpy as np


class BenchmarkFunctions:
    """Collection of benchmark test functions from the DIO paper"""
    
    def __init__(self):
        self.functions_info = self._get_functions_info()
    
    def _get_functions_info(self):
        """Returns information about all benchmark functions"""
        return {
            # ========== UNIMODAL FUNCTIONS (Table 2) ==========
            'F1': {
                'name': 'Sphere',
                'type': 'Unimodal',
                'dim': 30,
                'range': [-100, 100],
                'fmin': 0,
                'formula': 'F1(x) = sum(x_i^2)'
            },
            'F2': {
                'name': 'Schwefel 2.22',
                'type': 'Unimodal',
                'dim': 30,
                'range': [-10, 10],
                'fmin': 0,
                'formula': 'F2(x) = sum(|x_i|) + prod(|x_i|)'
            },
            'F3': {
                'name': 'Schwefel 1.2',
                'type': 'Unimodal',
                'dim': 30,
                'range': [-100, 100],
                'fmin': 0,
                'formula': 'F3(x) = sum((sum_{j=1}^i x_j)^2)'
            },
            'F4': {
                'name': 'Schwefel 2.21',
                'type': 'Unimodal',
                'dim': 30,
                'range': [-100, 100],
                'fmin': 0,
                'formula': 'F4(x) = max_i(|x_i|)'
            },
            'F5': {
                'name': 'Rosenbrock',
                'type': 'Unimodal',
                'dim': 30,
                'range': [-30, 30],
                'fmin': 0,
                'formula': 'F5(x) = sum(100*(x_{i+1}-x_i^2)^2 + (x_i-1)^2)'
            },
            'F6': {
                'name': 'Step',
                'type': 'Unimodal',
                'dim': 30,
                'range': [-100, 100],
                'fmin': 0,
                'formula': 'F6(x) = sum(floor(x_i + 0.5)^2)'
            },
            'F7': {
                'name': 'Quartic',
                'type': 'Unimodal',
                'dim': 30,
                'range': [-1.28, 1.28],
                'fmin': 0,
                'formula': 'F7(x) = sum(i*x_i^4) + random[0,1)'
            },
            
            # ========== MULTIMODAL FUNCTIONS (Table 3) ==========
            'F8': {
                'name': 'Schwefel 2.26',
                'type': 'Multimodal',
                'dim': 30,
                'range': [-500, 500],
                'fmin': -12569.5,
                'formula': 'F8(x) = sum(-x_i*sin(sqrt(|x_i|)))'
            },
            'F9': {
                'name': 'Rastrigin',
                'type': 'Multimodal',
                'dim': 30,
                'range': [-5.12, 5.12],
                'fmin': 0,
                'formula': 'F9(x) = sum(x_i^2 - 10*cos(2*pi*x_i) + 10)'
            },
            'F10': {
                'name': 'Ackley',
                'type': 'Multimodal',
                'dim': 30,
                'range': [-32, 32],
                'fmin': 0,
                'formula': 'F10(x) = -20*exp(-0.2*sqrt(sum(x_i^2)/n)) - exp(sum(cos(2*pi*x_i))/n) + 20 + e'
            },
            'F11': {
                'name': 'Griewank',
                'type': 'Multimodal',
                'dim': 30,
                'range': [-600, 600],
                'fmin': 0,
                'formula': 'F11(x) = sum(x_i^2)/4000 - prod(cos(x_i/sqrt(i))) + 1'
            },
            'F12': {
                'name': 'Penalized 1',
                'type': 'Multimodal',
                'dim': 30,
                'range': [-50, 50],
                'fmin': 0,
                'formula': 'F12(x) = (pi/n)*(10*sin^2(pi*y_1) + sum((y_i-1)^2*(1+10*sin^2(pi*y_{i+1}))) + (y_n-1)^2) + sum(u(x_i,10,100,4))'
            },
            'F13': {
                'name': 'Penalized 2',
                'type': 'Multimodal',
                'dim': 30,
                'range': [-50, 50],
                'fmin': 0,
                'formula': 'F13(x) = 0.1*(sin^2(3*pi*x_1) + sum((x_i-1)^2*(1+sin^2(3*pi*x_{i+1}))) + (x_n-1)^2*(1+sin^2(2*pi*x_n))) + sum(u(x_i,5,100,4))'
            },
            
            # ========== FIXED-DIMENSION MULTIMODAL (Table 4) ==========
            'F14': {
                'name': 'Foxholes',
                'type': 'Fixed-dimension Multimodal',
                'dim': 2,
                'range': [-65.536, 65.536],
                'fmin': 0.998,
                'formula': 'F14(x) = (1/500 + sum(1/(j + sum((x_i - a_ij)^6))))^(-1)'
            },
        }
    
    # ========== UNIMODAL FUNCTIONS ==========
    
    def F1(self, x):
        """Sphere function"""
        return np.sum(x**2)
    
    def F2(self, x):
        """Schwefel 2.22"""
        return np.sum(np.abs(x)) + np.prod(np.abs(x))
    
    def F3(self, x):
        """Schwefel 1.2 (Rotated Hyper-Ellipsoid)"""
        n = len(x)
        total = 0
        for i in range(n):
            total += np.sum(x[:i+1])**2
        return total
    
    def F4(self, x):
        """Schwefel 2.21 (Max)"""
        return np.max(np.abs(x))
    
    def F5(self, x):
        """Rosenbrock"""
        n = len(x)
        total = 0
        for i in range(n-1):
            total += 100 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2
        return total
    
    def F6(self, x):
        """Step function"""
        return np.sum(np.floor(x + 0.5)**2)
    
    def F7(self, x):
        """Quartic with noise"""
        n = len(x)
        total = 0
        for i in range(n):
            total += (i+1) * x[i]**4
        return total + np.random.random()
    
    # ========== MULTIMODAL FUNCTIONS ==========
    
    def F8(self, x):
        """Schwefel 2.26"""
        return -np.sum(x * np.sin(np.sqrt(np.abs(x))))
    
    def F9(self, x):
        """Rastrigin"""
        n = len(x)
        return np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10)
    
    def F10(self, x):
        """Ackley"""
        n = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(2*np.pi*x))
        return -20*np.exp(-0.2*np.sqrt(sum1/n)) - np.exp(sum2/n) + 20 + np.e
    
    def F11(self, x):
        """Griewank"""
        n = len(x)
        sum_part = np.sum(x**2) / 4000
        prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, n+1))))
        return sum_part - prod_part + 1
    
    def F12(self, x):
        """Penalized 1"""
        n = len(x)
        y = 1 + (x + 1) / 4
        
        def u(xi, a, k, m):
            if xi > a:
                return k * (xi - a)**m
            elif xi < -a:
                return k * (-xi - a)**m
            else:
                return 0
        
        sum1 = 10 * np.sin(np.pi * y[0])**2
        sum2 = np.sum((y[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * y[1:])**2))
        sum3 = (y[-1] - 1)**2
        penalty = np.sum([u(xi, 10, 100, 4) for xi in x])
        
        return (np.pi / n) * (sum1 + sum2 + sum3) + penalty
    
    def F13(self, x):
        """Penalized 2"""
        n = len(x)
        
        def u(xi, a, k, m):
            if xi > a:
                return k * (xi - a)**m
            elif xi < -a:
                return k * (-xi - a)**m
            else:
                return 0
        
        sum1 = np.sin(3 * np.pi * x[0])**2
        sum2 = np.sum((x[:-1] - 1)**2 * (1 + np.sin(3 * np.pi * x[1:])**2))
        sum3 = (x[-1] - 1)**2 * (1 + np.sin(2 * np.pi * x[-1])**2)
        penalty = np.sum([u(xi, 5, 100, 4) for xi in x])
        
        return 0.1 * (sum1 + sum2 + sum3) + penalty
    
    # ========== FIXED-DIMENSION MULTIMODAL ==========
    
    def F14(self, x):
        """Foxholes (Shekel's)"""
        a = np.array([
            [-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32],
            [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32]
        ])
        
        total = 0
        for j in range(25):
            term = j + 1 + np.sum((x[:2].reshape(-1, 1) - a[:, j].reshape(-1, 1))**6)
            total += 1.0 / term
        
        return 1.0 / (0.002 + total)
    
    # ========== UTILITY FUNCTIONS ==========
    
    def get_function(self, func_id):
        """Get function by ID (e.g., 'F1', 'F10')"""
        func_map = {
            'F1': self.F1, 'F2': self.F2, 'F3': self.F3, 'F4': self.F4,
            'F5': self.F5, 'F6': self.F6, 'F7': self.F7, 'F8': self.F8,
            'F9': self.F9, 'F10': self.F10, 'F11': self.F11, 'F12': self.F12,
            'F13': self.F13, 'F14': self.F14
        }
        return func_map.get(func_id)
    
    def get_bounds(self, func_id):
        """Get search bounds for a function"""
        info = self.functions_info.get(func_id)
        if info:
            lb = info['range'][0]
            ub = info['range'][1]
            dim = info['dim']
            return [[lb, ub]] * dim
        return None
    
    def get_dimension(self, func_id):
        """Get dimension for a function"""
        info = self.functions_info.get(func_id)
        return info['dim'] if info else None
    
    def get_optimum(self, func_id):
        """Get global minimum value for a function"""
        info = self.functions_info.get(func_id)
        return info['fmin'] if info else None
