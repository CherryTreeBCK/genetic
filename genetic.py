from neural_network import NeuralNetwork
import numpy as np
import random

class GeneticAlgorithm:
    # Genetic Algorithm class
    # network_size: The size of the neural network
    # pop_size: Population size
    # mr: Mutation rate
    # dr: Decay rate - 0 means no decay, 1 means instant decay
    def __init__(self, network_size=[2, 3, 1], pop_size=100, mr=0.05, dr=0.001, cr=0.90, loss_fnc='log') -> None:
        self.population = []
        self.mr = mr # mutation rate
        self.dr = dr # decay rate
        self.cr = cr # crossover rate
        self.generation = 0
        self.loss_fnc = loss_fnc # loss function
        self.network_size = network_size
        
        for i in range(pop_size):
            self.population.append(NeuralNetwork(shape=network_size))
    
    # get current generation
    def get_generation(self):
        return self.generation
    
    # generates fitness scores for each agent
    def fitness(self, inputs, targets):
        fitness_scores = []
        for network in self.population:
            outputs = network.forward(inputs)
            fitness_score = 0
            if self.loss_fnc == 'log':
                loss = network.log_loss(outputs, targets)
                fitness_score = 1 / (loss + 1e-9)
            elif self.loss_fnc == 'acc':
                fitness_score = network.acc(outputs, targets)
            else:
                loss = network.mse_loss(outputs, targets)
                fitness_score = 1 / (loss + 1e-9)
            # Convert log loss to fitness score, where lower loss = higher fitness
            fitness_scores.append(fitness_score)
        return fitness_scores

    def mutation(self, network):
        cur_mr = self.mr * np.exp(-self.dr * self.generation)
        for layer in range(len(network.weights)):
            network.weights[layer] += (np.random.randn(*network.weights[layer].shape) * 0.5) * cur_mr
            network.bias[layer] += (np.random.randn(*network.bias[layer].shape) * 0.5) * cur_mr
            
    def crossover(self, parent1, parent2):
        """
        Performs a uniform crossover between two parent neural networks
        to produce an offspring neural network.

        Args:
            parent1 (NeuralNetwork): The first parent network.
            parent2 (NeuralNetwork): The second parent network.

        Returns:
            NeuralNetwork: The offspring network resulting from the crossover.
        """
        # Ensure both networks have the same structure
        assert len(parent1.weights) == len(parent2.weights), "Parents must have the same number of layers"
        assert all(p1w.shape == p2w.shape for p1w, p2w in zip(parent1.weights, parent2.weights)), "Parents must have matching layer shapes"
        
        offspring = NeuralNetwork([len(parent1.bias[0])])  # Dummy shape, will be overwritten
        offspring.weights = []
        offspring.bias = []
        
        for p1w, p2w, p1b, p2b in zip(parent1.weights, parent2.weights, parent1.bias, parent2.bias):
            # For weights
            mask_w = np.random.rand(*p1w.shape) > 0.5  # Generate a mask of booleans
            offspring_w = np.where(mask_w, p1w, p2w)  # Choose from p1w where mask is True, else from p2w
            
            # For biases
            mask_b = np.random.rand(*p1b.shape) > 0.5  # Similar mask for biases
            offspring_b = np.where(mask_b, p1b, p2b)
            
            offspring.weights.append(offspring_w)
            offspring.bias.append(offspring_b)
            
        return offspring

    
    def selection(self, fitness_scores, method="Roulette"):
        # Example of roulette selection process
        # [0.4, 0.1, 0.2, 0.3] -> [1]
        
        # rand(0, 1) = 0.25
        # 1. 0.25 - 0.1 = 0.15
        # 2. 0.15 - 0.2 = -0.05
        
        # rand(0, 1) = 0.6
        # 1. 0.6 - 0.1 = 0.5
        # 2. 0.5 - 0.2 = 0.3
        # 3. 0.3 - 0.3 = 0.0   
        
        # rand(0, 1) = 0.61
        # 1. 0.61 - 0.1 = 0.51
        # 2. 0.51 - 0.2 = 0.31
        # 3. 0.31 - 0.3 = 0.01
        # 4. 0.01 - 0.4 = -0.39
        
        if method=="roulette":
            return np.random.choice(self.population, p=fitness_scores).clone()
                
        # Otherwise return the best network
        return self.population[np.argmax(fitness_scores)].clone()
        
    
    def make_babies(self, inputs, targets):
        new_population = []
        fitness_scores = self.fitness(inputs, targets)
        # Elitism - ensure the next generation is at least as good
        new_population.append(self.selection(fitness_scores, method="Elitism"))
        
        # Roulette Fill the remaining population
        while len(new_population) < len(self.population):
            parent1 = self.selection(fitness_scores)
            parent2 = self.selection(fitness_scores)
            
            child = self.crossover( parent1, parent2)
            # prob = random.random()
            # if prob < self.cr / 2:
            #     child = self.crossover(parent1, parent2)
            # elif prob < self.cr:
            #     child = self.crossover(parent1, parent2)
            #     child.mutate()
            # else:
            #     child = NeuralNetwork(shape=self.network_size)
            self.mutation(child)
            
            new_population.append(child)
            
        self.population = new_population
        self.generation += 1
        
    def find_best(self, inputs, targets):
        fitness_scores = self.fitness(inputs, targets)
        return self.population[np.argmax(fitness_scores)]
