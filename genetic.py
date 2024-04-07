from neural_network import NeuralNetwork
import numpy as np
import random

class GeneticAlgorithm:
    # Genetic Algorithm class
    # network_size: The size of the neural network
    # mr: Mutation rate
    # pop_size: Population size
    # dr: Decay rate
    def __init__(self, network_size=[2, 3, 1], pop_size=100, mr=0.05, dr=0.001, loss_fnc='mse') -> None:
        self.population = []
        self.mr = mr
        self.dr = dr
        self.generation = 0
        self.loss_fnc = loss_fnc
        
        for i in range(pop_size):
            self.population.append(NeuralNetwork(shape=network_size))
        
    def get_generation(self):
        return self.generation
    
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
        # Elitism
        new_population.append(self.selection(fitness_scores, method="Elitism"))
        
        while len(new_population) < len(self.population):
            child = self.selection(fitness_scores)
            self.mutation(child)
            new_population.append(child)
            
        self.population = new_population
        self.generation += 1
        
    def find_best(self, inputs, targets):
        fitness_scores = self.fitness(inputs, targets)
        return self.population[np.argmax(fitness_scores)]
