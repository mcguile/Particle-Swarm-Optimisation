import numpy as np
import math
import time
from cec2005real.cec2005 import Function

"""
Algorithm that utilises a Genetic Algorithm to find a solution to a problem set.
In this case, CEC 2005 Benchmark mathemtical functions.
"""


class GeneticAlgorithm():
    """
    Algorithm to imitate evolutionary processes in biology. A population of
    individuals are randomly assigned positions in the search space. individuals
    are randomly selected to compete against each other for best fitness, then
    children are made by combining winners' traits (tournament selection and
    crossver). Mutation is carried out probabilistically to ensure diversity.
    """
    def __init__(self, pop_size, mutation_rate, func, dimen, bound_min, bound_max, k):
        self.pop_size = pop_size # Number of individuals
        self.mutation_rate = mutation_rate
        self.func = func # Benchmark function number 1 -> 25
        self.dimen = dimen # Dimensionality (10/ 30/ 50)
        self.bound_min = bound_min # boundaries of search space
        self.bound_max = bound_max
        self.k = k # Number of competitors in tournament selection

    def _initialize(self):
        """Generate random positions in search space"""
        self.population = np.random.uniform(self.bound_min, self.bound_max, (self.pop_size, self.dimen))

    def _calculate_fitness(self, func):
        """Return the fitness scores of full population"""
        pop_fitness = np.empty(self.pop_size)
        for i,individual in enumerate(self.population):
            fitness = func(individual)
            pop_fitness[i]=fitness
        return pop_fitness

    def _tournamentSelection(self, fitnesses):
        """Get the index of the winner of a tournament"""
        # Replace is possibility of choosing same parent twice
        competitors = np.random.choice(fitnesses,self.k,replace=False)
        winner = competitors.min() # Best score from K competitors
        return np.where(fitnesses==winner)[0][0]

    def _mutate(self, individual):
        """Mutate an element of an individual probablistically"""
        # vectorization approach - more memory req but faster than iterating
        # over each element in individual
        return np.where(np.random.random(len(individual)) < self.mutation_rate,
                    np.random.uniform(self.bound_min, self.bound_max, len(individual)),
                    individual)

    def _twoPointCrossover(self, parent1, parent2):
        """
        Two individuals, or parents, are passed in to breed. Two crossover
        points are selected at random and used to split each parent into three
        sections. The middle section of each is swapped to create two new
        children.
        """
        N = len(parent1)
        # choose two unique crossover points
        # no empty crossovers - i.e. at least one element per crossover point
        cx1 = np.random.randint(1,N)
        cx2 = np.random.randint(1,N)
        if cx1 == cx2:
            if cx1 == 1:
                cx2 += 1
            else:
                cx1 -= 1
        if cx2 < cx1:
            cx1,cx2 = cx2,cx1

        child1 = np.concatenate((parent1[:cx1], parent2[cx1:cx2], parent1[cx2:]))
        child2 = np.concatenate((parent2[:cx1], parent1[cx1:cx2], parent2[cx2:]))
        return child1, child2

    def main(self, iterations):
        self._initialize()

        fbench = Function(self.func,self.dimen)
        fitness_func = fbench.get_eval_function()
        min_fitness = None # Best fitness of all individuals

        for epoch in range(iterations):
            #Begin search for global optimum
            if epoch % 100 == 0 and epoch > 1:
               print("Epoch = " + str(epoch) +
               " best error = %s" % str(min_fitness))

            pop_fitnesses = self._calculate_fitness(fitness_func)
            min_fitness = pop_fitnesses.min()
            # Determine the next generation
            new_pop = np.empty(shape=(self.pop_size,self.dimen))
            for i in np.arange(0, self.pop_size, 2):
                idx1 = self._tournamentSelection(pop_fitnesses)
                idx2 = self._tournamentSelection(pop_fitnesses)
                # Perform crossover to produce children
                child1, child2 = self._twoPointCrossover(self.population[idx1],
                                                        self.population[idx2])
                # Save mutated children for next generation
                new_pop[i] = self._mutate(child1)
                new_pop[i+1] = self._mutate(child2)
            self.population = new_pop

        print (str(min_fitness) + ",")

if __name__ == "__main__":
    #np.random.seed(12)
    ga = GeneticAlgorithm(pop_size=100, mutation_rate=.01, func=4, dimen=10,
        bound_min=-100., bound_max=100., k=3)
    # Desired number of runs
    for i in range (1):
        start_time = time.time()
        ga.main(iterations=5000)
        print("Runtime: %s seconds" % (time.time() - start_time))
