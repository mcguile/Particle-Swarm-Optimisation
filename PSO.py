import math
import numpy as np
import sys     # max float
import time
from cec2005real.cec2005 import Function;

"""
Algorithm that utilises Particle Swarms to find a solution to a problem set. In
this case, CEC 2005 Benchmark mathemtical functions.
"""


class Particle():
    """
    This class represents a particle object. The position, velocity,
    fitness, best previous position, best previous fitness, and informants
    make up the object.
    """

    def __init__(self, func, dim, bound_min, bound_max):
        self.func = func # benchmark function
        self.position = np.random.uniform(bound_min,bound_max,dim) # random pos
        self.velocity = np.zeros(dim) # Initial velocity is zero in all dims
        self.informants = None
        self.fitness = func(self.position) # Curr fitness
        self.best_part_pos = np.copy(self.position) # Prev best position
        self.best_part_fitness = self.fitness # Prev best fitness

    def _setFitness(self):
        """Change the particle's fitness after moving to new position"""
        self.fitness = self.func(self.position)

class ParticleSwarmOptimisation():
    """
    Algorithm that imitates biological swarm behaviour. Each particle has a
    group of informants that shares information on their best position in the
    search space, as well as receiving information of the global best position.
    The position of each particle is updated with a velocity of attraction
    towards its own previous best, its informants best, and the global best.
    """

    def _getInformantBestPos(self,particle, swarm):
        """ Return the index in the swam of the best position
        given by a particle's informants
        """
        best_fitness = sys.float_info.max
        best_pos = None
        for i in particle.informants:
            if best_fitness > swarm[i].fitness:
                best_fitness = swarm[i].fitness
                best_pos = swarm[i].position
        return best_pos

    def main(self,iterations, n, func, dim, bound_min, bound_max):
        # Create n random particles
        swarm = [Particle(func, dim, bound_min, bound_max) for i in range(n)]
        # Create groups of informants and apply them to particles
        # No particle shares informants i.e. no overlap
        idx = [i for i in range(len(swarm))]
        np.random.shuffle(idx)
        groups = [idx[x:x+2] for x in range(0,len(idx),2)]
        for group in groups:
            for informant in group:
                swarm[informant].informants = np.array(group)

        best_swarm_fitness = sys.float_info.max # Set to max as opt fitness = 0
        for i in range(n): # Check each particle
            if swarm[i].fitness < best_swarm_fitness:
                # Found new best global fitness
                best_swarm_fitness = swarm[i].fitness
                best_swarm_pos = np.copy(swarm[i].position)

        alpha = 0.7 # Inertia
        beta = 2 # Cognitive (particle)
        gamma = 1.5 # Social (global best)
        delta = 0.5 # Soial (informants)

        for epoch in range(iterations):
            # Begin search for global optimum
            if epoch % 100 == 0 and epoch > 1:
                print("Epoch = " + str(epoch) +
                    " best fitness = %f" % best_swarm_fitness)

            for i in range(n): # Process each particle
                best_inf_pos = self._getInformantBestPos(swarm[i], swarm)
                # Update velocity of curr particle in every dimension
                for k in range(dim):
                    r1 = np.random.random() # velocity coefficients
                    r2 = np.random.random()
                    r3 = np.random.random()

                    swarm[i].velocity[k] = ((alpha * swarm[i].velocity[k]) +
                        (beta * r1 * (swarm[i].best_part_pos[k] -
                            swarm[i].position[k])) +
                        (gamma * r2 * (best_inf_pos[k] -
                            swarm[i].position[k])) +
                        (delta * r3 * (best_swarm_pos[k] -
                            swarm[i].position[k])))

                    # Ensure swarm stays within bounds
                    if swarm[i].velocity[k] < bound_min:
                        swarm[i].velocity[k] = bound_min
                    elif swarm[i].velocity[k] > bound_max:
                        swarm[i].velocity[k] = bound_max

                # Compute new position using new velocity
                for k in range(dim):
                    swarm[i].position[k] += swarm[i].velocity[k]

                # Compute fitness of new position
                swarm[i]._setFitness()

                # Is new position a new best for the particle?
                if swarm[i].fitness < swarm[i].best_part_fitness:
                    swarm[i].best_part_fitness = swarm[i].fitness
                    swarm[i].best_part_pos = np.copy(swarm[i].position)

                # Is new position a new best overall?
                if swarm[i].fitness < best_swarm_fitness:
                    best_swarm_fitness = swarm[i].fitness
                    best_swarm_pos = np.copy(swarm[i].position)

        print ("Best fitness:" + str(best_swarm_fitness))

if __name__ == "__main__":
    #np.random.seed(8)
    pso = ParticleSwarmOptimisation()
    # Desired number of runs
    for i in range(1):
        start_time = time.time()
        dim=10  # Dimensionality (10, 30, or 50)
        fbench = Function(4,dim)    # Function number 1-> 25
        fitnessFunc = fbench.get_eval_function()
        pso.main(iterations=10000, n=50, func=fitnessFunc, dim=dim,
            bound_min=-100., bound_max=100.)
        print("Runtime: %s seconds" % (time.time() - start_time))
