from matplotlib import pyplot as pl
import numpy as np
from operator import itemgetter
from content import get_size, linear_gaussian_kernel, mask_neighborhood, mask_bounds, generate_random_vertices, generate_circle_vertices


class TravelingSalesmanALl:
    def __init__(self, vertices: list) -> None:
        self.vertices = vertices
        self.order = []

    
    def main_loop(self):
        left = range(self.vertices.__len__())

        pass

    def get_best_points(self):
        best = self.order
        pts_x = []
        pts_y = []
        for i in best:
            pts_x.append(self.vertices[i][0])
            pts_y.append(self.vertices[i][1])
        return [pts_x, pts_y]


class TravelingSalesmanGenetic:
    def __init__(self, vertices: list, popsize: int, stepsmax: int, eta: float, keep_on: int, signalive = 10) -> None:
        self.vertices = vertices
        self.stepsmax = stepsmax
        self.eta = eta
        self.keep_on = keep_on
        self.signalalive = signalive
        # TODO add list of high scores
        self.best_histogram = []

        self.population = []
        vert_indices = range(0,vertices.__len__())
        for _ in range(popsize):
            chromosome = np.random.choice(vert_indices, size=vertices.__len__(), replace=False)
            self.population.append((self.__evaluate(chromosome), chromosome))
        self.population.sort(key=itemgetter(0))
        self.best_histogram.append(self.population[0])
        
        self.__GENE_INDICES__ = np.array(range(vertices.__len__()), dtype=np.int32)
        self.__GENE_PROPABI__ = linear_gaussian_kernel(vertices.__len__(), vertices.__len__())
        self.__POPU_INDICES__ = np.array(range(self.population.__len__()), dtype=np.int16)

        self.__STOP_COUNTER__ = 0
    

    def main_loop(self):
        for iteration in range(self.stepsmax):

            new_population = []

            # Selection by chromosome quality
            # Elitism - keep 2 best
            new_population.append(self.population[0])
            new_population.append(self.population[1])

            # Propability distribution calculation
            pmax = max(self.population, key=itemgetter(0))[0]
            pmin = min(self.population, key=itemgetter(0))[0]
            normalized = [(1.01-((el[0]-pmin)/(pmax-pmin))) for el in self.population]
            n_sum = sum(normalized)
            p = [(nu/n_sum) for nu in normalized]
            # Selection of chromosome indices with calculated propability
            crossovers = np.random.choice(self.__POPU_INDICES__, p=p, replace=False, size=self.population.__len__() - 2)

            # Crossover via a variant of Partially Mapped Crossover (PMX)
            for i in range(0, crossovers.__len__(), 2):
                [c1, c2] = self.__cross(self.population[crossovers[i]][1], self.population[crossovers[i+1]][1])
                new_population.append((self.__evaluate(c1), c1))
                new_population.append((self.__evaluate(c2), c2))
                pass

            # Mutation: Choose and mutate 1 chromosome from population
            mutated_index = np.random.choice(self.__POPU_INDICES__, size=2)
            mutation1 = self.__mutate(new_population[mutated_index[0]][1])
            new_population[mutated_index[0]] = ( self.__evaluate(mutation1), mutation1 )
            mutation2 = self.__mutate(new_population[mutated_index[1]][1])
            new_population[mutated_index[1]] = ( self.__evaluate(mutation2), mutation2 )

            # Swap to new population
            self.population = new_population
            
            # Sort population for next iteration
            self.population.sort(key=itemgetter(0))
            currentbest = min(self.population, key=itemgetter(0))

            # Write iteration stats stats to console
            if (iteration % self.signalalive == 0):
                print(f"Iteration {iteration: >8}\tcurrent best: {currentbest[0]: >10.2f}")

            self.best_histogram.append(currentbest)

            # Exit if change eta reached
            #print(f"{iteration} {currentbest[0]} {self.best_histogram[-1][0]} {currentbest[0] - self.best_histogram[-1][0]}")
            #print(self.best_histogram[-1][0])
            if self.eta > 0 and self.best_histogram[-2][0] - currentbest[0] < self.eta:
                self.__STOP_COUNTER__ += 1
            else:
                self.__STOP_COUNTER__ = 0

            if (self.__STOP_COUNTER__ > self.keep_on):
                print(f"For {self.keep_on} generations improvement was less than eta: {self.eta}, stopping...")
                break
        pass

    def __evaluate(self, chromosome: list):
        # Include start to end
        v0 = np.array(self.vertices[chromosome[0]])
        v1 = np.array(self.vertices[chromosome[-1]])
        sum = np.sqrt(pow(v0[0]-v1[0], 2) + pow(v0[1]-v1[1], 2))
        # Calculate all other pairs
        for i in range(chromosome.__len__()-1):
            v0 = np.array(self.vertices[chromosome[i]])
            v1 = np.array(self.vertices[chromosome[i+1]])
            sum += np.sqrt(pow(v0[0]-v1[0], 2) + pow(v0[1]-v1[1], 2))
        return sum

    
    def __cross(self, parent1: list, parent2: list):
        index1 = np.random.choice(self.__GENE_INDICES__, p=self.__GENE_PROPABI__)
        mask = mask_neighborhood(self.__GENE_INDICES__.__len__(), index1, 1)
        indices_no_neighborhood = np.delete(self.__GENE_INDICES__, mask)
        propabi_no_neighborhood = np.delete(self.__GENE_PROPABI__, mask)
        propabi_no_neighborhood = propabi_no_neighborhood / sum(propabi_no_neighborhood)

        indices = [index1, np.random.choice(indices_no_neighborhood, p=propabi_no_neighborhood)]
        indices.sort()

        [m_inner, m_outer] = mask_bounds(self.__GENE_INDICES__.__len__(), indices[0], indices[1])
        # Child 1: keep m_inner region, fill with outer by occurence
        # Child 2: keep m_outer region, fill with inner by occurence
        tmp1 = np.delete(parent1, m_inner)
        left1 = [x for x in parent2 if x in tmp1]
        tmp2 = np.delete(parent2, m_outer)
        left2 = [x for x in parent1 if x in tmp2]

        child1 = []; child2 = []

        # TODO
        for i in self.__GENE_INDICES__:
            # print(f"i={i: >3}, left len{left1.__len__(): > 4}")
            if m_inner[i]:
                child1.append(parent1[i])

                child2.append(left2[0])
                left2 = np.delete(left2, 0)
            else:
                child1.append(left1[0])
                left1 = np.delete(left1, 0)

                child2.append(parent2[i])
            pass

        # print(f"{'Parent 1': <10}", end="")
        # for v in parent1:
        #     print(f"{v: >3}", end=" ")
        # print("")
        # print(f"{'Parent 1': <10}", end="")
        # for v in parent2:
        #     print(f"{v: >3}", end=" ")
        # print("")
        # print(f"{'Child 1': <10}", end="")
        # for v in child1:
        #     print(f"{v: >3}", end=" ")
        # print("")
        # print(f"{'Child 2': <10}", end="")
        # for v in child2:
        #     print(f"{v: >3}", end=" ")
        # print("")
        # exit()

        return [child1, child2]
    

    def __mutate(self, chromosome):
        
        # print("")
        # print(f"{'Before': <10}", end="")
        # for v in chromosome:
        #     print(f"{v: >3}", end=" ")

        swaps = np.random.choice(self.__GENE_INDICES__, size=2, replace=False)
        gene = chromosome[swaps[0]]
        chromosome[swaps[0]] = chromosome[swaps[1]]
        chromosome[swaps[1]] = gene

        # print("")
        # print(f"{'After': <10}", end="")
        # for v in chromosome:
        #     print(f"{v: >3}", end=" ")
        # exit()

        return chromosome
        

    def test_eval(self):
        for c in self.population:
            print(self.__evaluate(c[1]))

    
    def dump_to_files(self):
        from datetime import datetime as dt
        path = "./results/r_" + dt.now().strftime("%Y-%d-%m_%H-%M-%S") + "_"
        best_approx_path = min(self.best_histogram, key=itemgetter(0))
        with open(path + "vertices.csv", "w") as f:
            for v in self.vertices:
                f.write(f"{v[0]};{v[1]}\n")
        with open(path + "approx_best_path.csv", "w") as f:
            for v in best_approx_path[1]:
                f.write(f"{v}\n")
        with open(path + "histogram.csv", "w") as f:
            for i,d in enumerate(self.best_histogram):
                f.write(f"{i};{d[0]};{d[1]}\n")
        with open(path + "summary.txt", "w") as f:
            f.write(f"### ITERATIONS: {self.best_histogram.__len__()} ###\n")
            f.write(f"### BEST PATH {best_approx_path[0]} ###\n")
            string = ""
            for d in best_approx_path[1]:
                string += f"{d};"
            f.write(string)
            f.write("\n")


    def __str__(self) -> str:
        str =   f"""TravelingSalesmanGenetic
                    stepsmax: {self.stepsmax}
                    eta: {self.eta}
                    signal alive every {self.signalalive} iterations

                    VERTICES:
                    """.replace("                    ", "")
        for i, v in enumerate(self.vertices):
            str += f"{i}: {v[0]}, {v[1]}\n"

        str += "\n\nPOPULATION\n"
        for i, c in enumerate(self.population):
            str += f"{i}: "
            for g in c[1]:
                str += f"{g} "
            str += "\n"
        return str + "\n"
    
    def get_best(self):
        return self.best_histogram[-1]
    

    def get_best_points(self):
        best = min(self.best_histogram, key=itemgetter(0))[1]
        pts_x = []
        pts_y = []
        for i in best:
            pts_x.append(self.vertices[i][0])
            pts_y.append(self.vertices[i][1])
        return [pts_x, pts_y]


def main():
    vertices = generate_random_vertices(20)
    # vertices = generate_circle_vertices(10, 100, (0,0))
    a = TravelingSalesmanGenetic(vertices=vertices, popsize=400, stepsmax=5000, eta=5e-5, keep_on=200, signalive=10)
    a.main_loop()
    print(get_size(a))
    a.dump_to_files()

    print(a.get_best())
    [bx, by] = a.get_best_points()
    fig = pl.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')
    ax.plot(bx, by, 'b')
    ax.plot(bx[0], by[0], 'go')
    pl.show()
    pass


if __name__ == "__main__":
    main()