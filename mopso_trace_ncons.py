import numpy as np
import time
import copy as c
from ypstruct import structure
import random

global par
parpso = structure()

########
parpso.no_of_objfun1 = 2  # For dominance level 1
parpso.no_of_objfun2 = 3  # For dominance level 2
parpso.wmax = 0.8  # inertia weight 1
parpso.wmin = 0.5  # inertia weight 2
parpso.ca1 = 1.7  # acceleration factor 1
parpso.ca2 = 1.7  # acceleration factor 2


def generate_particles():  # To generate initial population
    np.random.seed()
    xl = 0
    xu = 1
    widths = np.random.rand(parpso.n, parpso.m)
    for i in range(parpso.n):
        for k in range(parpso.m):
            widths[i][k] = xl + (xu - xl) * widths[i][k]
        widths[i] = [round(x, 2) for x in widths[i]]
    widths = widths.tolist()
    return widths


def cal_fitness_level1(Xpop, fitness1, fitness2, gpinfo_cons, n_cons):
    X = c.deepcopy(Xpop)
    X = np.array(X)
    f1 = fitness1(X, gpinfo_cons, n_cons)
    f2 = fitness2(X, gpinfo_cons, n_cons)
    F = np.column_stack([f1, f2])
    return F


def cal_fitness_level2(Xpop, fitness1, fitness2, fitness3, y, gpinfo):
    X = c.deepcopy(Xpop)
    X = np.array(X)
    f1 = fitness1(X, gpinfo)
    f2 = fitness2(X, y, gpinfo)
    f3 = fitness3(X, y, gpinfo)
    F = np.column_stack([f1, f2, f3])
    return F


def cal_rankings(X, fitness1, fitness2, fitness3, fitness4, fitness5, y, gpinfo, gpinfo_cons, n_cons):
    rank1 = calrank_level1(X, fitness4, fitness5, gpinfo_cons, n_cons)
    rank2 = calrank_level2(X, fitness1, fitness2, fitness3, y, gpinfo)
    R = np.column_stack([rank1, rank2])
    return R


def calrank_level1(Xpop, fitness1, fitness2, gpinfo_cons, n_cons):
    X = c.deepcopy(Xpop)
    index_history = []
    for i in range(len(X)):
        index_history.append(i)
    index_rank = [None] * len(X)
    X_fitness = cal_fitness_level1(X, fitness1, fitness2, gpinfo_cons, n_cons)
    X_fitness = X_fitness.tolist()
    rankvalue = 0
    archive_size = len(X)
    while archive_size > 0:
        paretofront = []
        paretofront_fit = []
        paretofront_size = 0
        paretofront_index = []
        for i in range(len(X)):
            if paretofront_size == 0:
                # insert nondominated solution from X into paretofront",
                paretofront.append(X[i])
                paretofront_fit.append(X_fitness[i])
                paretofront_size += 1
                paretofront_index.append(i)
            else:
                flag = 0
                k = 0
                while k < paretofront_size:
                    count1 = 0
                    count2 = 0
                    for j in range(parpso.no_of_objfun1):
                        if X_fitness[i][j] <= paretofront_fit[k][j]:
                            count1 += 1
                    for j in range(parpso.no_of_objfun1):
                        if X_fitness[i][j] == paretofront_fit[k][j]:
                            count2 += 1
                    if count1 == count2 and count2 != parpso.no_of_objfun1:
                        flag = 1
                        break
                    if count1 == parpso.no_of_objfun1:
                        paretofront.pop(k)
                        paretofront_fit.pop(k)
                        paretofront_index.pop(k)
                        paretofront_size -= 1
                    elif count1 == 0:
                        flag = 1
                        break
                    k += 1
                if flag == 0:
                    paretofront.append(X[i])
                    paretofront_fit.append(X_fitness[i])
                    paretofront_index.append(i)
                    paretofront_size += 1
        for b in range(len(paretofront_index)):
            original_index = index_history[paretofront_index[b]]
            index_rank[original_index] = rankvalue
        paretofront_index.sort(reverse=True)
        for b in range(len(paretofront_index)):
            X.pop(paretofront_index[b])
            X_fitness.pop(paretofront_index[b])
            index_history.pop(paretofront_index[b])
        rankvalue += 1
        archive_size = len(X)
    return index_rank


def calrank_level2(Xpop, fitness1, fitness2, fitness3, y, gpinfo):
    X = c.deepcopy(Xpop)
    index_history = []
    for i in range(len(X)):
        index_history.append(i)
    index_rank = [None] * len(X)
    X_fitness = cal_fitness_level2(X, fitness1, fitness2, fitness3, y, gpinfo)
    X_fitness = X_fitness.tolist()
    rankvalue = 0
    archive_size = len(X)
    while archive_size > 0:
        paretofront = []
        paretofront_fit = []
        paretofront_size = 0
        paretofront_index = []
        for i in range(len(X)):
            if paretofront_size == 0:
                # insert nondominated solution from X into paretofront",
                paretofront.append(X[i])
                paretofront_fit.append(X_fitness[i])
                paretofront_size += 1
                paretofront_index.append(i)
            else:
                flag = 0
                k = 0
                while k < paretofront_size:
                    count1 = 0
                    count2 = 0
                    for j in range(parpso.no_of_objfun2):
                        if X_fitness[i][j] <= paretofront_fit[k][j]:
                            count1 += 1
                    for j in range(parpso.no_of_objfun2):
                        if X_fitness[i][j] == paretofront_fit[k][j]:
                            count2 += 1
                    if count1 == count2 and count2 != parpso.no_of_objfun2:
                        flag = 1
                        break
                    if count1 == parpso.no_of_objfun:
                        paretofront.pop(k)
                        paretofront_fit.pop(k)
                        paretofront_size -= 1
                    elif count1 == 0:
                        flag = 1
                        break
                    k += 1
                if flag == 0:
                    paretofront.append(X[i])
                    paretofront_fit.append(X_fitness[i])
                    paretofront_index.append(i)
                    paretofront_size += 1
        for b in range(len(paretofront_index)):
            original_index = index_history[paretofront_index[b]]
            index_rank[original_index] = rankvalue
        paretofront_index.sort(reverse=True)
        for b in range(len(paretofront_index)):
            X.pop(paretofront_index[b])
            X_fitness.pop(paretofront_index[b])
            index_history.pop(paretofront_index[b])
        rankvalue += 1
        archive_size = len(X)
    return index_rank


def ranking_sort(X, rank):
    size = len(X)
    rank_len = len(rank[0])
    X_sort = c.deepcopy(X)
    r = c.deepcopy(rank)
    r = r.tolist()
    for i in range(size):
        X_sort[i] = X_sort[i] + r[i]
    X_new = sorted(X_sort, key=lambda x: (x[parpso.m], x[parpso.m + 1]))
    gbest = []
    g = 1
    gbest.append(X_new[0])
    while g < size:
        if (X_new[g][parpso.m] <= gbest[0][parpso.m]) and (X_new[g][parpso.m + 1] <= gbest[0][parpso.m + 1]):
            gbest.append(X_new[g])
            g += 1
        else:
            break
    gbest_pos = []
    gbest_rank = []
    for g in range(len(gbest)):
        gbest_pos.append(X_new[g][:parpso.m])
        gbest_rank.append(X_new[g][-rank_len:])
    return gbest_pos


def velocity_update(population, velocity, pbest_pos, w, gbest_pos):
    vel = c.deepcopy(velocity)
    pop = c.deepcopy(population)
    pbest = c.deepcopy(pbest_pos)
    for index in range(parpso.n):
        choice = np.random.randint(0, len(gbest_pos))
        gbest = c.deepcopy(gbest_pos[choice])
        for j in range(parpso.m):
            vel[index][j] = ((w * vel[index][j]) + (
                    parpso.ca1 * (pbest[index][j] - pop[index][j]) * np.random.uniform(0, 1, 1)[0])
                             + (parpso.ca2 * np.random.uniform(0, 1, 1)[0] * (gbest[j] - pop[index][j])))
            pop[index][j] = pop[index][j] + vel[index][j]
            if pop[index][j] > 1:
                pop[index][j] = random.random()
            elif pop[index][j] < 0:
                pop[index][j] = random.random()
    pop = np.array(pop)
    pop = np.round(pop, 4)
    pop = pop.tolist()
    return pop, vel


def update_pbestgbest(pop, pbest, gbest, fitness1, fitness2, fitness3, fitness4, fitness5,
                                   y, gpinfo, gpinfo_cons, n_cons):
    x_best = pop + pbest + gbest
    R = cal_rankings(x_best, fitness1, fitness2, fitness3, fitness4, fitness5, y, gpinfo, gpinfo_cons, n_cons)
    for i in range(parpso.n):
        flag = 0
        if R[i][0] < R[i + parpso.n][0] or ((R[i][0] == R[i+parpso.n][0]) and (R[i][1] <= R[i + parpso.n][1])):
            flag = 1
        if flag == 1:
            pbest[i] = pop[i]
    x_gbest = pop + gbest
    R1 = R[:len(pop)]
    #R2 = R[-len(pbest):]
    R3 = R[-len(gbest):]
    R4 = np.concatenate([R1, R3])
    gbest = ranking_sort(x_gbest, R4)
    return gbest, pbest


def mopso(decvar_dim, fitness1, fitness2, fitness3, fitness4, fitness5, y, gpinfo, yc, gpinfo_cons, n_cons, swarmsize,
          maxiter):
    parpso.n = swarmsize
    parpso.maxiter = maxiter
    parpso.m = decvar_dim
    population = np.random.rand(parpso.n, parpso.m)
    population = np.round(population, 4)
    population = population.tolist()
    pbest= c.deepcopy(population)
    velocity = [[[] for j in range(parpso.m)] for i in range(parpso.n)]
    for i in range(parpso.n):
        velocity[i] = c.deepcopy((np.array(population[i]) * 0.1)).tolist()
    pop_allrank = cal_rankings(population, fitness1, fitness2, fitness3, fitness4, fitness5, y, gpinfo, gpinfo_cons, n_cons)
    gbest = ranking_sort(population, pop_allrank)
    for ite in range(maxiter):
        w = parpso.wmax - (parpso.wmax - parpso.wmin) * ite / maxiter
        population, velocity = velocity_update(population, velocity, pbest, w, gbest)
        gbest, pbest = update_pbestgbest(population, pbest, gbest, fitness1, fitness2, fitness3, fitness4, fitness5,
                                   y, gpinfo, gpinfo_cons, n_cons)
        # print("done in " + str(round((time.time() - lasttime), 2)))
    choice_indices = np.random.choice(len(gbest), 1, replace=False)
    choices = gbest[choice_indices[0]]
    choices = np.array(choices)
    return choices


if __name__ == '__main__':
    start = time.time()
    print("done in " + str(round((time.time()), 2)))
