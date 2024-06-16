import numpy as np
from math import sqrt, log
from statistics import median


def cauchy(x_0, gamma):
    r = np.random.uniform(0, 1)
    return np.tanh(np.pi * r - 0.5) * gamma + x_0

def jSO_algorithm(fitness_function, bounds):
    # Размерность
    D = len(bounds)

    # Настройки алгоритма
    G_max = 500 # Максимальное число поколений
    max_num_evaluations = D * 10000 # Максимальное число вычислений целевой функции
    init_pop_size = round(25 * log(D) * sqrt(D)) # Размер начальной популяции
    min_pop_size = 4 # Минимальный размер популяции
    memory_size = 5 # Размер памяти
    p_max = 0.25 # Максимальное значение параметра мутации
    p_min = p_max / 2 # Минимальное значение параметра мутации
    
    A = np.zeros([memory_size, D]) # Архив
    
    M_F = np.full(memory_size, 0.5) # Среднее значение параметра F (инициализируется 0.5)
    M_CR = np.full(memory_size, 0.8) # Среднее значение параметра CR (инициализируется 0.8)
 
    M_F_new = M_CR
    M_CR_new = M_CR
 
    nfes = 0 # Текущее число вычислений целевой функции
    k = 0 # index counter
    g = 0 # Текущая итерация
    p = p_max # Исходное значение p 
    
    pop = np.random.uniform(-100, 100, (init_pop_size, D))
    NP = len(pop)
    print(pop)
    # Сортировка индивидов (от лучшего к худшему, по функции принадлежности)
    fitness = np.array([fitness_function(ind) for ind in pop])
    if len(np.shape(fitness)) > 1:
        fitness = np.reshape(fitness, (NP,))
    sorted_indices = np.argsort(fitness)
    pop = pop[sorted_indices]
    
    while g < G_max:
        S_CR, S_F = [], [] # Неограничены
        delta_f = [] # Неограничен, как SF и SCR
        CR, F = np.zeros(NP), np.zeros(NP)
        
        u = np.zeros([NP, D])
        new_pop = np.zeros([NP, D])
        
        nfes = nfes + NP # !!!!!!!!!!!!!!!!!!!!!!
        
        for i in range(NP):
            r = np.random.randint(memory_size)
            if r == memory_size - 1:
                M_F[r], M_CR[r] = 0.9, 0.9
            
            if M_CR[r] < 0:
                CR[i] = 0
            else:
                CR[i] = np.random.normal(M_CR[r], 0.1)
            if g < 0.25 * G_max:
                CR[i] = max(CR[i], 0.7)
            elif g < 0.5 * G_max:
                CR[i] = max(CR[i], 0.6)
                
            F[i] = cauchy(M_F[r], 0.1)
            if g < 0.6 * G_max and F[i] > 0.7:
                F[i] = (0.7)
                
            # DE/current-to-pBest-w/1
            # f_w = F_i_g
            f_w = F[i] # !!!!!!!!!!!!!!!!!!!!
            if nfes < 0.2 *  max_num_evaluations:
                # f_w = 0.7 * F_i_g
                f_w = 0.7 * F[i]
            elif nfes < 0.4 *  max_num_evaluations:
                # f_w = 0.8 * F_i_g
                f_w = 0.8 * F[i]
            else:
                # f_w = 1.2 * F_i_g
                f_w = 1.2 * F[i]
                
            if p * NP > 1:
                p_best = np.random.randint(p * NP)
            else:
                p_best = 0    
            r1 = np.random.randint(NP)
            r2 = np.random.randint(NP + len(A))
            
            if r2 < NP:
                # v = pop[i] + f_w * (pop[p_best] - pop[i]) + F_i_g * (pop[r1] - pop[r2])
                v = pop[i] + f_w * (pop[p_best] - pop[i]) + F[i] * (pop[r1] - pop[r2])
            else:
                v = pop[i] + f_w * (pop[p_best] - pop[i]) + F[i] * (pop[r1] - A[r2 - NP])
            
            for j in range(D):
                rand_num = np.random.uniform(0, 1)
                j_rand = np.random.randint(NP)
                if rand_num <= CR[i] or j == j_rand:
                    u[i][j] = v[j]
                else:
                    u[i][j] = pop[i][j]
            # print(F[i])

        for i in range(NP):
            if fitness_function(u[i]) <= fitness_function(pop[i]):
                new_pop[i] = u[i]
            else:
                new_pop[i] = pop[i]
                
            delta_fit = fitness_function(u[i]) - fitness_function(pop[i])
            
            if fitness_function(u[i]) < fitness_function(pop[i]):
                A[k] = pop[i]
                S_CR.append(CR[i])
                S_F.append(F[i])
                delta_f.append(abs(delta_fit))
                # nfes = nfes + 2 # !!!!!!!!!!!!!!!!!!!!!!!!
                
            # Memory update in the iL-SHADE algorithm
            if len(S_CR) != 0 and len(S_F) != 0:
                # print(g, S_CR)
                if M_CR[k] == 1 or max(S_CR) == 0:
                    # print(max(S_CR))
                    M_CR[k] = 1 # !!!!
                else:
                    n_cr, d_cr = 0, 0 
                    for j in range(len(S_CR)):
                        w_k = delta_f[j]/sum(delta_f)
                        n_cr += w_k * S_CR[j]**2        
                        d_cr += w_k * S_CR[j]
                    mean_WL_S_CR = n_cr / d_cr    
                    M_CR_new[k] = (mean_WL_S_CR + M_CR[k]) / 2

                n_f, d_f = 0, 0
                for j in range(len(S_CR)):
                    w_k = delta_f[j]/sum(delta_f)
                    n_f += w_k * S_F[j]**2        
                    d_f += w_k * S_F[j]
                mean_WL_S_F = n_f / d_f    
                M_F_new[k] = (mean_WL_S_F + M_F[k]) / 2
                
                k += 1
                if k > memory_size - 1:
                    k = 0
            else:
                M_CR_new[k] = M_CR[k]
                M_F_new[k] = M_F[k]
                
        M_CR = M_CR_new
        M_F = M_F_new

        # print(M_CR)
        pop = new_pop.copy()
        
        # LPSR
        fitness = np.array([fitness_function(ind) for ind in pop])
        if len(np.shape(fitness)) > 1:
            fitness = np.reshape(fitness, (NP,))
        sorted_indices = np.argsort(fitness)
        pop = pop[sorted_indices]
        N_pop_size = round((min_pop_size - init_pop_size) / max_num_evaluations * nfes + init_pop_size)
        pop = pop[:N_pop_size, :D]
        
        best_idx = np.argmin(fitness)
        print(g, min(fitness))
        
        p = (p_min - p_max) /  max_num_evaluations * nfes + p_max
        NP = len(pop)
        g += 1
        
    return pop[best_idx]

def rastrigin(x):
    A = 10
    return A * len(x) + sum((xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x)

def run_rastrigin_test():
    # Define bounds for the variables
    D = 2  # Dimensionality of the problem
    bounds = [(-100, 100)] * D

    # Number of runs
    runs = 10
    results = []

    # Run the jSO algorithm 40 times and store the best results
    for _ in range(runs):
        best_solution = jSO_algorithm(rastrigin, bounds)
        best_value = rastrigin(best_solution)
        results.append(best_value)
        print(f"Запуск {_ + 1}: Значение функции = {best_value}")

    min_value = min(results)
    max_value = max(results)
    median_value = median(results)

    print("\nRastrigin Function")
    print("Minimum:", min_value)
    print("Median:", median_value)
    print("Maximum:", max_value)


if __name__ == "__main__":
    run_rastrigin_test()
