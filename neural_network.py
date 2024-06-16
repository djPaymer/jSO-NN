import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras import models
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from ucimlrepo import fetch_ucirepo

concrete_compressive_strength = fetch_ucirepo(id=165)

X = concrete_compressive_strength.data.features
y = concrete_compressive_strength.data.targets

data_X = pd.DataFrame(X)
data_X = data_X.values

data_y = pd.DataFrame(y)
data_y = data_y.values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(data_X)

x_train, x_test, y_train, y_test = train_test_split(X_scaled, data_y)

model = models.Sequential()
model.add(layers.Dense(10, activation='tanh', input_shape=(8,)))
model.add(layers.Dense(10, activation='tanh'))
model.add(layers.Dense(1))

model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae']) # mae - mean absolute error

history = model.fit(x_train,
                    y_train,
                    epochs=1)

initial_weights = model.get_weights()

individ = []
for i in range(len(initial_weights)):
    for j in range(len(initial_weights[i])):
        if i % 2 == 0:
            for k in range(len(initial_weights[i][j])):
                individ.append(initial_weights[i][j][k])
        else:
            individ.append(initial_weights[i][j])

def cauchy(x_0, gamma):
    r = np.random.uniform(0, 1)
    return np.tanh(np.pi * r - 0.5) * gamma + x_0

def jSO_algorithm_1(fitness_function, bounds):
    # Размерность
    D = len(bounds)

    # Настройки алгоритма
    G_max = 30 # Максимальное число поколений
    max_num_evaluations = D * 10000 # Максимальное число вычислений целевой функции
    # init_pop_size = 10
    init_pop_size = 50 # Размер начальной популяции
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
    
    pop = np.random.uniform(-1, 1, (init_pop_size, D))    
    NP = len(pop)
    
    # Сортировка индивидов (от лучшего к худшему, по функции принадлежности)
    fitness_cache = {tuple(ind): fitness_function(ind) for ind in pop}
    fitness = np.array(list(fitness_cache.values()))
    sorted_indices = np.argsort(fitness)
    pop = pop[sorted_indices]
    
    print(1)
    
    while g < G_max:
        S_CR, S_F = [], [] # Неограничены
        delta_f = [] # Неограничен, как SF и SCR
        CR, F = np.zeros(NP), np.zeros(NP)
        
        u = np.zeros([NP, D])
        new_pop = np.zeros([NP, D])
        
        nfes += NP # Обновляем число вычислений целевой функции
        
        for i in range(NP):
            r = np.random.randint(memory_size)
            if r == memory_size - 1:
                M_F[r], M_CR[r] = 0.9, 0.9
            
            CR[i] = max(0, np.random.normal(M_CR[r], 0.1))
            if g < 0.25 * G_max:
                CR[i] = max(CR[i], 0.7)
            elif g < 0.5 * G_max:
                CR[i] = max(CR[i], 0.6)
                
            F[i] = cauchy(M_F[r], 0.1)
            if g < 0.6 * G_max and F[i] > 0.7:
                F[i] = 0.7
                
            # DE/current-to-pBest-w/1
            f_w = F[i] # !!!!
            if nfes < 0.2 * max_num_evaluations:
                f_w = 0.7 * F[i]
            elif nfes < 0.4 * max_num_evaluations:
                f_w = 0.8 * F[i]
            else:
                f_w = 1.2 * F[i]
                
            if p * NP > 1:
                p_best = np.random.randint(p * NP)
            else:
                p_best = 0    
            r1 = np.random.randint(NP)
            r2 = np.random.randint(NP + len(A))
            
            if r2 < NP:
                v = pop[i] + f_w * (pop[p_best] - pop[i]) + F[i] * (pop[r1] - pop[r2])
            else:
                v = pop[i] + f_w * (pop[p_best] - pop[i]) + F[i] * (pop[r1] - A[r2 - NP])
            
            for j in range(D):
                rand_num = np.random.uniform(0, 1)
                j_rand = np.random.randint(D)
                if rand_num <= CR[i] or j == j_rand:
                    u[i][j] = v[j]
                else:
                    u[i][j] = pop[i][j]

        for i in range(NP):
            u_tuple = tuple(u[i])
            if u_tuple not in fitness_cache:
                fitness_cache[u_tuple] = fitness_function(u[i])
            new_u_fitness = fitness_cache[u_tuple]
            
            pop_tuple = tuple(pop[i])
            if pop_tuple not in fitness_cache:
                fitness_cache[pop_tuple] = fitness_function(pop[i])
            old_u_fitness = fitness_cache[pop_tuple]
            
            if new_u_fitness <= old_u_fitness:
                new_pop[i] = u[i]
            else:
                new_pop[i] = pop[i]
                
            delta_fit = new_u_fitness - old_u_fitness
            
            if new_u_fitness < old_u_fitness:
                A[k % memory_size] = pop[i]
                S_CR.append(CR[i])
                S_F.append(F[i])
                delta_f.append(abs(delta_fit))
                
            if len(S_CR) != 0 and len(S_F) != 0:
                if M_CR[k] == 1 or max(S_CR) == 0:
                    M_CR[k] = 1
                else:
                    n_cr = sum((delta_f[j] / sum(delta_f)) * S_CR[j]**2 for j in range(len(S_CR)))
                    d_cr = sum((delta_f[j] / sum(delta_f)) * S_CR[j] for j in range(len(S_CR)))
                    mean_WL_S_CR = n_cr / d_cr    
                    M_CR_new[k] = (mean_WL_S_CR + M_CR[k]) / 2

                n_f = sum((delta_f[j] / sum(delta_f)) * S_F[j]**2 for j in range(len(S_F)))
                d_f = sum((delta_f[j] / sum(delta_f)) * S_F[j] for j in range(len(S_F)))
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

        pop = new_pop.copy()
        
        fitness = np.array([fitness_cache[tuple(ind)] for ind in pop])
        sorted_indices = np.argsort(fitness)
        pop = pop[sorted_indices]
        N_pop_size = round((min_pop_size - init_pop_size) / max_num_evaluations * nfes + init_pop_size)
        pop = pop[:N_pop_size, :D]
        
        best_idx = np.argmin(fitness)
        print(g, min(fitness))
        
        p = (p_min - p_max) / max_num_evaluations * nfes + p_max
        NP = len(pop)
        g += 1
        
    return pop[best_idx]

def loss(indiv):
    weights = initial_weights

    counter = 0
    for i in range(len(initial_weights)):
        for j in range(len(initial_weights[i])):
            if i % 2 == 0:
                for k in range(len(initial_weights[i][j])):
                    weights[i][j][k] = indiv[counter]
                    counter += 1
            else:
                weights[i][j] = indiv[counter]
                counter += 1

    model.set_weights(weights)
    history = model.fit(x_train,
                    y_train,
                    epochs=5,
                    verbose=0)

    return history.history['loss'][-1]

jSO_w = jSO_algorithm_1(loss, [-5, 5]*len(individ))

weights = initial_weights

counter = 0
for i in range(len(initial_weights)):
    for j in range(len(initial_weights[i])):
        if i % 2 == 0:
            for k in range(len(initial_weights[i][j])):
                weights[i][j][k] = jSO_w[counter]
                counter += 1
        else:
            weights[i][j] = jSO_w[counter]
            counter += 1

model.set_weights(weights)
history_with_jSO = model.fit(x_train,
                             y_train,
                             batch_size=13,
                             epochs=1000)

fig = plt.figure(figsize=(12,6))

loss_jSO = history_with_jSO.history['loss']
epochs = range(1, len(loss_jSO) + 1)

plt.plot(epochs, loss_jSO, 
         color='k', linestyle='-',
         label='Ошибка jSO + ИНС', zorder=3)

plt.grid(zorder=0)
plt.xlabel('$Эпохи$')
plt.ylabel('$MSE$')
plt.legend()

plt.show()
