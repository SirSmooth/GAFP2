import matplotlib.pyplot as plt
import numpy as np
from sympy import Range
np.random.seed(42)


# Definimos parametros para el MC
M_i = 111111111111  # Tones/hora
E_i = 24*np.sort(np.random.rand(100)*11111111111)  # kWh
eta1 = np.linspace(0.83, 0.79, 330)*0.007 # Porcentaje de recuperacion mineral
eta2 = np.linspace(0.71, 0.67, 308)*0.007 # Porcentaje de recuperacion mineral
eta3 = np.linspace(0.84, 0.8, 334)*0.007 # Porcentaje de recuperacion mineral
T_avg = 3.5*30  # Dias (promedio entre 3 y 4 meses)
T_cycle = np.sort(3*30 + np.random.rand(100)*30)  # Arreglo lineal entre 3 y 4 meses
T_rep = 3  # Dias
C_energy = 0.183  # $/kWh
omega = 10000  # $/tone
C_rep = 80000 + 400000  # Costo de la mano de obra + costo de los mill liners

# Inventados
rho = np.array(np.random.normal(8.95, 0.2, 3))  # Densidad g/cm³

# Calculados
rho_avg = np.average(rho)  # g/cm³
print('rho_avg: ', rho_avg)
'''
N = np.zeros(3)
for k in range(0, len(T_cycle)):
    N[k] = (T_cycle[k] + T_rep)/(T_avg + T_rep)
    k += 1
print('N: ', N)

C_DT = omega*M_i
print('C_DT: ', C_DT)

C_ins = 20000/11*np.linspace(0.33, 0.825, 350) # Inspeccion mensual
print('C_ins: [', C_ins[0]*30,', ', C_ins[349]*30,']')

P_gross = np.zeros(3)
P_gross1 = 0
P_gross2 = 0
P_gross3 = 0
for i in range(1, T_cycle[0]):
    P_gross1 = (P_gross1 + M_i*24*eta1[i]*omega - E_i[i]*C_energy - C_ins[i])
    i += 1
P_gross1 = (P_gross1 - C_DT - C_rep)*N[0]
for i in range(1, T_cycle[1]):
    P_gross2 = (P_gross1 + M_i*24*eta2[i]*omega - E_i[i]*C_energy - C_ins[i])
    i += 1
P_gross2 = (P_gross2 - C_DT - C_rep)*N[1]
for i in range(1, T_cycle[2]):
    P_gross3 = (P_gross1 + M_i*24*eta3[i]*omega - E_i[i]*C_energy - C_ins[i])
    i += 1
P_gross3 = (P_gross3 - C_DT - C_rep)*N[2]

P_gross[0] = P_gross1
P_gross[1] = P_gross2
P_gross[2] = P_gross3
print('P_gross :', P_gross)

P = np.zeros(3)
for i in range(0, 3):
    P[i] = M_i*24*0.007*0.81*T_cycle[i]*omega
    i += 1
print('P_j :', P)

P_total = np.sum(P)
print('P_total :', P_total)

w = P/P_total
print('w :', w)

T_cyc = np.zeros(3)
for i in range(0, 3):
    T_cyc[i] = rho_avg/rho[i]*T_avg
    i += 1
print('T_cyc :', T_cyc)

T_eff = sum(w*T_cyc)

print('T_eff :', T_eff)

# General
Promedio = sum(T_cyc)/3
print('Promedio: ', Promedio)
Desviacion = 0
for k in range(len(T_cyc)):
    Desviacion = Desviacion + (Promedio - T_cyc[k])**2
Desviacion = np.sqrt(Desviacion/3)
print('Desviacion: ', Desviacion)

dias = np.random.normal(T_eff, Desviacion, 1_000)

plt.figure(1)
plt.clf()

plt.hist(dias, bins=np.arange(315, 345, 0.5), density=True)

plt.xlabel('dias optimos para el reemplazo')
plt.ylabel('Frecuencia')

plt.legend()
plt.show()


# Mina A
Promedio = 330
print('Promedio: ', Promedio)
Desviacion = 0
for k in range(len(T_cyc)):
    Desviacion = Desviacion + (Promedio - T_cyc[k])**2
Desviacion = np.sqrt(Desviacion/3)
print('Desviacion: ', Desviacion)

dias = np.random.normal(Promedio, Desviacion, 1_000)

plt.figure(2)
plt.clf()

plt.hist(dias, bins=np.arange(320, 340, 0.5), density=True)

plt.xlabel('dias optimos para el reemplazo mina A')
plt.ylabel('Frecuencia')

plt.legend()
plt.show()


# Mina B
Promedio = 308
print('Promedio: ', Promedio)
Desviacion = 0
for k in range(len(T_cyc)):
    Desviacion = Desviacion + (Promedio - T_cyc[k])**2
Desviacion = np.sqrt(Desviacion/3)
print('Desviacion: ', Desviacion)

dias = np.random.normal(Promedio, Desviacion, 1_000)

plt.figure(3)
plt.clf()

plt.hist(dias, bins=np.arange(290, 330, 0.5), density=True)

plt.xlabel('dias optimos para el reemplazo mina B')
plt.ylabel('Frecuencia')

plt.legend()
plt.show()


# Mina C
Promedio = 334
print('Promedio: ', Promedio)
Desviacion = 0
for k in range(len(T_cyc)):
    Desviacion = Desviacion + (Promedio - T_cyc[k])**2
Desviacion = np.sqrt(Desviacion/3)
print('Desviacion: ', Desviacion)

dias = np.random.normal(Promedio, Desviacion, 1_000)

plt.figure(4)
plt.clf()

plt.hist(dias, bins=np.arange(320, 350, 0.5), density=True)

plt.xlabel('dias optimos para el reemplazo mina C')
plt.ylabel('Frecuencia')

plt.legend()
plt.show()

'''