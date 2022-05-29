import matplotlib.pyplot as plt
import numpy as np
from sympy import Range
np.random.seed(42)


# Definimos parametros para el MC
M_i = 6000  # Tones/hora
E_i = 24*np.sort(np.random.rand(4*30)*745.7*400*6)  # kWh
T_avg = 3.5*30  # Dias (promedio entre 3 y 4 meses)
T_cycle = np.sort(np.random.randint(3*30, 4*30, 100))  # Arreglo lineal entre 3 y 4 meses

# Eta (Porcentaje de recuperacion mineral)
ley = 0.007
r_i = np.sort(np.random.normal(0.8, 0.01, len(T_cycle))) # Recuperacion inicial de material por eficiencia de la maquina
r_f = np.sort(np.random.normal(0.7, 0.01, len(T_cycle)))  # Recuperacion final de material por eficiencia de la maquina
eta = np.zeros(((len(T_cycle), (T_cycle[len(T_cycle)-1]))))

k = 0
for i in range(len(T_cycle)*(T_cycle[len(T_cycle)-1])):
    if i < T_cycle[len(T_cycle)-1] + T_cycle[len(T_cycle)-1]*k:
        if k == 0:
            xrx = np.linspace(r_i[k], r_f[k], T_cycle[len(T_cycle)-1])*ley  # Usamos el dato maximo para la iteracion
            eta[k, i] = xrx[i]
        else:
            xrx = np.linspace(r_i[k], r_f[k], T_cycle[len(T_cycle)-1])*ley  # Usamos el dato maximo para la iteracion
            eta[k, i-T_cycle[len(T_cycle)-1]*k-1] = xrx[i-T_cycle[len(T_cycle)-1]*k-1]
    else:
        k += 1
        xrx = np.linspace(r_i[k], r_f[k], T_cycle[len(T_cycle)-1])*ley  # Usamos el dato maximo para la iteracion
        eta[k, i-T_cycle[len(T_cycle)-1]*k-1] = xrx[i-T_cycle[len(T_cycle)-1]*k-1]
        
T_rep = 3  # Dias
C_energy = 0.183  # $/kWh
omega = 10000  # $/tone
C_rep = 80000 + 400000  # Costo de la mano de obra + costo de los mill liners

# Inventados
rho = np.sort(np.random.normal(8.95, 0.2, len(T_cycle)))  # Densidad g/cm³

# Calculados
rho_avg = np.average(rho)  # g/cm³
print('--------------------------------------------------------------------')
print('rho_avg: ', rho_avg)
print('--------------------------------------------------------------------')

N = np.zeros(len(T_cycle))
for i in range(len(N)):
    N[i] = (T_cycle[i] + T_rep)/(T_avg + T_rep)
    i += 1
print('N: ', N)
print('--------------------------------------------------------------------')

C_DT = omega*M_i
print('C_DT: ', C_DT)
print('--------------------------------------------------------------------')

C_ins = 20000/11*np.linspace(0.33, 0.825, 4*30) # Inspeccion mensual
print('C_ins: [', C_ins[0]*30,', ', C_ins[4*30-1]*30,']')
print('--------------------------------------------------------------------')

P_gross = np.zeros(len(T_cycle))

k = 0
while k <= 99:
    p_gross = 0
    for i in range(1, T_cycle[k]-1):
        p_gross = (p_gross + M_i*24*eta[k, i]*omega - E_i[i]*C_energy - C_ins[i])
        i += 1
    P_gross[k] = (p_gross - C_DT - C_rep)*N[k]
    k += 1

print('P_gross :', P_gross)
print('--------------------------------------------------------------------')
'''
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