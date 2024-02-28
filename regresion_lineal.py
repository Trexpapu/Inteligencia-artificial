import matplotlib.pyplot as plt
import numpy as np

def Auxiliar(w, b):
    yhat = Prediccion(x, w, b)
    error = MSE(y, yhat)
    return error 

def Prediccion(datos, w, b):
    yhat = [w*x + b for x in datos]
    return yhat

def MSE(reales, predichos):
    lista = [(r-p) ** 2 for r, p in zip(reales, predichos)]
    promedio = sum(lista) / len(lista)
    return promedio

def Actualizar(X, y, W, b, alpha):
    dL_dw, dL_db = Gradientes(X, y, W, b)
    W = W - alpha * dL_dw
    b = b - alpha * dL_db
    return W, b

def Gradientes(X, y, W, b):
    yhat = Prediccion(X, W, b)
    dL_dw = np.dot(X, (yhat - y)) / len(X)
    dL_db = np.mean(yhat - y)
    return dL_dw, dL_db

def Entrenamiento(DatosEntrada, DatosSalida, w0, b0, NumEpocas, alpha):
    errores = []
    pesos = []
    sesgos = []
    for epoca in range(NumEpocas):
        ## Predicción
        yhat = Prediccion(DatosEntrada, w0, b0)

        ## Cálculo del error
        error = MSE(DatosSalida, yhat)
        errores.append(error)

        ## Actualizar
        w0, b0 = Actualizar(DatosEntrada, DatosSalida, w0, b0, alpha)
        pesos.append(w0)
        sesgos.append(b0)
        print(f'Época {epoca+1}: Error = {error}, w = {w0}, b = {b0}')

    return w0, b0, errores, pesos, sesgos

x = np.array([-0.2, -2.9, 2.2, -0.5, -1, -2, 1.8, -2.2]) # Entrada
y = np.array([-4.55, -11.85, 2.69, -6.05, -6.22, -8.18, 2.47, -8.98]) # Reales
w = -1.3
b = 3.2
muchosPesos = np.linspace(-5,5,10) # lo pone en notacion matricial
muchos_sesgos = np.linspace(-35, 35, 10)

w_final, b_final, errores, pesos, sesgos = Entrenamiento(x, y, w, b, 20000, alpha=0.001)


#antes del cambio
mapa_errores=[[Auxiliar(w,b) for w in muchosPesos] for b in muchos_sesgos]
plt.contourf(muchosPesos, muchos_sesgos, mapa_errores, cmap = 'coolwarm') # frios son bajos y calientes son altos
plt.scatter(w, b, color="red", marker='X')
plt.scatter(2.80, -3.40, color="purple", marker='o') # minimo porque son los valores reales de C a F que es 1.8, 32
plt.show()


w = w_final
b = b_final

#despues del cambio
mapa_errores=[[Auxiliar(w,b) for w in muchosPesos] for b in muchos_sesgos]
plt.contourf(muchosPesos, muchos_sesgos, mapa_errores, cmap = 'coolwarm') # frios son bajos y calientes son altos
plt.scatter(w, b, color="red", marker='X')
plt.scatter(2.80, -3.40, color="purple", marker='o') # minimo porque son los valores reales de C a F que es 1.8, 32
plt.show()


# Gráfico de error durante el entrenamiento
plt.plot(range(1, len(errores)+1), errores, marker='o', linestyle='-', color="red")
plt.title('Error durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Error')
plt.show()

# Gráfico de cómo cambian los pesos y el sesgo durante el entrenamiento
plt.plot(range(1, len(pesos)+1), pesos, label='Peso (w)')
plt.plot(range(1, len(sesgos)+1), sesgos, label='Sesgo (b)')
plt.title('Cambios en los pesos y el sesgo durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Valor')
plt.legend()
plt.show()

# Gráfico de la regresión lineal final
plt.scatter(x, y, color='blue', label='Datos reales')
plt.plot(x, Prediccion(x, w_final, b_final), color='red', label='Regresión lineal')
plt.title('Regresión lineal final')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

print(f'w final: {w_final}, b final: {b_final}')
