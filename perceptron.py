import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class Perceptron:
    def __init__(self, log=False):
        self.pesos = None
        self.sesgo = None
        self.log = log

    def entrenar(self, X, y, max_iteraciones=1000, tolerancia=3):
        num_muestras, num_caracteristicas = X.shape
        self.pesos = np.full(num_caracteristicas,0.5)
        self.sesgo = 0.5
        errores = []

        for iteracion in range(max_iteraciones):
            error_total = 0

            for indice, muestra in enumerate(X):
                
                # print(f"Indice: {indice}, Muestra: {muestra}")
                
                d_x = np.dot(muestra, self.pesos) + self.sesgo
                y_predicho = self.funcion_activacion(d_x)
                valor_real = y[indice] 
                
                if self.log:
                    print(f"\n---Iteración {iteracion+1}, Muestra {indice+1}---")
                    print(f"Pesos: {self.pesos}")
                    print(f"Muestra: {muestra}")
                    print(f"Sesgo (Theta): {self.sesgo}")
                    print(f"d(x) (Salida Lineal): {d_x}")
                    print(f"Predicho: {y_predicho}, Actual: {y[indice]}")
                    print(f"Ajuste (Valor real): {valor_real}")
                
                if y_predicho == y[indice]: continue
                
                #AJUSTES PESOS y SESGO (THETA)
                self.pesos += valor_real * muestra
                self.sesgo += valor_real
                
                if self.log:
                    print(f"Pesos Actualizados: {self.pesos}")
                    print(f"Sesgo Actualizado: {self.sesgo}\n")
                    
                error_total += 1

            errores.append(error_total)
            if self.log:
                print(f"\nFin de la Iteración {iteracion+1}, Errores Totales: {error_total}\n")

            if len(errores) > tolerancia and all(e == errores[-1] for e in errores[-tolerancia:]):
                if self.log:
                    print(f"!!Deteniendo temprano ya que el error no ha cambiado en las últimas {tolerancia} iteraciones.")
                break

            if iteracion + 1 >= max_iteraciones:
                break

    def predecir(self, X):
        d_x = np.dot(X, self.pesos) + self.sesgo
        return self.funcion_activacion(d_x)

    def funcion_activacion(self, x):
        return np.where(x <= 0, -1, 1)
    
def graficar(X, y, perceptron):
    # Reducir las dimensiones a 2D usando PCA
    pca = PCA(n_components=2)
    X_reducido = pca.fit_transform(X)

    # Graficar puntos
    for clase in np.unique(y):
        plt.scatter(X_reducido[y == clase][:, 0], X_reducido[y == clase][:, 1], label=f"Clase {clase}")

    # Calcular los límites de la gráfica
    x_min, x_max = X_reducido[:, 0].min() - 1, X_reducido[:, 0].max() + 1
    y_min, y_max = X_reducido[:, 1].min() - 1, X_reducido[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = perceptron.predecir(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    # Graficar la línea de decisión
    plt.contourf(xx, yy, Z, alpha=0.1)
    plt.plot([], [], ' ', label=f"Pesos: {perceptron.pesos}, Sesgo: {perceptron.sesgo}")
    plt.title("Perceptrón - Muestras y Función de Activación (PCA Reducido)")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.legend()
    plt.show()

def load_dataset(csv_path):
    data = pd.read_csv(csv_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

def main(): 
    
    while op := input("¿Desea cargar un dataset CSV? (s/n): ").lower():
        match op:
            case "s":
                try:
                    csv_path = input("Ingrese el path del dataset CSV: ") 
                    csv_path = f'CSV/{csv_path}'
                    X, y = load_dataset(csv_path)

                    perceptron = Perceptron(log=True)
                    perceptron.entrenar(X, y, max_iteraciones=100, tolerancia=3)

                    print("Entrenamiento completado.")
                    
                    print(perceptron.pesos)

                    # Probar predicciones
                    for muestra in X:
                        prediccion = perceptron.predecir(muestra)
                        print(f"Entrada: {muestra}, Predicción: {prediccion}")

                    # Graficar los resultados
                    graficar(X, y, perceptron)
                    
                except Exception as e:
                    print(e)
          
            case "n":
                print("Saliendo...")
                return
            case _:
                print("Opción inválida.")
        
# if __name__ == '__main__':
main()
