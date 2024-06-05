import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, pesos, sesgo, factor):
        self.pesos = np.array(pesos)
        self.sesgo = sesgo
        self.factor_aprendeizaje = factor

    def entrenar(self, X, y, max_iteraciones=10, tolerancia=3):
        num_muestras, num_caracteristicas = X.shape
        errores = []

        error_total = 0
        for indice_XY, muestra in enumerate(X):

            y_d = np.dot(self.pesos, muestra) + self.sesgo

            error =  np.abs(y[indice_XY] - y_d)
            errores.append(error)

            valores = zip(muestra, self.pesos)
            for indice, (muestra, peso) in enumerate(valores):
                self.pesos[indice] = peso + self.factor_aprendeizaje*(error)*muestra

            error_total += error

            print(f'\n---Interacion {indice_XY + 1} -> Patron {X[indice_XY]}:{y[indice_XY]}---')
            print(f'Valor de y: {round(y_d,4)}')
            print(f'Error: {round(error, 4)}')
            print(f'Pesos: {self.pesos}')
            print(f'Erro total: {round(error_total,4)}')

            errores.append(error_total)

            if (indice_XY+1 == num_muestras):
                print('\nFinalizado por cantidad de muestras')

            if len(errores) > tolerancia and all(e == errores[-1] for e in errores[-tolerancia:]):
                print(f"\n!!Deteniendo temprano ya que el error no ha cambiado en las últimas {tolerancia} iteraciones.")
                break

            if indice + 1 >= max_iteraciones:
                print('\nMaximo de iteraciones alcanzado')
                break

    def predecir(self, X):
        d_x = np.dot(X, self.pesos) + self.sesgo
        return self.funcion_activacion(d_x)

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

                    perceptron = Perceptron([0.840, 0.394, 0.783], 0, 0.3)
                    perceptron.entrenar(X, y, max_iteraciones=1000, tolerancia=3)

                    print("\nEntrenamiento completado.")

                except Exception as e:
                    print(e)
          
            case "n":
                print("Saliendo...")
                break
            case _:
                print("Opción inválida.")
        

main()
