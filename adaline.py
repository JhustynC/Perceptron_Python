import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pk

class Adaline:
    def __init__(self, sesgo, factor):
        self.pesos = np.array
        self.sesgo = sesgo
        self.factor_aprendizaje = factor
        self.error_total_modelo = 0
        self.errores = []
    
    #? Se pude agregar tolerancia_error=1e-5
    def entrenar(self, X, y, max_iteraciones, estabilizacion_error, tolerancia_error=1e-50):
        _, num_caracteristicas = X.shape
        self.pesos = np.random.rand(num_caracteristicas)
        # self.pesos = np.array([0.5, 0.84, 0.394, 0.783])
        self.errores = []
        
        for iteracion in range(max_iteraciones):
            error_total_iteracion = 0
            print(f'\n===== Iteración {iteracion + 1} =====')
            
            for indice_X, muestra in enumerate(X):
                y_d = np.dot(self.pesos, muestra) + self.sesgo
                error = y[indice_X] - y_d
                self.pesos += self.factor_aprendizaje * error * muestra
                error_total_iteracion += error**2

                print(f'\n--- Iteración {iteracion + 1} -> Patrón {muestra}: {y[indice_X]} ---')
                print(f'Valor de y: {round(y_d, 4)}')
                print(f'Error: {round(error, 4)}')
                print(f'Pesos: {self.pesos}')
                print(f'Error total: {round(error_total_iteracion, 4)}')

            # self.errores.append(np.abs(error_total_iteracion))
            self.errores.append(error_total_iteracion)

            if (len(self.errores) > estabilizacion_error and 
                all(e == self.errores[-1] for e in self.errores[-estabilizacion_error:])):
                print(f"\n!! Deteniendo temprano ya que el error no ha cambiado en las últimas {estabilizacion_error} iteraciones.")
                break
            
            if abs(error_total_iteracion) < tolerancia_error:
                print(f"\n!! Deteniendo temprano ya que el error total ({round(error_total_iteracion, 4)}) es menor que la tolerancia de error ({tolerancia_error}).")
                break

            if iteracion + 1 >= max_iteraciones:
                print('\nMáximo de iteraciones alcanzado')
                break
        
        # self.errores =  np.square(self.errores)
        self.error_total_modelo = sum(self.errores)
        self.guardar_pesos()
    
    def graficar_errores(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.errores, marker='o', linestyle='solid', color='b', label=f"Error total: {round(self.error_total_modelo,2)} \nPesos: {self.pesos}")
        plt.title('Errores durante el entrenamiento de la red')
        plt.xlabel('Iteración')
        plt.ylabel('Error')

        # Anotaciones para cada punto
        plt.annotate(f'{round(self.errores[0], 2)}', (0, self.errores[0]), textcoords="offset points", xytext=(0, 10), ha='center')
        plt.annotate(f'{round(self.errores[-1], 2)}', (len(self.errores) - 1, self.errores[-1]), textcoords="offset points", xytext=(0, 10), ha='center')

        plt.legend()
        plt.grid(True)
        plt.show()

    def predecir(self, muestra):
        y_p = np.dot(self.pesos, np.array(muestra)) + self.sesgo
        return y_p
    
    def guardar_pesos(self):
        if self.errores == []:
            print('No hay datos para guardar')
            return

        pesos = open('pesos.dat','wb')
        pk.dump(self.pesos, pesos, protocol=pk.HIGHEST_PROTOCOL)
        pesos.close()

    def cargar_pesos(self):
        try:
            pesos = open('pesos.dat','rb')
            self.pesos = pk.load(pesos)
            pesos.close()
            print('\n---Carga completa---')
        except Exception as e:
            print(e)

def vector_a_matriz(vector):
    return [vector[i:i+5] for i in range(0, len(vector), 5)]

# Función para imprimir una matriz 5x5 en la consola
def imprimir_matriz(matriz):
    print('\n')
    for fila in matriz:
        print(" ".join(str(celda) for celda in fila))


def load_dataset(csv_path):
    data = pd.read_csv(csv_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

def main(): 
    
    op = 0
    adaline = Adaline(0, 0.1)
    while op!=4:
        print('\n------Adeline------')
        print('1) Entrenar red')
        print('2) Clasificar muestra')
        print('3) Cargar pesos')
        print('4) Salir ')
        try: 
            op = int(input("opcion: "))
        except Exception as e:
            print(e)
        
        match op:
            case 1:
                try:
                    csv_path = input("Ingrese el path del dataset CSV: ") 
                    csv_path = f'CSV/{csv_path}'
                    X, y = load_dataset(csv_path)

                    adaline.entrenar(X, y, max_iteraciones=100, estabilizacion_error=5)

                    print("\nEntrenamiento completado.")
                    
                    print('-----IMPRIMIR LA ENTRADA-----')
                    n,m = X.shape
                    [imprimir_matriz(vector_a_matriz(X[n])) for n  in range(n)]
                    
                    print("\n---Grafica de errores---")
                    adaline.graficar_errores()
                    
                except Exception as e:
                    print(e)
            case 2:   
                try:  
                    print('\n---Clasificar muestras---')
                    
                    csv_path = input("Ingrese el path del dataset CSV: ") 
                    csv_path = f'CSV/{csv_path}'
                    X_test, y_test = load_dataset(csv_path)
                    
                    predictions = [adaline.predecir(muestra) for muestra in X_test]

                    # Evaluate the predictions
                    print("Predicciones vs Valores Reales:")
                    for i, (pred, real) in enumerate(zip(predictions, y_test)):
                        print(f"Muestra {i+1}: Predicción = {round(pred, 2)}, Valor Real = {real}")

                    print('-----IMPRIMIR LA ENTRADA-----')
                    n,m = X_test.shape
                    [imprimir_matriz(vector_a_matriz(X_test[n])) for n  in range(n)]
                
                except Exception as e:
                    print(e)
            case 3:
                adaline.cargar_pesos()
            case 4:
                print("Saliendo...")
                break
            case _:
                print("Opción inválida.")
        
main()
