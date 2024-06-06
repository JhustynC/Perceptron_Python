import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pk

class Perceptron:
    def __init__(self, pesos, sesgo, factor):
        self.pesos = np.array(pesos)
        self.sesgo = sesgo
        self.factor_aprendeizaje = factor
        self.error_total = 0
        self.errores = []
        
        try:
            self.cargar_pesos()
        except Exception as e:
            print(f'\n{e}')
            
    def entrenar(self, X, y, max_iteraciones=10, tolerancia=3):
        num_muestras, num_caracteristicas = X.shape
        self.errores = []
        self.error_total = 0
        for indice_X, muestra in enumerate(X):

            y_d = np.dot(self.pesos, muestra) + self.sesgo

            error =  np.abs(y[indice_X] - y_d)
            self.errores.append(error)
            
            self.pesos = self.pesos + self.factor_aprendeizaje * error * muestra

            self.error_total += error

            print(f'\n---Interacion {indice_X + 1} -> Patron {X[indice_X]}:{y[indice_X]}---')
            print(f'Valor de y: {round(y_d,4)}')
            print(f'Error: {round(error, 4)}')
            print(f'Pesos: {self.pesos}')
            print(f'Error total: {round(self.error_total,4)}')
            
            #! Criterios de parada
            if  indice_X + 1 == num_muestras:
                print('\nFinalizado por cantidad de muestras')
                break

            if len(self.errores) > tolerancia and all(e == self.errores[-1] for e in self.errores[-tolerancia:]):
                print(f"\n!!Deteniendo temprano ya que el error no ha cambiado en las últimas {tolerancia} iteraciones.")
                break

            if indice_X + 1 >= max_iteraciones:
                print('\nMaximo de iteraciones alcanzado')
                break
        
        self.guardar_pesos()
    
    def graficar_errores(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.errores, marker='o', linestyle='--', color='b', label=f"Error total: {round(self.error_total,2)}")
        plt.title('Errores durante el entrenamiento de la red')
        plt.xlabel('Iteración')
        plt.ylabel('Error')

        # Anotaciones para cada punto
        for i, error in enumerate(self.errores):
            plt.annotate(f'{round(error, 2)}', (i, error), textcoords="offset points", xytext=(0, 10), ha='center')

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
        pesos = open('pesos.dat','rb')
        self.pesos = pk.load(pesos)
        pesos.close()
        

def load_dataset(csv_path):
    data = pd.read_csv(csv_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

def main(): 
    
    op = 0
    perceptron = Perceptron([0.840, 0.394, 0.783], 0, 0.3)
    while op!=3:
        print('\n------Adeline------')
        print('1) Entrenar red')
        print('2) Clasificar muestra')
        print('3) Salir ')
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
 
                    perceptron.entrenar(X, y, max_iteraciones=7, tolerancia=3)

                    print("\nEntrenamiento completado.")
                    # print(perceptron.errores)
                    print("\n---Grafica de errores---")
                    perceptron.graficar_errores()
                    
                except Exception as e:
                    print(e)
            case 2:     
                print('\n---Clasificar muestras---')
                y = perceptron.predecir([1,0,1])
                print(f'Prediccion: {y}')

            case 3:
                print("Saliendo...")
                break
            case _:
                print("Opción inválida.")
        

main()
