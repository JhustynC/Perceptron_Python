import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

def load_dataset(csv_path):
    try:
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Error: El archivo no se encontró.")
        return None, None
    except pd.errors.EmptyDataError:
        print("Error: El archivo está vacío.")
        return None, None
    except pd.errors.ParserError:
        print("Error: Error al analizar el archivo.")
        return None, None
    
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

csv_path = input("Ingrese el path del dataset CSV: ") 
csv_path = f'CSV/{csv_path}'
X, y = load_dataset(csv_path)

if X is None or y is None:
    print("No se pudo cargar el dataset. Saliendo...")
    exit()

_, num_caracteristicas = X.shape

print("Número de características:", num_caracteristicas)

# Definición del modelo con 5 salidas
model = Sequential([
    Dense(units=25, activation='relu', input_shape=[num_caracteristicas]),
    Dense(units=25, activation='relu'),
    # Cambia el número de unidades a 5 para la salida
    Dense(units=5, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.01),
    # Si es clasificación multiclase, usa 'categorical_crossentropy'
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Suponiendo que 'y' contiene enteros de clases desde 0 hasta 4
y_one_hot = to_categorical(y, num_classes=5)

# Ahora puedes entrenar el modelo con las etiquetas en formato one-hot
print('Empieza el entrenamiento...')
historial = model.fit(X, y_one_hot, epochs=1000, verbose=False)
# historial = model.fit(X, y, epochs=1000, verbose=False)
print('Modelo entrenado')

plt.xlabel('# Época')
plt.ylabel('Magnitud de pérdida')
plt.plot(historial.history['loss'])
plt.show()

# Hacer predicciones
print('Prediccion')
csv_path = input("Ingrese el path del dataset CSV: ") 
csv_path = f'CSV/{csv_path}'
X_test, y_test = load_dataset(csv_path)
predicciones = model.predict(X_test)

# Imprimir las predicciones
for i, prediccion in enumerate(predicciones):
    print(f"Entrada: {X_test[i]}, Predicción: {prediccion}, Valor real: {y_test[i]} ")
