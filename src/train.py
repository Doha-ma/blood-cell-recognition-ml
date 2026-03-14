import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

# charger le modèle
model = load_model("models/blood_cell_classifier.keras")

# générateur de test
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    "data/raw/test",
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# évaluation
loss, accuracy = model.evaluate(test_generator)

print("Loss :", loss)
print("Accuracy :", accuracy)

# prédictions
predictions = model.predict(test_generator)

y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# matrice de confusion
cm = confusion_matrix(y_true, y_pred)

print(cm)

sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()