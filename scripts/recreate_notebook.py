import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()

nb.cells.append(new_markdown_cell(
"""# Classification des types de globules blancs

Ce notebook propose une chaîne de traitement complète pour identifier automatiquement les leucocytes (lymphocytes, monocytes, neutrophiles, éosinophiles, basophiles, etc.) à partir d'images de microscopie optique.
"""))

nb.cells.append(new_markdown_cell('## 1. Import des librairies et configuration'))

nb.cells.append(new_code_cell(
"""import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.metrics import classification_report, confusion_matrix

plt.style.use('seaborn-v0_8')
sns.set_context('talk')

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# === Chemins robustes (fonctionne depuis `notebooks/` ou depuis la racine) ===
# On cherche un marqueur du projet (README, main.py, requirements.txt) pour
# remonter à la racine même si le notebook est exécuté depuis `notebooks/`.

def find_project_root(start: Path = Path.cwd(), markers=('main.py', 'README.md', 'requirements.txt')):
    p = start
    while True:
        if any((p / m).exists() for m in markers):
            return p
        if p.parent == p:
            return start
        p = p.parent

PROJECT_ROOT = find_project_root()
print(f'Project root: {PROJECT_ROOT}')

print(f'TensorFlow {tf.__version__}')
print('GPU(s) disponible(s) :', tf.config.list_physical_devices('GPU'))
"""))

nb.cells.append(new_markdown_cell('## 2. Chargement des données (train/validation/test)'))

nb.cells.append(new_code_cell(
"""# Dossier de données (doit être présent à la racine du projet)
DATA_ROOT = PROJECT_ROOT / 'data' / 'raw'
TRAIN_DIR = DATA_ROOT / 'train'
VAL_DIR = DATA_ROOT / 'validation'
TEST_DIR = DATA_ROOT / 'test'

if not TRAIN_DIR.exists() or not VAL_DIR.exists() or not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Les dossiers train/validation/test sont introuvables. Vérifiez que '{DATA_ROOT}' existe."
    )

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

base_train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

base_val_ds = keras.utils.image_dataset_from_directory(
    VAL_DIR,
    shuffle=False,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

base_test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    shuffle=False,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = base_train_ds.class_names
num_classes = len(class_names)

print(f'Classes détectées ({num_classes}) : {class_names}')
print(f'Taille des images : {IMG_SIZE} - Batch size : {BATCH_SIZE}')
"""))

nb.cells.append(new_markdown_cell('## 3. Prétraitement des images et augmentation'))

nb.cells.append(new_code_cell(
"""AUTOTUNE = tf.data.AUTOTUNE

# Séquence d'augmentation appliquée à la volée (rotation/zoom/shift/flip/contraste)
data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.12),
    layers.RandomZoom(0.12),
    layers.RandomTranslation(0.08, 0.08),
    layers.RandomContrast(0.15),
], name='data_augmentation')

normalization_layer = layers.Rescaling(1.0 / 255)

def prepare_dataset(dataset, augment=False):
    def _process(images, labels):
        if augment:
            images = data_augmentation(images, training=True)
        images = normalization_layer(images)
        return images, labels

    return dataset.map(_process, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

train_ds = prepare_dataset(base_train_ds, augment=True)
val_ds = prepare_dataset(base_val_ds, augment=False)
test_ds = prepare_dataset(base_test_ds, augment=False)
"""))

nb.cells.append(new_markdown_cell('## 4. Construction du modèle CNN'))

nb.cells.append(new_code_cell(
"""def conv_block(x, filters, dropout_rate=0.2):
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    return x

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = conv_block(inputs, 32, dropout_rate=0.1)
x = conv_block(x, 64, dropout_rate=0.15)
x = conv_block(x, 128, dropout_rate=0.2)
x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.35)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = keras.Model(inputs, outputs, name='wbc_cnn_classifier')
model.summary()
"""))

nb.cells.append(new_markdown_cell('## 5. Compilation et entraînement'))

nb.cells.append(new_code_cell(
"""EPOCHS = 30
LEARNING_RATE = 1e-4

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

checkpoint_dir = PROJECT_ROOT / 'models' / 'checkpoints'
checkpoint_dir.mkdir(parents=True, exist_ok=True)
checkpoint_path = checkpoint_dir / 'wbc_cnn.keras'

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
    ),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
]

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks,
)
"""))

nb.cells.append(new_markdown_cell('## 6. Evaluation sur le jeu de test'))

nb.cells.append(new_code_cell(
"""test_loss, test_acc = model.evaluate(test_ds)
print(f'Loss test : {test_loss:.4f}')
print(f'Accuracy test : {test_acc:.4f}')

# Prédictions

y_true = np.concatenate([labels.numpy() for _, labels in test_ds], axis=0)
y_prob = model.predict(test_ds)
y_pred = np.argmax(y_prob, axis=1)

conf_mat = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Classe réelle')
plt.xlabel('Prédiction')
plt.title('Matrice de confusion')
plt.tight_layout()
plt.show()

report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
print('Classification report:')
print(report)
"""))

nb.cells.append(new_markdown_cell('## 7. Visualisation des prédictions'))

nb.cells.append(new_code_cell(
"""def show_sample_predictions(dataset, n_images=9):
    sample_batch = next(iter(dataset.unbatch().batch(n_images)))
    images, labels = sample_batch
    preds = model.predict(images)
    pred_labels = np.argmax(preds, axis=1)

    images = images.numpy()
    labels = labels.numpy()
    total = min(n_images, images.shape[0])

    cols = int(np.ceil(np.sqrt(total)))
    rows = int(np.ceil(total / cols))
    plt.figure(figsize=(3 * cols, 3 * rows))
    for idx in range(total):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(images[idx])
        plt.title(f'Reel: {class_names[labels[idx]]}\nPred: {class_names[pred_labels[idx]]}')
        plt.axis('off')
    plt.tight_layout()

show_sample_predictions(test_ds, n_images=9)
"""))

nbformat.write(nb, r'c:\Projet_IA\blood-cell-recognition-ml\notebooks\white_blood_cell_classification.ipynb')

print('Notebook réécrit avec des chemins robustes et sans sorties.')
