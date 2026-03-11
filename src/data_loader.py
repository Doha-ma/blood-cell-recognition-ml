import os
from kaggle.api.kaggle_api_extended import KaggleApi
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Téléchargement Kaggle ---
def download_blood_dataset():
    api = KaggleApi()
    api.authenticate()
    dataset_path = 'paultimothymooney/blood-cell-images'
    dest_path = 'data/raw/'
    os.makedirs(dest_path, exist_ok=True)
    api.dataset_download_files(dataset_path, path=dest_path, unzip=True)
    print("Dataset téléchargé et dézippé dans", dest_path)

# --- Générateurs de données corrigés ---
def get_data_generators(train_dir='data/raw/train', test_dir='data/raw/test', img_size=(64,64), batch_size=32):
    # Prétraitement et augmentation pour l'entraînement
    train_datagen = ImageDataGenerator(
        rescale=1./255,          # normalisation
        rotation_range=20,       # rotation aléatoire
        width_shift_range=0.1,   # translation horizontale
        height_shift_range=0.1,  # translation verticale
        zoom_range=0.2,          # zoom aléatoire
        horizontal_flip=True,    # flip horizontal
        fill_mode='nearest'      # pour les pixels manquants après transformation
    )

    # Prétraitement pour validation/test
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Générateur pour l'entraînement
    train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    # Générateur pour la validation/test
    test_generator = test_datagen.flow_from_directory(
        directory=test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, test_generator