from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle

# Paths
data_dir = 'dataset'  # change to your actual folder location

# Image settings
img_height, img_width = 64, 64
batch_size = 32

# Data generators with train/validation split
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # 20% validation
)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    subset='training'
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    subset='validation'
)

# Get class names for label mapping and saving
class_names = list(train_gen.class_indices.keys())  # should be ['cats_set', 'dogs_set']
num_classes = len(class_names)
print("Class names:", class_names)

# CNN Model architecture
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
epochs = 5
model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen
)

# Save model and class names
model.save("cnn_model.h5")
with open("cnn_classes.pkl", "wb") as f:
    pickle.dump(class_names, f)

print("Training complete! Model and classes saved.")
