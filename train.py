import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# Model definition
model = Sequential([
    ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    Flatten(),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Model compiled successfully")

# Dummy training example (replace with real data)
# model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Save model
model.save("models/ild_model.h5")
print("Model saved successfully")
