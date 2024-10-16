import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

# Load your dataset here, assuming it's in a CSV format
# Replace 'your_dataset.csv' with the actual dataset path
df = pd.read_csv('your_dataset.csv')

# Separate features and target
X = df.drop('compatibility', axis=1)  # Drop the target column
y = df['compatibility']  # Target column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler to use it for transforming new data
scaler_filename = "scaler.pkl"
with open(scaler_filename, 'wb') as f:
    pickle.dump(scaler, f)

# Create a deep learning model using TensorFlow/Keras
def create_model(input_dim):
    model = Sequential()
    
    # Input layer
    model.add(Dense(256, input_dim=input_dim, activation='relu'))
    
    # Hidden layers with Dropout to avoid overfitting
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))  # Dropout helps to prevent overfitting
    model.add(Dense(64, activation='relu'))
    
    # Output layer (assuming binary classification, adjust if multi-class)
    model.add(Dense(1, activation='sigmoid'))  # Use 'softmax' for multi-class classification
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Get the input dimension from the training set
input_dim = X_train.shape[1]

# Create the model
model = create_model(input_dim)

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, 
                    validation_data=(X_test, y_test),
                    epochs=100, 
                    batch_size=32, 
                    callbacks=[early_stopping])

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'Test Accuracy: {test_acc:.4f}')

# Save the model locally
model_filename = 'color_compatibility_model.h5'
model.save(model_filename)
print(f"Model saved as {model_filename}")

# Save the scaler object for later use
scaler_filename = 'color_scaler.pkl'
with open(scaler_filename, 'wb') as file:
    pickle.dump(scaler, file)

# To load the model in future for predictions:
# loaded_model = tf.keras.models.load_model(model_filename)

# To load the scaler:
# with open(scaler_filename, 'rb') as file:
#     loaded_scaler = pickle.load(file)

# Make predictions on new data
predictions = model.predict(X_test)

# Convert probabilities to class labels (binary classification)
predictions = (predictions > 0.5).astype(int)

# Show sample predictions
print(f"Sample predictions: {predictions[:5].flatten()}")
