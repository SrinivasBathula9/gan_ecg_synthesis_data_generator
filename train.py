import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input, Conv1D, Flatten, Reshape, Conv1DTranspose
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
import os

# Suppress AVX2 warning
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Parameters
signal_length = 187  # Adjusted to match real signal length
latent_dim = 32
learning_rate = 0.0002
beta_1 = 0.5
batch_size = 32
epochs = 50

# Load and preprocess MIT-BIH PTBDB normal/abnormal data
def load_and_preprocess_data():
    df_normal = pd.read_csv("ptbdb_normal.csv", header=None)
    df_abnormal = pd.read_csv("ptbdb_abnormal.csv", header=None)
    
    # Concatenate and shuffle
    data = pd.concat([df_normal, df_abnormal])
    data = data.sample(frac=1).reset_index(drop=True)
    
    # Extract signals
    signals = data.iloc[:, :-1].values  # Last column is the label
    
    # Trim signals to match required length
    signals = signals[:, :signal_length]
    
    # Normalize data
    scaler = StandardScaler()
    signals = scaler.fit_transform(signals)
    
    return signals

train_data = load_and_preprocess_data()
print("Shape of training data:", train_data.shape)

# Define the Generator
def build_generator():
    model = Sequential()
    model.add(Dense(128, input_dim=latent_dim))
    model.add(Dense(signal_length, activation='tanh'))  # Ensure correct output size
    model.add(Reshape((signal_length,)))  # Ensure reshaped size matches signal_length
    return model

# Define the Discriminator
def build_discriminator():
    model = Sequential()
    model.add(Reshape((signal_length, 1), input_shape=(signal_length,)))
    model.add(Conv1D(32, 5, strides=2, padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(Conv1D(64, 5, strides=2, padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

# Build and Compile the GAN
generator = build_generator()
discriminator = build_discriminator()
optimizer_g = Adam(learning_rate, beta_1=beta_1)
optimizer_d = Adam(learning_rate, beta_1=beta_1)

discriminator.compile(loss='binary_crossentropy', optimizer=optimizer_d, metrics=['accuracy'])
discriminator.trainable = False
gan_input = Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=optimizer_g)

# Training Loop
def train_gan(generator, discriminator, gan, train_data, latent_dim, epochs, batch_size):
    batch_count = train_data.shape[0] // batch_size
    history = {'d_loss': [], 'g_loss': []}
    
    for epoch in range(epochs):
        for _ in range(batch_count):
            real_signals = train_data[np.random.randint(0, train_data.shape[0], batch_size)]
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            generated_signals = generator.predict(noise)
            
            # Ensure generated signals have the correct shape
            generated_signals = generated_signals[:, :signal_length]  # Trim if necessary
            real_signals = real_signals[:, :signal_length]  # Ensure consistency
            
            X = np.concatenate([real_signals, generated_signals])
            y = np.array([1] * batch_size + [0] * batch_size)
            
            discriminator.trainable = True
            d_loss = discriminator.train_on_batch(X, y)
            
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            y_gan = np.ones((batch_size, 1))
            discriminator.trainable = False
            g_loss = gan.train_on_batch(noise, y_gan)
            
            history['d_loss'].append(d_loss[0])
            history['g_loss'].append(g_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, d_loss={d_loss[0]:.4f}, g_loss={g_loss:.4f}")
    
    return history

history = train_gan(generator, discriminator, gan, train_data, latent_dim, epochs, batch_size)

# Generate and Save Synthetic ECG Signals
def generate_and_save_ecg_signals(generator, latent_dim, num_signals=100, filename="synthetic_ecg_signals.csv"):
    noise = np.random.normal(0, 1, (num_signals, latent_dim))
    generated_signals = generator.predict(noise)
    
    # Ensure generated signals have the correct shape
    generated_signals = generated_signals[:, :signal_length]  # Trim if necessary
    
    df_generated = pd.DataFrame(generated_signals)
    df_generated.to_csv(filename, index=False)
    print(f"Generated ECG signals saved to {filename}")
    
    return generated_signals

generated_ecg_signals = generate_and_save_ecg_signals(generator, latent_dim)

plt.figure(figsize=(12, 6))
for i in range(5):
    plt.subplot(5, 1, i+1)
    plt.plot(generated_ecg_signals[i])
    plt.title(f"Generated ECG Signal {i+1}")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history['d_loss'], label='Discriminator Loss')
plt.plot(history['g_loss'], label='Generator Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('GAN Training Loss')
plt.legend()
plt.show()
