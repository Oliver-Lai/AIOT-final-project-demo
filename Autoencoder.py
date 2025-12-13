import tensorflow as tf
from tensorflow.keras import layers, models, Input

def build_autoencoder(input_dim=51):
    """
    建立 AutoEncoder 用於 FET 電訊號降維 (Slide 23)
    Input: 51維的電壓-電流數據
    Latent: 2維 (用於視覺化)
    """
    # Encoder (編碼器)
    input_img = Input(shape=(input_dim,))
    encoded = layers.Dense(32, activation='relu')(input_img)
    encoded = layers.Dense(16, activation='relu')(encoded)
    
    # Latent Space (潛在空間 - Slide 23 圖表)
    latent_vector = layers.Dense(2, activation='linear', name='latent_space')(encoded)
    
    # Decoder (解碼器)
    decoded = layers.Dense(16, activation='relu')(latent_vector)
    decoded = layers.Dense(32, activation='relu')(decoded)
    decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)
    
    # 完整的 AutoEncoder
    autoencoder = models.Model(input_img, decoded)
    
    # 獨立的 Encoder 模型 (用於提取特徵做視覺化)
    encoder = models.Model(input_img, latent_vector)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder