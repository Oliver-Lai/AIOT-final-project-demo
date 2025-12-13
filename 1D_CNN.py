import tensorflow as tf
from tensorflow.keras import layers, models

def build_1d_cnn_model(input_shape):
    """
    建立 1-D CNN 模型用於水質時序預測 (Slide 24)
    Input: 過去 N 小時的水質數據 (pH, Ammonia, Nitrate)
    Output: 未來的水質預測值
    """
    model = models.Sequential([
        # 第一層卷積：提取時序特徵 (Slide 169 - 擅長捕捉局部特徵)
        layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),
        
        # 第二層卷積
        layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        
        # 展平層
        layers.Flatten(),
        
        # 全連接層
        layers.Dense(50, activation='relu'),
        
        # 輸出層：預測 pH, Ammonia, Nitrate 三個數值
        layers.Dense(3) 
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

if __name__ == "__main__":
    # 測試模型架構
    # 假設輸入是過去 24 小時的數據，有 3 個特徵
    model = build_1d_cnn_model((24, 3))
    model.summary()