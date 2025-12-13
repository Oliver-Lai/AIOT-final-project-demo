import time
import serial  
import numpy as np
import tensorflow as tf
import openai
from collections import deque
from datetime import datetime


SERIAL_PORT = 'COM3'  # Windows: COMx, Mac/Linux: /dev/ttyUSB0
BAUD_RATE = 115200
FET_MODEL_PATH = 'models/fet_converter.h5'
CNN_MODEL_PATH = 'models/cnn_predictor.h5'
OPENAI_API_KEY = "your-api-key-here"
openai.api_key = OPENAI_API_KEY
HISTORY_WINDOW_SIZE = 24  
SAMPLING_INTERVAL = 3600  
# ==========================================================

class AquacultureSystem:
    def __init__(self):
        print("[System] 初始化系統中...")
        
        # 1. 初始化 Serial 連線 (連接 Arduino/ESP32)
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            print(f"[Hardware] 成功連接至 {SERIAL_PORT}")
        except serial.SerialException as e:
            print(f"[Error] 無法連接感測器硬體: {e}")
            raise

        # 2. 載入 AI 模型
        print("[AI] 正在載入模型...")
        try:
            # Slide 18: 定量分析模型 (電訊號 -> 化學濃度)
            self.fet_model = tf.keras.models.load_model(FET_MODEL_PATH)
            
            # Slide 24: 1-D CNN 時序預測模型
            self.cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
            print("[AI] 模型載入完成")
        except IOError:
            print("[Error] 找不到模型檔案，請確認 models/ 資料夾")
            raise

        # 3. 初始化資料緩衝區 (用於 CNN 時序輸入)
        # 使用 deque 建立固定長度的佇列，當新數據進來時，舊數據會自動移除
        self.history_buffer = deque(maxlen=HISTORY_WINDOW_SIZE)

    def read_sensor_raw_signal(self):
        """
        從 FET 感測器讀取原始電訊號 (Raw I-V Data)
        """
        if self.ser.in_waiting > 0:
            line = self.ser.readline().decode('utf-8').strip()
            if line:
                # 解析 CSV 格式的數據
                raw_data = [float(x) for x in line.split(',')]
                return np.array(raw_data).reshape(1, -1) # Reshape for Model
        return None

    def convert_signal_to_concentration(self, raw_signal):
        concentrations = self.fet_model.predict(raw_signal, verbose=0)
        return concentrations[0] # 回傳一維陣列

    def predict_future_trends(self):
        # 將緩衝區轉為 numpy array (Shape: 1 x 24 x 3)
        input_seq = np.array(self.history_buffer).reshape(1, HISTORY_WINDOW_SIZE, 3)
        
        # CNN 推論
        prediction = self.cnn_model.predict(input_seq, verbose=0)
        return prediction[0] # 回傳預測的 [Next_pH, Next_Ammonia, Next_Nitrate]

    def generate_llm_report(self, current, predicted):
        """
        Slide 25: 系統整合 — 將 CNN 預測結果輸入 LLM，生成最終分析報告
        """
        
        prompt = f"""
        角色: 專業水產養殖水質分析師
        
        [即時監測數據]
        - pH: {current[0]:.2f}
        - 氨氮 (Ammonia): {current[1]:.2f} mg/L
        - 硝酸鹽 (Nitrate): {current[2]:.2f} ppm
        
        [未來 24H 趨勢預測 (1-D CNN)]
        - pH 預測值: {predicted[0]:.2f}
        - 氨氮預測值: {predicted[1]:.2f}
        - 硝酸鹽預測值: {predicted[2]:.2f}
        
        任務:
        1. 分析當前水質風險 (氨中毒、藻類增生)。
        2. 根據預測趨勢提供具體行動建議 (如: 開啟增氧機、換水、停止投餌)。
        3. 請用簡潔的條列式回答。
        """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", # 或 gpt-4
            messages=[
                {"role": "system", "content": "你是一個幫助漁民管理魚塭的AI助手。"},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    def run(self):
        print(f"[System] 系統啟動，開始監測 (取樣間隔: {SAMPLING_INTERVAL}秒)...")
        
        while True:
            try:
                # 1. 讀取硬體訊號
                # 注意: 實際運作時，這行會等待 Arduino 傳送資料
                raw_signal = self.read_sensor_raw_signal()
                
                if raw_signal is not None:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # 2. 轉換為濃度 (Offline Process Step 1 的應用)
                    current_vals = self.convert_signal_to_concentration(raw_signal)
                    ph, nh3, no3 = current_vals
                    
                    print(f"[{timestamp}] 實測: pH={ph:.2f}, NH3={nh3:.2f}, NO3={no3:.2f}")
                    
                    # 3. 更新時序緩衝區
                    self.history_buffer.append(current_vals)
                    
                    # 4. 當收集滿 24 筆資料後，開始進行預測與報告 (Offline Process Step 2 的應用)
                    if len(self.history_buffer) == HISTORY_WINDOW_SIZE:
                        print("   > 資料充足，進行趨勢預測...")
                        future_vals = self.predict_future_trends()
                        
                        # 5. 呼叫 LLM (Online Process 最終輸出)
                        print("   > 生成 LLM 決策報告...")
                        report = self.generate_llm_report(current_vals, future_vals)
                        
                        print("\n" + "="*30)
                        print("★ AI 水產守護者分析報告 ★")
                        print("="*30)
                        print(report)
                        print("="*30 + "\n")
                    
                    # 等待下一次取樣
                    time.sleep(SAMPLING_INTERVAL)
                    
            except KeyboardInterrupt:
                print("\n[System] 停止監測")
                self.ser.close()
                break
            except Exception as e:
                print(f"[Error] 發生錯誤: {e}")
                time.sleep(5) # 錯誤後稍作等待重試

if __name__ == "__main__":
    system = AquacultureSystem()
    system.run()