# 🌊 智慧池塘水質監測系統 — AIOT Final Project

<img width="895" height="358" alt="系統架構圖" src="https://github.com/user-attachments/assets/8369d7f7-35c1-4801-9241-e7e3c28d4a4a" />

## 📋 專案概述

這是一個基於 **AIoT (人工智慧物聯網)** 的智慧水產養殖監測系統，整合了 FET 感測器硬體、深度學習模型、以及 LLM 智能分析，實現從原始電訊號到專業水質報告的端到端解決方案。

### 🎯 核心能力

| 功能模組 | 技術實現 | 說明 |
|---------|---------|------|
| **訊號轉換** | FET 感測器 + DNN | 將原始 I-V 電訊號轉換為化學濃度 |
| **時序預測** | 1D-CNN | 基於過去 24 小時數據預測未來趨勢 |
| **特徵降維** | AutoEncoder | 51 維電訊號壓縮至 2 維視覺化 |
| **模型優化** | Post-Training Quantization | INT8 量化部署至邊緣裝置 |
| **智能報告** | OpenAI GPT | 結合預測數據生成專業分析報告 |

---

## 🏗️ 專案架構

```
AIOT-final-project-demo/
├── 📄 index.html              # 前端監測儀表板 (可直接開啟)
├── 🐍 main.py                 # 主程式 — 完整監測系統
├── 🐍 report_gen.py           # LLM 報告生成模組
├── 🐍 1D_CNN.py               # 1D-CNN 時序預測模型
├── 🐍 Autoencoder.py          # AutoEncoder 降維模型
├── 🐍 Post-Training Quantization.PY  # ONNX INT8 量化
├── 📄 requirements.txt        # Python 依賴套件
└── 📖 README.md               # 專案說明文件
```

---

## 🧠 AI 模型詳解

### 1. 1D-CNN 時序預測模型 (`1D_CNN.py`)

用於預測未來 24 小時的水質趨勢，輸入過去 24 小時的監測數據。

```python
# 模型架構
Conv1D(64, kernel=3) → MaxPool → Conv1D(32, kernel=3) → MaxPool → Dense(50) → Dense(3)

# 輸入/輸出
Input:  (24, 3)  # 24 個時間點 × 3 個指標 (pH, NH3, NO3)
Output: (3,)     # 預測下一時刻的 3 個指標
```

### 2. AutoEncoder 降維模型 (`Autoencoder.py`)

將 51 維的 FET 電壓-電流數據壓縮至 2 維，用於視覺化和異常檢測。

```python
# 架構
Encoder: 51D → 32D → 16D → 2D (Latent Space)
Decoder: 2D → 16D → 32D → 51D

# 應用場景
- 資料視覺化 (2D 散點圖)
- 異常樣本檢測
- 特徵提取
```

### 3. 訓練後量化 (`Post-Training Quantization.PY`)

將 TensorFlow 模型轉換為 ONNX 格式並進行 INT8 靜態量化，適合部署於邊緣裝置。

```python
# 轉換流程
TensorFlow Model → ONNX (Float32) → ONNX (INT8)

# 量化設定
- 格式: QDQ (Quantize-Dequantize)
- 權重: INT8
- 激活值: INT8
```

---

## ⚙️ 主程式運作流程 (`main.py`)

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  FET 感測器     │ ──▶ │  訊號轉換模型   │ ──▶ │  時序緩衝區     │
│  (Serial Port)  │     │  (DNN)          │     │  (24 筆資料)    │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  LLM 分析報告   │ ◀── │  1D-CNN 預測    │ ◀── │  資料滿 24 筆   │
│  (OpenAI GPT)   │     │  (未來趨勢)     │     │  觸發預測       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### 系統參數配置

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `SERIAL_PORT` | COM3 | 感測器連接埠 |
| `BAUD_RATE` | 115200 | 串列通訊速率 |
| `HISTORY_WINDOW_SIZE` | 24 | CNN 輸入序列長度 |
| `SAMPLING_INTERVAL` | 3600 秒 | 取樣間隔 (1 小時) |

---

## 🖥️ 前端展示介面

### 功能特色

- **即時數據監測**：每 2 秒自動更新一筆數據
- **三維度指標**：硝酸鹽 (NITRATE)、酸鹼值 (PH)、氨氮 (AMMONIA)
- **動態圖表**：使用 Chart.js 實現平滑曲線動畫
- **AI 分析報告**：一鍵生成完整水質評估

### 使用方法

#### 方法一：直接開啟
```bash
# 用瀏覽器開啟 index.html
start index.html  # Windows
open index.html   # macOS
```

#### 方法二：VS Code Live Server
1. 安裝 Live Server 擴充套件
2. 右鍵 `index.html` → Open with Live Server

### 介面區塊

| 區塊 | 功能 |
|------|------|
| 頂部標題區 | 系統名稱與副標題 |
| 狀態監控條 | 6 個關鍵指標即時顯示 |
| 左側圖表區 | 動態三線折線圖 |
| 右側分析區 | AI 報告生成面板 |

---

## 🚀 快速開始

### 環境需求

- Python 3.8+
- TensorFlow 2.10+
- Node.js (選用，前端開發)

### 安裝步驟

```bash
# 1. 複製專案
git clone https://github.com/your-repo/AIOT-final-project-demo.git
cd AIOT-final-project-demo

# 2. 安裝 Python 依賴
pip install -r requirements.txt

# 3. 設定 OpenAI API Key
# 編輯 main.py 中的 OPENAI_API_KEY
```

### 執行後端系統

```bash
# 確保感測器已連接
python main.py
```

### 執行前端 Demo

```bash
# 直接用瀏覽器開啟
start index.html
```

---

## 📦 依賴套件

```txt
pyserial>=3.5        # 串列通訊
tensorflow>=2.10.0   # 深度學習框架
numpy>=1.21.0        # 數值計算
openai>=0.27.0       # LLM API
tf2onnx              # TensorFlow 轉 ONNX (量化用)
onnxruntime          # ONNX 推論引擎
```

---

## 🔬 技術細節

### 水質判斷邏輯

| 指標 | 警戒值 | 建議行動 |
|------|--------|----------|
| pH < 5.3 | ⚠️ 過酸 | 調整水質、添加石灰 |
| 硝酸鹽 > 25 ppm | ⚠️ 過高 | 增加換水頻率 |
| 氨氮 > 0.05 mg/L | ⚠️ 毒性風險 | 使用益生菌、減少投餌 |

### 圖表配置

- 三個獨立 Y 軸對應三種指標
- 最多顯示 15 個時間點
- 平滑曲線 (tension: 0.4)
- 互動式 Tooltip

### 響應式設計

| 裝置 | 佈局 |
|------|------|
| 桌面 (≥1024px) | 雙欄並排 |
| 平板 (<1024px) | 單欄堆疊 |
| 手機 (<768px) | 垂直排列 |

---

## 💡 未來擴展

- [ ] MQTT/WebSocket 即時串流
- [ ] 多站點切換與比較
- [ ] 歷史數據查詢與匯出
- [ ] LINE/Telegram 警報通知
- [ ] 邊緣設備 (Raspberry Pi) 部署
- [ ] 後台管理系統

---

## 🛡️ 注意事項

1. **網路需求**：前端需載入 Chart.js CDN
2. **瀏覽器相容**：建議使用 Chrome/Edge/Firefox 最新版
3. **API 費用**：OpenAI API 需自行申請並注意用量
4. **硬體連接**：執行 `main.py` 前需確認感測器已連接

---

## 👥 團隊成員

| 角色 | 負責項目 |
|------|----------|
| 系統架構 | 整體設計與整合 |
| 前端開發 | 監測儀表板 UI |
| 後端開發 | 感測器通訊與資料處理 |
| AI 模型 | CNN/AutoEncoder 訓練 |

---

## 📧 聯絡方式

如有問題或建議，歡迎透過 GitHub Issues 聯繫！

---

<p align="center">
  <b>版本</b>：v1.0.0 | <b>最後更新</b>：2025-12-13
</p>
