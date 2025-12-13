import openai  

def generate_water_quality_report(current_data, prediction_data):
    """
    生成水質分析報告
    """
    
    prompt = f"""
    角色與目標: 你是一位專業的 AI 水質分析專家。
    
    一、背景資訊:
    監測場域: 高密度龍膽石斑魚養殖魚塭。
    核心指標: pH (穩定性), 氨 (毒性), 硝酸鹽 (藻類風險)。
    
    二、輸入數據:
    [即時量測數據]
    - pH: {current_data['ph']}
    - 氨 (Ammonia): {current_data['ammonia']} mg/L
    - 硝酸鹽 (Nitrates): {current_data['nitrate']} mg/L
    
    [1-D CNN 模型預測結果 (未來 24 小時)]
    - pH 趨勢: {prediction_data['ph_trend']}
    - 氨趨勢: {prediction_data['ammonia_trend']}
    - 硝酸鹽趨勢: {prediction_data['nitrate_trend']}
    
    三、任務:
    請生成一份「智慧魚塭水質分析與預警報告」，包含：
    1. 數據摘要
    2. 綜合分析 (指標關聯性)
    3. 趨勢預測與風險評估 (指出最大風險)
    4. 潛在原因推斷 (例如：投餌過量、硝化系統效率不足)
    5. 行動建議 (按優先順序：緊急、短期、觀察)
    
    請使用 Markdown 格式輸出。
    """
    

    response = openai.ChatCompletion.create(...)
    return response.choices[0].message.content
    
    print("--- 傳送給 LLM 的 Prompt ---")
    print(prompt)
    print("----------------------------")
    
    return " 這是由 LLM 生成的專業水質報告..."