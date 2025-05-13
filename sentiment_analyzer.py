"""
SentiTune-CN - 情感分析核心
Created: 2025-05-13 17:23:37 UTC
Author: XinLeiYo
Version: 1.1.0

此模組提供中文文本情感分析功能，
包含自定義字典支援和信心分數計算。
"""

from snownlp import SnowNLP
from typing import Dict, Any, List
import logging
import json
from datetime import datetime
import os
from src.agents.rag_agent import RAGAgent

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            os.path.join('logs', f'analyzer_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.log'),
            encoding='utf-8'
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        """初始化情感分析器"""
        # 初始化設定
        self.custom_dict = {}
        self.thresholds = {
            "positive": 0.7,
            "negative": 0.3
        }
        
        # 確保必要的目錄存在
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('src/knowledge_base', exist_ok=True)
        
        # 載入自定義設定
        self._load_custom_settings()
        
        # 初始化 RAG Agent
        self.rag_agent = RAGAgent()
        
        logger.info("情感分析器初始化完成")

    def _load_custom_settings(self):
        """載入自定義設定"""
        try:
            # 載入自定義字典
            if os.path.exists('models/custom_dict.json'):
                with open('models/custom_dict.json', 'r', encoding='utf-8') as f:
                    self.custom_dict = json.load(f)
                logger.info(f"已載入自定義字典，共 {len(self.custom_dict)} 個詞")
            
            # 載入閾值設定
            if os.path.exists('models/thresholds.json'):
                with open('models/thresholds.json', 'r', encoding='utf-8') as f:
                    self.thresholds = json.load(f)
                logger.info(f"已載入閾值設定: {self.thresholds}")
                
        except Exception as e:
            logger.error(f"載入自定義設定時發生錯誤: {str(e)}")

    def analyze(self, text: str) -> Dict[str, Any]:
        """分析文字情感"""
        try:
            # 基礎分析
            s = SnowNLP(text)
            sentiment_score = s.sentiments
            
            # RAG 分析
            rag_result = self.rag_agent.analyze_with_context(text, [])
            
            # 應用自定義字典進行調整
            words = s.words
            custom_scores = []
            word_weights = []
            
            for word in words:
                if word in self.custom_dict:
                    custom_scores.append(self.custom_dict[word])
                    polarity = abs(self.custom_dict[word] - 0.5) * 2
                    word_weights.append(1 + polarity)
                else:
                    word_weights.append(1.0)
            
            # 計算調整後的分數
            if custom_scores:
                weighted_custom_score = sum(score * weight for score, weight 
                                            in zip(custom_scores, word_weights))
                total_weight = sum(word_weights)
                adjusted_score = (sentiment_score + weighted_custom_score / total_weight) / 2
            else:
                adjusted_score = sentiment_score
            
            # 應用 RAG 規則調整
            if rag_result["matched_rules"]:
                for rule in rag_result["matched_rules"]:
                    adjusted_score += rule["adjustment"]["sentiment_score"]
            
            # 確保分數在合理範圍內
            adjusted_score = max(0.0, min(1.0, adjusted_score))
            
            # 改進的信心分數計算
            confidence_factors = [
                # 1. 基於情感極性的信心
                abs(adjusted_score - 0.5) * 2,
                
                # 2. 基於自定義字典匹配度的信心
                len(custom_scores) / len(words) if words else 0,
                
                # 3. 基於文本長度的信心
                min(len(words) / 20, 1.0),
                
                # 4. 基於情感一致性的信心
                1.0 if not custom_scores else 
                1.0 - abs(sentiment_score - sum(custom_scores) / len(custom_scores)) / 2,
                
                # 5. RAG規則匹配的信心
                len(rag_result["matched_rules"]) * 0.1
            ]
            
            # 計算加權平均的信心分數
            weights = [0.3, 0.2, 0.1, 0.2, 0.2]  # 調整後的權重
            confidence_score = sum(factor * weight 
                                for factor, weight in zip(confidence_factors, weights))
            
            # 確保信心分數在合理範圍內
            confidence_score = max(0.3, min(confidence_score, 1.0))
            
            # 判斷情感類別
            if adjusted_score > self.thresholds["positive"]:
                sentiment = "正面"
            elif adjusted_score < self.thresholds["negative"]:
                sentiment = "負面"
            else:
                sentiment = "中性"
                confidence_score *= 0.9
            
            # 提取關鍵詞
            keywords = s.keywords(3)
            
            return {
                "狀態": "成功",
                "情感分數": adjusted_score,
                "情感類別": sentiment,
                "信心分數": confidence_score,
                "信心因素": {
                    "情感極性": confidence_factors[0],
                    "字典匹配": confidence_factors[1],
                    "文本長度": confidence_factors[2],
                    "情感一致性": confidence_factors[3],
                    "規則匹配": confidence_factors[4]
                },
                "關鍵詞": keywords,
                "分析時間": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "rag_analysis": {
                    "matched_rules": [rule["name"] for rule in rag_result["matched_rules"]],
                    "context_patterns": rag_result["context_patterns"]
                }
            }
            
        except Exception as e:
            logger.error(f"分析過程發生錯誤: {str(e)}")
            return {
                "狀態": "失敗",
                "錯誤訊息": str(e)
            }

    def update_custom_dict(self, new_dict: Dict[str, float]) -> bool:
        """更新自定義字典"""
        try:
            # 驗證新字典
            for word, score in new_dict.items():
                if not isinstance(score, (int, float)) or not 0 <= score <= 1:
                    raise ValueError(f"詞語 '{word}' 的分數無效: {score}")
            
            # 更新字典
            self.custom_dict.update(new_dict)
            
            # 保存到文件
            with open('models/custom_dict.json', 'w', encoding='utf-8') as f:
                json.dump(self.custom_dict, f, ensure_ascii=False, indent=4)
            
            logger.info(f"自定義字典更新成功，當前共有 {len(self.custom_dict)} 個詞")
            return True
            
        except Exception as e:
            logger.error(f"更新自定義字典時發生錯誤: {str(e)}")
            return False

    def update_thresholds(self, new_thresholds: Dict[str, float]) -> bool:
        """更新閾值設定"""
        try:
            # 驗證新閾值
            if not all(0 <= new_thresholds.get(k, v) <= 1 
                        for k, v in self.thresholds.items()):
                raise ValueError("閾值必須在 0 到 1 之間")
            
            # 更新閾值
            self.thresholds.update(new_thresholds)
            
            # 保存到文件
            with open('models/thresholds.json', 'w', encoding='utf-8') as f:
                json.dump(self.thresholds, f, indent=4)
            
            logger.info(f"閾值更新成功: {self.thresholds}")
            return True
            
        except Exception as e:
            logger.error(f"更新閾值時發生錯誤: {str(e)}")
            return False

def main():
    """用於測試的主函數"""
    analyzer = SentimentAnalyzer()
    
    test_texts = [
        "這個產品真的很讚！用了就愛上了。",
        "服務品質差，態度很不好。",
        "還可以啦，但有改進空間。",
        "哇！真是太厲害了呢！（反諷）"
    ]
    
    for text in test_texts:
        result = analyzer.analyze(text)
        print(f"\n測試文本: {text}")
        print(f"分析結果: {json.dumps(result, ensure_ascii=False, indent=2)}")

if __name__ == "__main__":
    main()