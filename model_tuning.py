"""
中文情感分析系統 - 模型調優工具
Created: 2025-05-08 11:47:37 UTC
Author: XinLeiYo
Version: 1.0.0

此模組用於對情感分析模型進行調優，包括：
1. 自定義字典建立
2. 情感閾值優化
3. 模型效能評估
"""

import pandas as pd
import numpy as np
from snownlp import SnowNLP
from collections import defaultdict
import logging
from typing import Dict, List, Any
from datetime import datetime
import json
import os

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=f'model_tuning_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.log'
)
logger = logging.getLogger(__name__)

class ModelTuner:
    def __init__(self):
        """初始化模型調優器"""
        self.custom_dict = defaultdict(float)
        self.thresholds = {
            "positive": 0.7,
            "negative": 0.3
        }
        self.training_stats = []
        self.validation_metrics = {}
        
        # 建立必要的目錄
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
    def load_training_data(self, file_path: str) -> pd.DataFrame:
        """載入訓練數據"""
        try:
            data = pd.read_csv(file_path)
            logger.info(f"成功載入訓練數據: {len(data)} 條記錄")
            return data
        except Exception as e:
            logger.error(f"載入訓練數據失敗: {str(e)}")
            raise

    def build_custom_dict(self, texts: List[str], sentiments: List[float]):
        """建立自定義情感字典"""
        try:
            word_sentiments = defaultdict(list)
            
            # 收集每個詞的所有情感分數
            for text, sentiment in zip(texts, sentiments):
                s = SnowNLP(text)
                words = s.words
                
                for word in words:
                    word_sentiments[word].append(sentiment)
            
            # 計算加權平均情感分數
            for word, scores in word_sentiments.items():
                # 使用指數加權，讓極端情感的權重更大
                weights = np.exp(np.abs(np.array(scores) - 0.5))
                self.custom_dict[word] = np.average(scores, weights=weights)
            
            logger.info(f"自定義字典建立完成，共包含 {len(self.custom_dict)} 個詞")
            
            # 保存字典
            self._save_dict()
            
        except Exception as e:
            logger.error(f"建立自定義字典時發生錯誤: {str(e)}")
            raise

    def optimize_thresholds(self, texts: List[str], true_sentiments: List[float]):
        """優化情感閾值"""
        try:
            scores = []
            for text, true_sentiment in zip(texts, true_sentiments):
                s = SnowNLP(text)
                scores.append((s.sentiments, true_sentiment))
            
            # 使用網格搜索找到最佳閾值
            best_accuracy = 0
            best_thresholds = self.thresholds.copy()
            
            # 定義搜索範圍
            pos_range = np.arange(0.6, 0.9, 0.02)
            neg_range = np.arange(0.2, 0.5, 0.02)
            
            for pos_thresh in pos_range:
                for neg_thresh in neg_range:
                    if pos_thresh <= neg_thresh:
                        continue
                        
                    accuracy = self._calculate_accuracy(scores, pos_thresh, neg_thresh)
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_thresholds = {
                            "positive": pos_thresh,
                            "negative": neg_thresh
                        }
            
            self.thresholds = best_thresholds
            logger.info(f"優化後的閾值: 正面={best_thresholds['positive']:.2f}, "
                        f"負面={best_thresholds['negative']:.2f}, "
                        f"準確率={best_accuracy:.2f}")
            
            # 保存閾值
            self._save_thresholds()
            
        except Exception as e:
            logger.error(f"優化閾值時發生錯誤: {str(e)}")
            raise

    def _calculate_accuracy(self, scores: List[tuple], pos_thresh: float, 
                            neg_thresh: float) -> float:
        """計算準確率"""
        correct = 0
        total = len(scores)
        
        for pred_score, true_sentiment in scores:
            # 預測的情感類別
            pred_sentiment = (
                1.0 if pred_score > pos_thresh else
                0.0 if pred_score < neg_thresh else
                0.5
            )
            
            # 計算預測是否正確（允許一定的誤差範圍）
            if abs(pred_sentiment - true_sentiment) < 0.3:
                correct += 1
        
        return correct / total

    def evaluate(self, texts: List[str], true_sentiments: List[float]) -> Dict[str, float]:
        """評估模型性能"""
        try:
            predictions = []
            confidence_scores = []
            
            for text in texts:
                s = SnowNLP(text)
                score = s.sentiments
                
                # 應用優化後的閾值
                if score > self.thresholds["positive"]:
                    pred = 1.0
                elif score < self.thresholds["negative"]:
                    pred = 0.0
                else:
                    pred = 0.5
                    
                predictions.append(pred)
                
                # 計算信心分數
                confidence = abs(score - 0.5) * 2
                confidence_scores.append(confidence)
            
            # 計算評估指標
            metrics = self._calculate_metrics(predictions, true_sentiments, confidence_scores)
            
            # 記錄評估結果
            self.validation_metrics = metrics
            logger.info(f"評估結果: {metrics}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"評估過程中發生錯誤: {str(e)}")
            raise

    def _calculate_metrics(self, predictions: List[float], true_values: List[float],
                            confidence_scores: List[float]) -> Dict[str, float]:
        """計算詳細的評估指標"""
        total = len(predictions)
        correct = sum(1 for p, t in zip(predictions, true_values) 
                        if abs(p - t) < 0.3)
        
        # 計算各種指標
        metrics = {
            "accuracy": correct / total,
            "average_error": sum(abs(p - t) for p, t in zip(predictions, true_values)) / total,
            "average_confidence": sum(confidence_scores) / total,
            "high_confidence_accuracy": sum(1 for p, t, c in zip(predictions, true_values, confidence_scores)
                                            if c > 0.8 and abs(p - t) < 0.3) /
                                        sum(1 for c in confidence_scores if c > 0.8)
                                            if any(c > 0.8 for c in confidence_scores) else 0
        }   
        
        return metrics

    def _save_dict(self):
        """保存自定義字典"""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f'models/custom_dict_{timestamp}.json'
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(dict(self.custom_dict), f, ensure_ascii=False, indent=2)
                
            # 同時更新當前使用的字典檔案
            with open('custom_dict.json', 'w', encoding='utf-8') as f:
                json.dump(dict(self.custom_dict), f, ensure_ascii=False, indent=2)
                
            logger.info(f"自定義字典已保存: {filename}")
            
        except Exception as e:
            logger.error(f"保存字典時發生錯誤: {str(e)}")
            raise

    def _save_thresholds(self):
        """保存優化後的閾值"""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f'models/thresholds_{timestamp}.json'
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.thresholds, f, indent=2)
                
            # 同時更新當前使用的閾值檔案
            with open('thresholds.json', 'w', encoding='utf-8') as f:
                json.dump(self.thresholds, f, indent=2)
                
            logger.info(f"閾值配置已保存: {filename}")
            
        except Exception as e:
            logger.error(f"保存閾值時發生錯誤: {str(e)}")
            raise

def main():
    """主函數"""
    try:
        # 初始化調優器
        tuner = ModelTuner()
        
        # 載入訓練數據
        data = tuner.load_training_data('training_data.csv')
        
        # 分割訓練集和測試集
        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        # 建立自定義字典
        tuner.build_custom_dict(train_data['text'].tolist(), 
                                train_data['sentiment'].tolist())
            
        # 優化閾值
        tuner.optimize_thresholds(train_data['text'].tolist(),
                                train_data['sentiment'].tolist())
        
        # 評估模型
        metrics = tuner.evaluate(test_data['text'].tolist(),
                                  test_data['sentiment'].tolist())
        
        # 輸出最終結果
        logger.info("模型調優完成！")
        logger.info(f"最終評估指標: {metrics}")
        
    except Exception as e:
        logger.error(f"程序執行過程中發生錯誤: {str(e)}")
        raise

if __name__ == "__main__":
    start_time = datetime.utcnow()
    logger.info(f"=== 開始執行模型調優 - {start_time} ===")
    main()
    end_time = datetime.utcnow()
    logger.info(f"=== 完成模型調優 - {end_time} ===")
    logger.info(f"總耗時: {end_time - start_time}")