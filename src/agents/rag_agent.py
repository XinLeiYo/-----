"""
RAG Agent for SentiTune-CN
Created: 2025-05-13
Author: XinLeiYo
"""

from typing import Dict, List, Any
import json
from datetime import datetime
import logging
from pathlib import Path
import numpy as np
import jieba
import jieba.posseg as pseg

test_cases = [
    # 案例1：明顯的情感波動
    """
    這家餐廳真是太棒了！
    食物美味得讓人感動。
    但服務員的態度真是糟糕透頂。
    簡直是最差勁的用餐體驗。
    """,
    
    # 案例2：漸進式情感變化
    """
    一開始感覺還不錯。
    後來開始有點不太滿意。
    再後來越來越差。
    最後簡直氣壞了！
    """,
    
    # 案例3：強烈的情感對比
    """
    我超愛這款手機的外觀和效能！
    但是電池續航卻爛到不行。
    相機功能又特別優秀。
    可是價格貴得嚇死人。
    """
]

class RAGAgent:
    def __init__(self):
        """初始化 RAG Agent"""
        self.knowledge_base = self._load_knowledge_base()
        self.context_window = []
        self.max_context_length = 5
        
        # 初始化 jieba
        jieba.initialize()
        # 載入自定義詞典（如果有的話）
        custom_dict_path = Path("models/custom_dict.txt")
        if custom_dict_path.exists():
            jieba.load_userdict(str(custom_dict_path))
        
        logging.info("RAG Agent 初始化完成")
        
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """載入知識庫"""
        try:
            kb_path = Path("src/knowledge_base/sentiment_rules.json")
            if kb_path.exists():
                with open(kb_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logging.error(f"載入知識庫時發生錯誤: {str(e)}")
            return {}
            
    def analyze_with_context(self, text: str, history: List[Dict]) -> Dict[str, Any]:
        """使用上下文進行增強分析"""
        # 分割輸入文本為句子，並過濾空白
        sentences = [s.strip() for s in text.replace('。', '。\n').split('\n') if s.strip()]
        
        # 更新上下文窗口
        if sentences:
            # 如果有歷史記錄，先添加歷史
            if history:
                for prev_text in history[-self.max_context_length:]:
                    self.context_window.append(prev_text.get("text", ""))
            
            # 添加新句子
            self.context_window.extend(sentences)
            
            # 保持最大上下文長度
            self.context_window = self.context_window[-self.max_context_length:]
            
            logging.debug(f"當前上下文窗口: {self.context_window}")
        
        # 分析結果
        context_patterns = self._analyze_context_patterns()
        matched_rules = self._match_rules(text)
        
        result = {
            "matched_rules": matched_rules,
            "context_patterns": context_patterns,
            "analysis_time": datetime.utcnow().isoformat(),
            "metrics": {
                "context_length": len(self.context_window),
                "rule_matches": len(matched_rules),
                "confidence_score": self._calculate_confidence_score(context_patterns, text)
            },
            "text_info": {
                "original_text": text,
                "sentences": sentences,
                "context_window": self.context_window.copy()  # 返回副本以避免外部修改
            }
        }
        
        logging.debug(f"RAG分析完成: {json.dumps(result, ensure_ascii=False)}")
        return result
        
    def _match_rules(self, text: str) -> List[Dict[str, Any]]:
        """匹配知識庫規則"""
        matched_rules = []
        for rule in self.knowledge_base.get("rules", []):
            if any(pattern in text for pattern in rule.get("patterns", [])):
                matched_rules.append(rule)
        return matched_rules
        
    def _analyze_context_patterns(self) -> Dict[str, Any]:
        """分析上下文模式"""
        # 檢查上下文窗口是否為空
        if not self.context_window:
            return {
                "context_length": 0,
                "sentiment_flow": {
                    "trend": 0.0,
                    "volatility": 0.0
                }
            }
        
        # 分析每個句子
        sentences = []
        for text in self.context_window:
            # 使用標點符號分割句子
            text_sentences = [s.strip() for s in text.split('。') if s.strip()]
            sentences.extend(text_sentences)
        
        return {
            "context_length": len(sentences),
            "sentiment_flow": self._calculate_sentiment_flow(),
            "sentences": sentences  # 保存分析的句子列表
        }
        
    def _calculate_sentiment_flow(self) -> Dict[str, float]:
        """計算情感流動趨勢"""
        # 確保有足夠的上下文
        if not self.context_window:
            return {
                "trend": 0.0,
                "volatility": 0.0
            }
        
        # 初始化情感分數列表
        sentiment_scores = []
        
        # 計算每個句子的情感分數
        for text in self.context_window:
            # 處理每個句子
            sentences = [s.strip() for s in text.split('。') if s.strip()]
            for sentence in sentences:
                score = self._calculate_text_sentiment(sentence)
                # 將分數標準化到 [-1, 1] 範圍
                score = max(min(score, 1.0), -1.0)
                sentiment_scores.append(score)
                logging.debug(f"句子: {sentence}, 情感分數: {score}")
        
        # 如果只有一個分數，無法計算趨勢和波動
        if len(sentiment_scores) < 2:
            return {
                "trend": sentiment_scores[0] if sentiment_scores else 0.0,
                "volatility": 0.0
            }
        
        # 計算趨勢（使用線性回歸斜率）
        x = np.array(range(len(sentiment_scores)))
        y = np.array(sentiment_scores)
        trend = np.polyfit(x, y, 1)[0]
        
        # 計算波動性（使用最大最小值差異）
        volatility = max(sentiment_scores) - min(sentiment_scores)
        
        # 添加額外的波動指標
        result = {
            "trend": float(trend),
            "volatility": float(volatility),
            "max_score": float(max(sentiment_scores)),
            "min_score": float(min(sentiment_scores)),
            "score_range": float(volatility),
            "scores": sentiment_scores
        }
        
        logging.debug(f"情感流動分析: {result}")
        return result
    
    def _calculate_text_sentiment(self, text: str) -> float:
        """計算單個文本的情感分數"""
        words = pseg.cut(text)
        sentiment_score = 0.0
        word_count = 0
        
        # 從知識庫獲取情感詞典和強度詞
        sentiment_dict = self.knowledge_base.get("sentiment_dict", {})
        positive_dict = sentiment_dict.get("positive", {})
        negative_dict = sentiment_dict.get("negative", {})
        intensifiers = self.knowledge_base.get("intensifiers", {})
        
        # 先找出所有強度詞
        intensity_multiplier = 1.0
        for intensifier in intensifiers:
            if intensifier in text:
                intensity_multiplier *= intensifiers[intensifier]
        
        # 計算情感分數
        for word, flag in words:
            # 檢查正面詞典
            if word in positive_dict:
                score = positive_dict[word]
                sentiment_score += score
                word_count += 1
            # 檢查負面詞典
            elif word in negative_dict:
                score = negative_dict[word]
                sentiment_score += score
                word_count += 1
        
        # 應用強度詞的影響
        if word_count > 0:
            sentiment_score = (sentiment_score * intensity_multiplier) / word_count
        
        # 確保分數在 [-1, 1] 範圍內
        return max(min(sentiment_score, 1.0), -1.0)

    def _calculate_confidence_score(self, context_patterns: Dict, text: str) -> float:
        """計算分析結果的置信度分數
        
        Args:
            context_patterns (Dict): 上下文模式分析結果
            text (str): 當前分析的文本
            
        Returns:
            float: 置信度分數 (0-1)
        """
        confidence = 0.0
        
        # 根據上下文長度調整
        context_length = context_patterns.get("context_length", 0)
        if context_length >= 3:
            confidence += 0.3
        elif context_length >= 1:
            confidence += 0.2
        
        # 根據情感波動調整
        flow = context_patterns.get("sentiment_flow", {})
        volatility = flow.get("volatility", 0.0)
        if volatility < 0.3:  # 情感穩定
            confidence += 0.2
        
        # 根據規則匹配數量調整
        matched_rules = self._match_rules(text)  # 使用傳入的文本
        rules_count = len(matched_rules)
        
        if rules_count >= 2:
            confidence += 0.3
        elif rules_count >= 1:
            confidence += 0.2
        
        return min(confidence + 0.2, 1.0)  # 基礎置信度 0.2

    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """獲取知識庫統計信息"""
        return {
            "total_rules": len(self.knowledge_base.get("rules", [])),
            "version": self.knowledge_base.get("version", "unknown"),
            "last_updated": self.knowledge_base.get("last_updated", "unknown")
        }

    def update_knowledge_base(self, new_rules: List[Dict[str, Any]]) -> bool:
        """更新知識庫規則"""
        try:
            # 驗證新規則格式
            for rule in new_rules:
                if not all(key in rule for key in ["id", "name", "patterns", "conditions", "adjustment"]):
                    raise ValueError(f"規則格式錯誤: {rule}")
            
            # 更新現有規則
            self.knowledge_base["rules"].extend(new_rules)
            self.knowledge_base["last_updated"] = datetime.utcnow().isoformat()
            
            # 保存到文件
            kb_path = Path("src/knowledge_base/sentiment_rules.json")
            with open(kb_path, "w", encoding="utf-8") as f:
                json.dump(self.knowledge_base, f, ensure_ascii=False, indent=4)
            
            logging.info(f"成功更新知識庫，新增 {len(new_rules)} 條規則")
            return True
            
        except Exception as e:
            logging.error(f"更新知識庫時發生錯誤: {str(e)}")
            return False
    
    def run_sentiment_test(self) -> None:
        """運行情感分析測試"""
        print("\n=== 開始情感分析測試 ===\n")
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n▶ 測試案例 {i}")
            print("輸入文本：")
            print(case)
            print("\n分析結果：")
            
            # 執行分析
            result = self.analyze_with_context(case, [])
            flow = result["context_patterns"]["sentiment_flow"]
            
            # 顯示詳細結果
            print("\n1. 情感流動：")
            print(f"  - 趨勢: {flow['trend']:.3f}")
            print(f"  - 波動: {flow['volatility']:.3f}")
            print(f"  - 最高分: {flow.get('max_score', 0):.3f}")
            print(f"  - 最低分: {flow.get('min_score', 0):.3f}")
            
            print("\n2. 句子分析：")
            for j, (sentence, score) in enumerate(zip(
                result["text_info"]["sentences"], 
                flow.get("scores", [])), 1):
                print(f"  [{j}] {sentence}")
                print(f"      得分: {score:.3f}")
            
            print("\n3. 分析指標：")
            metrics = result["metrics"]
            print(f"  - 上下文長度: {metrics['context_length']}")
            print(f"  - 規則匹配數: {metrics['rule_matches']}")
            print(f"  - 置信度: {metrics['confidence_score']:.3f}")
            
            print("\n" + "="*50)

if __name__ == "__main__":
        # 設置日誌級別
        logging.basicConfig(level=logging.INFO)
        
        # 創建 RAG Agent 實例
        agent = RAGAgent()
        
        # 運行測試
        agent.run_sentiment_test()