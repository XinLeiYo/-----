"""
中文情感分析系統 - 訓練數據生成器
Created: 2025-05-08 12:36:13 UTC
Author: XinLeiYo
Version: 2.0.0

此模組用於生成中文情感分析的訓練數據，
包含正面、負面、中性和反諷評論。
"""

import pandas as pd
import random
from datetime import datetime
import logging
import os
from typing import List, Dict, Optional

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingDataGenerator:
    def __init__(self):
        """初始化訓練數據生成器"""
        # 正面評論模板
        self.positive_templates = [
            "真的超讚der",
            "整個無敵神",
            "太強了吧",
            "這個真心推薦",
            "簡直是神器",
            "太香了",
            "真的很可以",
            "根本就是救星",
            "完全看不出有什麼缺點",
            "我直接吹爆",
            "超級無敵讚",
            "根本離不開了",
            "用了就愛上",
            "推爆這個",
            "品質穩穩的",
            "完全沒話說",
            "超級棒的啦",
            "好用到哭",
            "讚到爆炸",
            "超乎期待"
        ]

        # 負面評論模板
        self.negative_templates = [
            "這什麼爛東西",
            "根本是智障設計",
            "完全不推好吧",
            "這也太扯了吧",
            "爛透了啦",
            "退錢啦",
            "真的是浪費時間",
            "超大雷",
            "這不行啊",
            "根本是詐騙",
            "難用死了",
            "超級雷",
            "踩雷了啦",
            "爛透了",
            "氣死人了",
            "根本垃圾",
            "不推不推",
            "超級雷包",
            "有夠糟糕",
            "太誇張了吧"
        ]

        # 反諷評論模板
        self.sarcastic_templates = [
            "那你很厲害哦",
            "好啦隨便你",
            "反正你最厲害",
            "是是是你說得對",
            "哇塞厲害了",
            "真是太棒了呢",
            "好棒棒哦",
            "你最棒了啦",
            "真是太有水準了",
            "沒錯就是這樣呢",
            "誰理你啊",
            "隨便啦",
            "我也是這樣覺得呢",
            "真是特別呢",
            "好厲害喔",
            "你最厲害了",
            "我覺得很ok呢",
            "真是傑作啊",
            "高手高手",
            "太專業了吧",
            "厲害了我的哥",
            "就醬子吧",
            "可不就是這樣",
            "whatever啦",
            "嗯哼"
        ]

        # 中性評論模板
        self.neutral_templates = [
            "還行吧",
            "普普通通",
            "一般般",
            "也就這樣",
            "不算差但也不算好",
            "看情況吧",
            "可以考慮",
            "有優點也有缺點",
            "見仁見智吧",
            "待觀察",
            "還可以啦",
            "馬馬虎虎",
            "湊合著用",
            "將就了",
            "還算ok",
            "還行啦",
            "凱子啦",
            "還好啦",
            "普通啦",
            "看你要怎麼想"
        ]

        # 產品/服務關鍵詞
        self.subjects = [
            "這款APP", "這個遊戲", "這個直播", "這支影片",
            "這個商品", "這家店", "這個外送", "這個服務",
            "這個課程", "這個平台", "這個功能", "這個設計",
            "這個更新", "這個介面", "這款手機", "這個系統",
            "這個老師", "這個客服", "這個主播", "這個博主",
            "這個配送", "這家餐廳", "這個社群", "這個版本"
        ]

    def generate_text(self, template: str, subject: str = "", add_subject: bool = True) -> str:
        """生成完整的評論文本"""
        if not add_subject or not subject:
            return template
        
        template_first = random.random() > 0.5
        return f"{template}！{subject}" if template_first else f"{subject}{template}"

    def generate_data(self, count_per_type: int = 100) -> pd.DataFrame:
        """生成訓練數據"""
        data = []
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        try:
            # 生成正面評論
            for _ in range(count_per_type):
                template = random.choice(self.positive_templates)
                subject = random.choice(self.subjects)
                text = self.generate_text(template, subject)
                data.append({
                    "text": text,
                    "sentiment": round(random.uniform(0.7, 1.0), 2),
                    "type": "positive"
                })

            # 生成負面評論
            for _ in range(count_per_type):
                template = random.choice(self.negative_templates)
                subject = random.choice(self.subjects)
                text = self.generate_text(template, subject)
                data.append({
                    "text": text,
                    "sentiment": round(random.uniform(0.0, 0.3), 2),
                    "type": "negative"
                })

            # 生成反諷評論
            for _ in range(count_per_type):
                template = random.choice(self.sarcastic_templates)
                subject = random.choice(self.subjects) if random.random() > 0.7 else ""
                text = self.generate_text(template, subject, add_subject=bool(subject))
                data.append({
                    "text": text,
                    "sentiment": round(random.uniform(0.1, 0.4), 2),
                    "type": "sarcastic"
                })

            # 生成中性評論
            for _ in range(count_per_type):
                template = random.choice(self.neutral_templates)
                subject = random.choice(self.subjects)
                text = self.generate_text(template, subject)
                data.append({
                    "text": text,
                    "sentiment": round(random.uniform(0.3, 0.7), 2),
                    "type": "neutral"
                })

            # 轉換為DataFrame並隨機打亂順序
            df = pd.DataFrame(data)
            df = df.sample(frac=1).reset_index(drop=True)
            
            # 確保資料目錄存在
            os.makedirs('data', exist_ok=True)
            
            # 保存數據
            filename = f'data/training_data_{timestamp}.csv'
            df.to_csv(filename, index=False, encoding='utf-8')
            
            # 同時保存一個當前版本
            df.to_csv('data/current_training_data.csv', index=False, encoding='utf-8')
            
            logger.info(f"成功生成訓練數據，共 {len(df)} 條記錄")
            logger.info(f"數據已保存至: {filename}")
            
            return df
            
        except Exception as e:
            logger.error(f"生成訓練數據時發生錯誤: {str(e)}")
            raise

    def get_stats(self, df: pd.DataFrame) -> Dict[str, int]:
        """獲取數據統計信息"""
        return {
            "總數量": len(df),
            "正面評論": len(df[df['type'] == 'positive']),
            "負面評論": len(df[df['type'] == 'negative']),
            "反諷評論": len(df[df['type'] == 'sarcastic']),
            "中性評論": len(df[df['type'] == 'neutral'])
        }

def main():
    """主函數"""
    try:
        logger.info("開始生成訓練數據...")
        
        generator = TrainingDataGenerator()
        df = generator.generate_data()
        
        # 顯示統計信息
        stats = generator.get_stats(df)
        logger.info("數據統計:")
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
        
        # 顯示樣本
        logger.info("\n數據樣本:")
        for type_ in ['positive', 'negative', 'sarcastic', 'neutral']:
            sample = df[df['type'] == type_].sample(n=1).iloc[0]
            logger.info(f"{type_} 樣本: {sample['text']} (情感分數: {sample['sentiment']})")
        
        return df
        
    except Exception as e:
        logger.error(f"程序執行失敗: {str(e)}")
        raise

if __name__ == "__main__":
    start_time = datetime.utcnow()
    logger.info(f"開始執行 - {start_time}")
    
    try:
        df = main()
        logger.info("訓練數據生成完成！")
    except Exception as e:
        logger.error(f"程序執行失敗: {str(e)}")
        raise
    finally:
        end_time = datetime.utcnow()
        logger.info(f"結束執行 - {end_time}")
        logger.info(f"總耗時: {end_time - start_time}")