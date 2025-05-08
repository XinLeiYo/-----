"""
中文情感分析系統 - 調優與更新整合腳本
Created: 2025-05-08 11:52:30 UTC
Author: XinLeiYo
Version: 1.0.0

此模組整合了訓練數據生成和模型調優功能，
並確保更新後的模型參數能夠順利整合到主系統中。
"""

import logging
from datetime import datetime
import os
from generate_training_data_v2 import TrainingDataGenerator
from model_tuning import ModelTuner
import pandas as pd

# 設定日誌
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            os.path.join(log_dir, f'tuning_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.log'),
            encoding='utf-8'
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TuningManager:
    def __init__(self):
        """初始化調優管理器"""
        self.timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.data_generator = TrainingDataGenerator()
        self.model_tuner = ModelTuner()
        self.training_data = None
        
    def generate_training_data(self, count_per_type: int = 100):
        """生成訓練數據"""
        try:
            logger.info("開始生成訓練數據...")
            
            # 生成數據
            self.training_data = self.data_generator.generate_data(count_per_type)
            
            # 保存數據
            training_data_path = f'data/training_data_{self.timestamp}.csv'
            os.makedirs('data', exist_ok=True)
            self.training_data.to_csv(training_data_path, index=False, encoding='utf-8')
            
            # 同時保存一個當前版本
            self.training_data.to_csv('data/current_training_data.csv', 
                                    index=False, encoding='utf-8')
            
            logger.info(f"訓練數據已保存: {training_data_path}")
            
            # 顯示數據分布
            self._show_data_distribution()
            
        except Exception as e:
            logger.error(f"生成訓練數據時發生錯誤: {str(e)}")
            raise
            
    def _show_data_distribution(self):
        """顯示數據分布情況"""
        if self.training_data is not None:
            distribution = {
                "正面評論(>0.7)": len(self.training_data[self.training_data['sentiment'] > 0.7]),
                "負面評論(<0.3)": len(self.training_data[self.training_data['sentiment'] < 0.3]),
                "中性評論(0.3-0.7)": len(self.training_data[
                    (self.training_data['sentiment'] >= 0.3) & 
                    (self.training_data['sentiment'] <= 0.7)
                ])
            }
            logger.info("數據分布情況：")
            for category, count in distribution.items():
                logger.info(f"{category}: {count}")
                
    def tune_model(self):
        """執行模型調優"""
        try:
            if self.training_data is None:
                raise ValueError("未找到訓練數據，請先生成訓練數據")
                
            logger.info("開始模型調優...")
            
            # 分割訓練集和測試集
            train_size = int(len(self.training_data) * 0.8)
            train_data = self.training_data[:train_size]
            test_data = self.training_data[train_size:]
            
            # 建立自定義字典
            logger.info("建立自定義字典...")
            self.model_tuner.build_custom_dict(
                train_data['text'].tolist(),
                train_data['sentiment'].tolist()
            )
            
            # 優化閾值
            logger.info("優化閾值...")
            self.model_tuner.optimize_thresholds(
                train_data['text'].tolist(),
                train_data['sentiment'].tolist()
            )
            
            # 評估模型
            logger.info("評估模型性能...")
            metrics = self.model_tuner.evaluate(
                test_data['text'].tolist(),
                test_data['sentiment'].tolist()
            )
            
            logger.info("模型調優完成！")
            logger.info(f"評估指標: {metrics}")
            
            # 保存評估結果
            self._save_evaluation_results(metrics)
            
        except Exception as e:
            logger.error(f"模型調優過程中發生錯誤: {str(e)}")
            raise
            
    def _save_evaluation_results(self, metrics: dict):
        """保存評估結果"""
        try:
            eval_dir = 'evaluation'
            os.makedirs(eval_dir, exist_ok=True)
            
            eval_file = os.path.join(eval_dir, f'evaluation_{self.timestamp}.txt')
            
            with open(eval_file, 'w', encoding='utf-8') as f:
                f.write(f"評估時間: {datetime.utcnow()}\n")
                f.write("評估指標:\n")
                for metric, value in metrics.items():
                    f.write(f"{metric}: {value}\n")
                    
            logger.info(f"評估結果已保存: {eval_file}")
            
        except Exception as e:
            logger.error(f"保存評估結果時發生錯誤: {str(e)}")
            raise

def main():
    """主函數"""
    try:
        logger.info("=== 開始執行調優流程 ===")
        
        # 初始化調優管理器
        manager = TuningManager()
        
        # 生成訓練數據
        manager.generate_training_data(count_per_type=100)
        
        # 執行模型調優
        manager.tune_model()
        
        logger.info("=== 調優流程完成 ===")
        
    except Exception as e:
        logger.error(f"調優流程執行失敗: {str(e)}")
        raise

if __name__ == "__main__":
    start_time = datetime.utcnow()
    logger.info(f"開始執行 - {start_time}")
    
    try:
        main()
    except Exception as e:
        logger.error(f"程序執行失敗: {str(e)}")
        raise
    finally:
        end_time = datetime.utcnow()
        logger.info(f"結束執行 - {end_time}")
        logger.info(f"總耗時: {end_time - start_time}")