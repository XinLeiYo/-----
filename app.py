"""
SentiTune-CN - 主應用程式
Created: 2025-05-08 13:39:05 UTC
Author: XinLeiYo
Version: 1.1.0

此應用程式提供中文情感分析的網頁介面，
整合了情感分析、模型調優和數據視覺化功能。
"""

# 基礎套件
import os
import json
from datetime import datetime

# 第三方套件
import streamlit as st
import pandas as pd
import altair as alt
import logging

# 自定義模組
from sentiment_analyzer import SentimentAnalyzer
from src.agents.rag_agent import RAGAgent

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            os.path.join('logs', f'app_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.log'),
            encoding='utf-8'
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SentimentApp:
    def __init__(self):
        """初始化應用程式"""
        self.analyzer = SentimentAnalyzer()
        self.load_model_info()
        self.history = []
        
    def load_model_info(self):
        """載入模型資訊"""
        self.model_info = {
            "last_updated": "未知",
            "dictionary_size": 0,
            "threshold_info": {},
            "evaluation_metrics": {},
            "rag_info": {}
        }
        
        try:
            # 檢查自定義字典
            if os.path.exists('models/custom_dict.json'):
                with open('models/custom_dict.json', 'r', encoding='utf-8') as f:
                    custom_dict = json.load(f)
                    self.model_info["dictionary_size"] = len(custom_dict)
                    self.model_info["last_updated"] = datetime.fromtimestamp(
                        os.path.getmtime('models/custom_dict.json')
                    ).strftime("%Y-%m-%d %H:%M:%S")
            
            # 檢查閾值設定
            if os.path.exists('models/thresholds.json'):
                with open('models/thresholds.json', 'r', encoding='utf-8') as f:
                    self.model_info["threshold_info"] = json.load(f)
                    
            # 檢查評估指標
            if os.path.exists('evaluation/latest_metrics.json'):
                with open('evaluation/latest_metrics.json', 'r', encoding='utf-8') as f:
                    self.model_info["evaluation_metrics"] = json.load(f)
                    
            # 檢查 RAG 知識庫
            kb_path = os.path.join('src', 'knowledge_base', 'sentiment_rules.json')
            if os.path.exists(kb_path):
                with open(kb_path, 'r', encoding='utf-8') as f:
                    kb_data = json.load(f)
                    self.model_info["rag_info"] = {
                        "rules_count": len(kb_data.get("rules", [])),
                        "version": kb_data.get("version", "unknown"),
                        "last_updated": datetime.fromtimestamp(
                            os.path.getmtime(kb_path)
                        ).strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
        except Exception as e:
            logger.error(f"載入模型資訊時發生錯誤: {str(e)}")

    def update_input_text(self, text: str):
        """更新輸入文本"""
        st.session_state.input_text = text
    
    def run(self):
        """運行應用程式"""
        # 初始化 session_state
        if "input_text" not in st.session_state:
            st.session_state.input_text = ""
        
        # 頁面配置
        st.set_page_config(
            page_title="SentiTune-CN",
            page_icon="🎭",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # 主標題
        st.title("SentiTune-CN 中文情感分析系統")
        st.markdown("---")
        
        # 側邊欄
        with st.sidebar:
            st.header("系統資訊")
            st.info(f"""
            ### 模型狀態
            - 最後更新：{self.model_info['last_updated']}
            - 字典大小：{self.model_info['dictionary_size']} 個詞
            
            ### 閾值設定
            - 正面閾值：{self.model_info['threshold_info'].get('positive', 0.7)}
            - 負面閾值：{self.model_info['threshold_info'].get('negative', 0.3)}
            """)
            
            # 評估指標
            if self.model_info["evaluation_metrics"]:
                st.success(f"""
                ### 模型效能
                - 準確率：{self.model_info['evaluation_metrics'].get('accuracy', 0):.2f}
                - 平均誤差：{self.model_info['evaluation_metrics'].get('average_error', 0):.2f}
                - 高信心準確率：{self.model_info['evaluation_metrics'].get('high_confidence_accuracy', 0):.2f}
                """)
            
            # RAG 系統狀態
            if self.model_info.get("rag_info"):
                st.info(f"""
                ### RAG 系統狀態
                - 規則數量：{self.model_info['rag_info'].get('rules_count', 0)}
                - 版本：{self.model_info['rag_info'].get('version', 'unknown')}
                - 最後更新：{self.model_info['rag_info'].get('last_updated', '未知')}
                """)
            
            # 優化按鈕
            if st.button("🔄 執行模型優化", help="執行模型優化流程，可能需要幾分鐘"):
                self.run_optimization()
        
        # 主要分析介面
        col_input, col_example = st.columns([7, 3])
        
        with col_input:
            user_input = st.text_area(
                "請輸入要分析的文字：",
                value=st.session_state.input_text,
                height=150,
                key="input_area",
                help="輸入任何中文文字進行情感分析"
            )
            
        with col_example:
            st.markdown("### 示例文本")
            example_texts = [
                "這個產品真的很讚！用了就愛上了。",
                "服務品質差，態度很不好。",
                "還可以啦，但有改進空間。",
                "哇！真是太厲害了呢！（反諷）"
            ]
            for text in example_texts:
                st.button(
                    text,
                    key=f"example_{text}",
                    on_click=self.update_input_text,
                    args=(text,)
                )
        
        # 分析按鈕
        if st.button("✨ 開始分析", key="analyze_button"):
            if not user_input:
                st.warning("⚠️ 請先輸入要分析的文字！")
                return
                
            with st.spinner("🔄 正在分析中..."):
                result = self.analyzer.analyze(user_input)
                self.history.append((user_input, result))
                
                if result["狀態"] == "成功":
                    # 主要結果顯示
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.subheader("情感分數")
                        st.metric(
                            label="分數", 
                            value=f"{result['情感分數']:.2f}"
                        )
                        
                    with col2:
                        st.subheader("情感類別")
                        sentiment_color = {
                            "正面": "green",
                            "負面": "red",
                            "中性": "gray"
                        }
                        st.markdown(
                            f"<h3 style='color: {sentiment_color[result['情感類別']]}'>"
                            f"{result['情感類別']}</h3>",
                            unsafe_allow_html=True
                        )
                        
                    with col3:
                        st.subheader("信心分數")
                        confidence_color = (
                            "normal" if result['信心分數'] >= 0.7 else
                            "inverse" if result['信心分數'] >= 0.5 else
                            "off"
                        )
                        st.metric(
                            label="信心",
                            value=f"{result['信心分數']:.2f}",
                            delta=None,
                            delta_color=confidence_color
                        )
                    
                    # 信心分數詳情
                    with st.expander("📊 查看信心分數詳情"):
                        st.write("### 信心因素分析")
                        
                        factors = result["信心因素"]
                        for factor, score in factors.items():
                            col_label, col_progress = st.columns([2, 8])
                            with col_label:
                                st.write(f"{factor}:")
                            with col_progress:
                                st.progress(score)
                                st.write(f"{score:.2f}")
                        
                        st.info("""
                        信心因素說明：
                        - 情感極性：文本的情感傾向強度
                        - 字典匹配：與自定義情感字典的匹配程度
                        - 文本長度：文本的充分性評估
                        - 情感一致性：各部分情感評價的一致程度
                        - 規則匹配：與知識庫規則的匹配程度
                        """)
                    
                    # RAG 分析結果
                    if "rag_analysis" in result:
                        with st.expander("🔍 RAG 分析詳情"):
                            st.write("### 規則匹配")
                            if result["rag_analysis"]["matched_rules"]:
                                for rule in result["rag_analysis"]["matched_rules"]:
                                    st.info(f"匹配規則: {rule}")
                            else:
                                st.write("未匹配到特定規則")
                                
                            st.write("### 上下文模式")
                            context_patterns = result["rag_analysis"]["context_patterns"]
                            if context_patterns:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric(
                                        "上下文長度",
                                        context_patterns.get("context_length", 0)
                                    )
                                with col2:
                                    flow = context_patterns.get("sentiment_flow", {})
                                    st.metric(
                                        "情感趨勢",
                                        f"{flow.get('trend', 0):.2f}",
                                        delta=f"波動: {flow.get('volatility', 0):.2f}"
                                    )
                            else:
                                st.write("無上下文模式分析")
                    
                    # 關鍵詞分析
                    st.subheader("關鍵詞分析")
                    if result['關鍵詞']:
                        keyword_cols = st.columns(len(result['關鍵詞']))
                        for i, (keyword, col) in enumerate(zip(result['關鍵詞'], keyword_cols)):
                            with col:
                                st.markdown(f"""
                                <div style='
                                    background-color: rgba(28, 131, 225, 0.1);
                                    border: 1px solid rgba(28, 131, 225, 0.3);
                                    border-radius: 5px;
                                    padding: 10px;
                                    text-align: center;
                                '>
                                    {keyword}
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.write("沒有找到顯著的關鍵詞")
                    
                    # 歷史分析
                    if len(self.history) > 1:
                        with st.expander("📈 查看歷史分析"):
                            history_df = pd.DataFrame([
                                {
                                    "文本": text,
                                    "情感分數": res["情感分數"],
                                    "情感類別": res["情感類別"],
                                    "信心分數": res["信心分數"],
                                    "時間": res["分析時間"]
                                }
                                for text, res in self.history if res["狀態"] == "成功"
                            ])
                            
                            # 繪製趨勢圖
                            chart = alt.Chart(history_df).mark_line().encode(
                                x=alt.X('時間:T', title='分析時間'),
                                y=alt.Y('情感分數:Q', title='情感分數'),
                                tooltip=['文本', '情感分數', '情感類別', '信心分數']
                            ).interactive()
                            
                            st.altair_chart(chart, use_container_width=True)
                            st.dataframe(history_df)
                    
                    # 時間戳記
                    st.caption(f"分析時間：{result['分析時間']}")
                    
                else:
                    st.error(f"分析失敗：{result['錯誤訊息']}")
        
        # 頁尾
        st.markdown("---")
        st.markdown(
            "Made with ❤️ by XinLeiYo | "
            "Last updated: 2025-05-13 17:30:00 UTC"
        )
    
    def run_optimization(self):
        """執行模型優化"""
        try:
            with st.spinner("正在執行模型優化..."):
                # 導入優化模組
                from tune_and_update import TuningManager
                
                # 執行優化
                manager = TuningManager()
                manager.generate_training_data()
                manager.tune_model()
                
                # 重新載入模型資訊
                self.load_model_info()
                
                st.success("✅ 模型優化完成！")
                st.info("請重新整理頁面以使用更新後的模型。")
                
        except Exception as e:
            st.error(f"優化過程發生錯誤：{str(e)}")

def main():
    app = SentimentApp()
    app.run()

if __name__ == "__main__":
    main()