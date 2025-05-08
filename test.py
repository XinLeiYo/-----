import pandas as pd
import random
from datetime import datetime

def generate_training_data():
    # 正面評論模板
    positive_templates = [
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
        "品質穩穩的"
    ]

    # 負面評論模板
    negative_templates = [
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
        "氣死人了"
    ]

    # 反諷評論模板（情感偏負面）
    sarcastic_templates = [
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
        "太專業了吧"
    ]

    # 中性評論模板
    neutral_templates = [
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
        "還算ok"
    ]

    # 產品/服務關鍵詞
    subjects = [
        "這款APP", "這個遊戲", "這個直播", "這支影片",
        "這個商品", "這家店", "這個外送", "這個服務",
        "這個課程", "這個平台", "這個功能", "這個設計",
        "這個更新", "這個介面", "這款手機", "這個系統",
        "這個老師", "這個客服", "這個主播", "這個博主"
    ]

    # 生成訓練數據
    data = []
    
    # 生成正面評論
    for _ in range(100):
        template = random.choice(positive_templates)
        subject = random.choice(subjects)
        text = f"{subject}{template}" if random.random() > 0.5 else f"{template}！{subject}"
        data.append({
            "text": text,
            "sentiment": round(random.uniform(0.7, 1.0), 2)
        })

    # 生成負面評論
    for _ in range(100):
        template = random.choice(negative_templates)
        subject = random.choice(subjects)
        text = f"{subject}{template}" if random.random() > 0.5 else f"{template}！{subject}"
        data.append({
            "text": text,
            "sentiment": round(random.uniform(0.0, 0.3), 2)
        })

    # 生成反諷評論（偏負面情感）
    for _ in range(100):
        template = random.choice(sarcastic_templates)
        subject = random.choice(subjects) if random.random() > 0.7 else ""  # 有30%機率不帶主題
        text = f"{subject} {template}" if subject else template
        data.append({
            "text": text,
            "sentiment": round(random.uniform(0.1, 0.4), 2)  # 反諷通常偏負面
        })

    # 生成中性評論
    for _ in range(100):
        template = random.choice(neutral_templates)
        subject = random.choice(subjects)
        text = f"{subject}{template}" if random.random() > 0.5 else f"{template}，{subject}"
        data.append({
            "text": text,
            "sentiment": round(random.uniform(0.3, 0.7), 2)
        })

    # 轉換為DataFrame並保存
    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True)  # 隨機打亂順序
    
    # 加入時間戳記
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f'training_data_{timestamp}.csv'
    df.to_csv(filename, index=False, encoding='utf-8')
    
    # 顯示部分樣本
    print(f"\n已生成訓練數據檔案：{filename}")
    print("\n展示部分樣本：")
    print("\n正面評論：")
    print(df[df['sentiment'] > 0.7].head(3))
    print("\n反諷評論：")
    print(df[(df['sentiment'] >= 0.1) & (df['sentiment'] <= 0.4)].head(3))
    print("\n負面評論：")
    print(df[df['sentiment'] < 0.3].head(3))
    print("\n中性評論：")
    print(df[(df['sentiment'] >= 0.3) & (df['sentiment'] <= 0.7)].head(3))
    
    return df

if __name__ == "__main__":
    print(f"開始生成訓練數據 - {datetime.utcnow()}")
    df = generate_training_data()
    print(f"完成訓練數據生成 - {datetime.utcnow()}")