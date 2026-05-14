#!/usr/bin/env python3
"""
prepare_sft_data.py
將排灣語語料轉換為智譜 GLM-4 微調格式

智譜微調數據格式：
{
  "messages": [
    {"role": "system", "content": "你是專業的排灣語翻譯助手"},
    {"role": "user", "content": "請將「...」翻譯成排灣語"},
    {"role": "assistant", "content": "..."}
  ]
}

作者: Claude
日期: 2026-05-12
"""

import json
import random
from pathlib import Path
from typing import List, Dict

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "scripts" / "sft" / "data"

SYSTEM_PROMPTS = [
    "你是專業的排灣語-中文翻譯助手。請根據用戶的請求，準確地進行雙向翻譯。",
    "你精通排灣語和中文，可以進行雙向翻譯。請直接輸出翻譯結果，不要解釋。",
    "你是排灣語翻譯專家，請將輸入翻譯成正確的排灣語或中文。",
]


def load_merged_corpus() -> List[Dict]:
    """載入 merged_corpus.json"""
    corpus_file = DATA_DIR / "merged_corpus.json"

    if not corpus_file.exists():
        raise FileNotFoundError(f"找不到語料檔案: {corpus_file}")

    with open(corpus_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "entries" in raw:
        return raw["entries"]
    elif isinstance(raw, list):
        return raw
    else:
        raise ValueError(f"未知的語料格式: {type(raw)}")


def clean_text(text: str) -> str:
    """清理文本"""
    text = text.strip()
    # 移除前綴標記 (B: A: 等)
    if len(text) > 2 and text[1] == ":":
        text = text[2:].strip()
    # 移除句尾標點（視情況保留）
    text = text.rstrip("。！？：；")
    return text


def create_translation_sample(paiwan: str, chinese: str) -> Dict:
    """創建一個翻譯樣本（雙向）"""

    # 隨機選擇 system prompt
    system_prompt = random.choice(SYSTEM_PROMPTS)

    # 隨機選擇翻譯方向
    if random.random() < 0.5:
        # 排灣語 → 中文
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"請將以下排灣語翻譯成中文：{paiwan}"},
                {"role": "assistant", "content": chinese}
            ]
        }
    else:
        # 中文 → 排灣語
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"請將以下中文翻譯成排灣語：{chinese}"},
                {"role": "assistant", "content": paiwan}
            ]
        }


def create_grammar_sample(paiwan: str, chinese: str) -> Dict:
    """創建語法學習樣本

    這類樣本幫助模型學習排灣語的語法結構
    """
    system_prompt = "你是排灣語語法老師。請分析排灣語的語法結構，並解釋如何構成句子。"

    # 嘗試分析一些語法特徵
    grammar_notes = []

    # 檢測時態標記
    if "na" in paiwan and "sun" in paiwan:
        grammar_notes.append("「na...sun」是現在進行式")

    # 檢測否定
    if "ini" in paiwan:
        grammar_notes.append("「ini」是否定詞")

    # 檢測疑問
    if "?" in paiwan or any(w in paiwan for w in ["pida", "tima", "anema"]):
        grammar_notes.append("句中有疑問詞")

    if grammar_notes:
        explanation = "；".join(grammar_notes)
    else:
        explanation = "這是一個基本的排灣語句子"

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"請分析這個排灣語句子的語法：{paiwan}（意思：{chinese}）"
            },
            {
                "role": "assistant",
                "content": f"這個句子的語法分析：{explanation}。理解這些結構有助於掌握排灣語的基本規律。"
            }
        ]
    }


def create_conversational_sample(paiwan: str, chinese: str) -> Dict:
    """創建對話式樣本

    模擬真實對話場景，讓模型學習自然表達
    """
    system_prompt = "你是友善的排灣語老師。用戶想學排灣語，請用自然的方式回答問題。"

    # 隨機選擇對話場景
    scenarios = [
        {
            "user": f"「{chinese}」用排灣語怎麼說？",
            "assistant": f"「{chinese}」用排灣語說是「{paiwan}」。"
        },
        {
            "user": f"我想學「{chinese}」這句話的排灣語",
            "assistant": f"沒問題！「{chinese}」的排灣語是「{paiwan}」。你可以跟著我練習：{paiwan}。"
        },
        {
            "user": f"排灣語的「{paiwan}」是什麼意思？",
            "assistant": f"「{paiwan}」的意思是「{chinese}」。"
        }
    ]

    scenario = random.choice(scenarios)

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": scenario["user"]},
            {"role": "assistant", "content": scenario["assistant"]}
        ]
    }


def prepare_sft_dataset(
    corpus: List[Dict],
    output_path: Path,
    sample_type: str = "mixed",
    train_ratio: float = 0.9,
    seed: int = 42
) -> None:
    """
    準備 SFT 數據集

    Args:
        corpus: 原始語料
        output_path: 輸出檔案路徑
        sample_type: 樣本類型 ("translation", "grammar", "conversational", "mixed")
        train_ratio: 訓練集比例
        seed: 隨機種子
    """
    random.seed(seed)

    samples = []

    for item in corpus:
        paiwan = clean_text(item.get("paiwan", ""))
        chinese = clean_text(item.get("chinese", ""))

        # 跳過無效數據
        if not paiwan or not chinese:
            continue
        if chinese.lower() == "undefined":
            continue
        if len(paiwan) < 2 or len(chinese) < 1:
            continue

        # 根據類型創建樣本
        if sample_type == "translation":
            samples.append(create_translation_sample(paiwan, chinese))
        elif sample_type == "grammar":
            samples.append(create_grammar_sample(paiwan, chinese))
        elif sample_type == "conversational":
            samples.append(create_conversational_sample(paiwan, chinese))
        elif sample_type == "mixed":
            # 混合模式：70% 翻譯，15% 語法，15% 對話
            rand = random.random()
            if rand < 0.7:
                samples.append(create_translation_sample(paiwan, chinese))
            elif rand < 0.85:
                samples.append(create_grammar_sample(paiwan, chinese))
            else:
                samples.append(create_conversational_sample(paiwan, chinese))

    # 打亂順序
    random.shuffle(samples)

    # 分割訓練集和驗證集
    split_idx = int(len(samples) * train_ratio)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    print(f"✅ 數據準備完成：")
    print(f"   - 總樣本數: {len(samples)}")
    print(f"   - 訓練集: {len(train_samples)}")
    print(f"   - 驗證集: {len(val_samples)}")

    # 保存訓練集
    train_path = output_path.parent / f"{output_path.stem}_train{output_path.suffix}"
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
    print(f"   - 訓練集已保存: {train_path}")

    # 保存驗證集
    val_path = output_path.parent / f"{output_path.stem}_val{output_path.suffix}"
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_samples, f, ensure_ascii=False, indent=2)
    print(f"   - 驗證集已保存: {val_path}")

    return len(samples)


def analyze_corpus(corpus: List[Dict]) -> None:
    """分析語料統計"""
    print("\n" + "=" * 50)
    print("📊 語料分析")
    print("=" * 50)

    total = len(corpus)
    valid = 0
    avg_paiwan_len = 0
    avg_chinese_len = 0

    for item in corpus:
        paiwan = clean_text(item.get("paiwan", ""))
        chinese = clean_text(item.get("chinese", ""))

        if paiwan and chinese and chinese.lower() != "undefined":
            valid += 1
            avg_paiwan_len += len(paiwan)
            avg_chinese_len += len(chinese)

    print(f"總筆數: {total}")
    print(f"有效筆數: {valid}")
    print(f"平均排灣語長度: {avg_paiwan_len / valid:.1f} 字符")
    print(f"平均中文長度: {avg_chinese_len / valid:.1f} 字符")

    # 來源分析
    sources = {}
    for item in corpus:
        source = item.get("source", "unknown")
        sources[source] = sources.get(source, 0) + 1

    print(f"\n來源分佈:")
    for source, count in sorted(sources.items(), key=lambda x: -x[1])[:5]:
        print(f"  - {source}: {count}")


def main():
    """主函數"""
    import argparse

    parser = argparse.ArgumentParser(description="準備排灣語 SFT 數據")
    parser.add_argument(
        "--sample-type",
        choices=["translation", "grammar", "conversational", "mixed"],
        default="mixed",
        help="樣本類型"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="訓練集比例"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="paiwan_sft_data.jsonl",
        help="輸出檔案名稱"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="只分析語料，不生成數據"
    )

    args = parser.parse_args()

    # 創建輸出目錄
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 載入語料
    print("📖 載入語料...")
    corpus = load_merged_corpus()
    print(f"✅ 載入 {len(corpus)} 筆語料")

    # 分析語料
    analyze_corpus(corpus)

    if args.analyze:
        return

    # 準備 SFT 數據
    output_path = OUTPUT_DIR / args.output

    # 如果是 jsonl 格式
    if output_path.suffix == ".jsonl":
        # 智譜微調支持 jsonl 格式，每行一個 JSON
        print(f"\n📝 生成 {args.sample_type} 樣本...")

        random.seed(42)
        samples = []

        for item in corpus:
            paiwan = clean_text(item.get("paiwan", ""))
            chinese = clean_text(item.get("chinese", ""))

            if not paiwan or not chinese:
                continue
            if chinese.lower() == "undefined":
                continue

            # 混合模式
            rand = random.random()
            if rand < 0.7:
                sample = create_translation_sample(paiwan, chinese)
            elif rand < 0.85:
                sample = create_grammar_sample(paiwan, chinese)
            else:
                sample = create_conversational_sample(paiwan, chinese)

            samples.append(sample)

        # 打亂
        random.shuffle(samples)

        # 分割
        split_idx = int(len(samples) * args.train_ratio)
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]

        # 保存 jsonl（智譜推薦格式）
        train_path = OUTPUT_DIR / f"{output_path.stem}_train.jsonl"
        val_path = OUTPUT_DIR / f"{output_path.stem}_val.jsonl"

        with open(train_path, "w", encoding="utf-8") as f:
            for sample in train_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        with open(val_path, "w", encoding="utf-8") as f:
            for sample in val_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        print(f"✅ 完成：")
        print(f"   - 訓練集: {train_path} ({len(train_samples)} 樣本)")
        print(f"   - 驗證集: {val_path} ({len(val_samples)} 樣本)")

    else:
        # JSON 格式
        prepare_sft_dataset(
            corpus,
            output_path,
            sample_type=args.sample_type,
            train_ratio=args.train_ratio
        )


if __name__ == "__main__":
    main()
