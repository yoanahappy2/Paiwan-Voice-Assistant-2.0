#!/usr/bin/env python3
"""
run_sft.py
智譜 GLM-4 微調腳本

使用智譯 GLM-4-Flash 免費微調 API
文檔：https://open.bigmodel.cn/dev/api#fine_tuning

作者: Claude
日期: 2026-05-12
"""

import os
import json
import time
import requests
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "scripts" / "sft" / "data"

# 智譜 API 配置
ZHIPU_API_KEY = os.getenv("ZHIPUAI_API_KEY")
API_BASE = "https://open.bigmodel.cn/api/paas/v4"

HEADERS = {
    "Authorization": f"Bearer {ZHIPU_API_KEY}",
    "Content-Type": "application/json"
}


# ============================================
# 智譜微調 API
# ============================================

def create_finetuning_job(
    model: str = "glm-4-flash",
    train_file: str = None,
    val_file: str = None,
    job_name: str = None,
    epoch: int = 3,
    batch_size: int = 32,
    learning_rate: float = 0.0001
) -> dict:
    """
    創建微調任務

    智譯微調 API 文檔：
    https://open.bigmodel.cn/dev/api#fine_tuning
    """
    if job_name is None:
        job_name = f"paiwan-translator-{datetime.now().strftime('%Y%m%d-%H%M')}"

    # 上傳訓練檔案（需要先上傳到 OSS 或使用公開 URL）
    # 智譯支持直接上傳 JSONL 檔案

    print("=" * 60)
    print("🚀 創建智譜 GLM-4 微調任務")
    print("=" * 60)

    # 方法 1: 使用雲端文件（推薦）
    # 需要先上傳檔案到可訪問的 URL

    payload = {
        "model": model,
        "training_file": train_file,  # 檔案 URL 或 file ID
        "validation_file": val_file,  # 可選
        "hyperparameters": {
            "epochs": epoch,
            "batch_size": batch_size,
            "learning_rate_multiplier": learning_rate
        },
        "suffix": "paiwan",  # 微調模型後綴
        "name": job_name
    }

    print(f"模型: {model}")
    print(f"任務名稱: {job_name}")
    print(f"訓練檔案: {train_file}")
    print(f"參數: epochs={epoch}, batch_size={batch_size}, lr={learning_rate}")

    # 實際 API 調用
    # response = requests.post(
    #     f"{API_BASE}/fine_tuning/jobs",
    #     headers=HEADERS,
    #     json=payload
    # )
    # return response.json()

    print("\n⚠️ 注意：智譜微調需要在控制台上傳檔案")
    print("📋 請按照以下步驟操作：")
    print_steps()
    return {}


def print_steps():
    """列出手動操作步驟"""
    print("\n" + "=" * 60)
    print("📋 智譜微調操作步驟")
    print("=" * 60)

    print("""
Step 1: 準備數據
--------
運行以下命令生成訓練數據：
    cd /Users/sbb-mei/Desktop/paiwan_competition_2026
    python scripts/sft/prepare_sft_data.py --output paiwan_sft.jsonl

生成的檔案位置：
    - 訓練集: scripts/sft/data/paiwan_sft_train.jsonl
    - 驗證集: scripts/sft/data/paiwan_sft_val.jsonl

Step 2: 上傳數據到智譜平台
--------
1. 訪問：https://open.bigmodel.cn/usercenter/fine-tuning
2. 點擊「創建微調任務」
3. 上傳 paiwan_sft_train.jsonl（選擇 GLM-4-Flash 模型）
4. （可選）上傳 paiwan_sft_val.jsonl 作為驗證集
5. 填寫任務名稱，如：paiwan-translator-v1

Step 3: 配置訓練參數（推薦設置）
--------
- 模型：glm-4-flash（免費）
- 訓練輪次：3-5 epochs
- 學習率：默認
- Batch size：默認（根據數據量自動調整）

Step 4: 等待訓練完成
--------
- 訓練時間：約 10-30 分鐘（取決於數據量）
- 可以在控制台查看訓練進度
- 訓練完成後會收到通知

Step 5: 獲取微調模型 ID
--------
訓練完成後，模型 ID 格式類似：
    glm-4-flash-paiwan-20250512

Step 6: 測試微調模型
--------
使用提供的測試腳本驗證效果：
    python scripts/sft/test_sft_model.py --model YOUR_MODEL_ID
    """)


def verify_data_format(file_path: Path) -> bool:
    """驗證數據格式是否符合智譜要求"""
    print(f"\n🔍 驗證數據格式: {file_path}")

    if not file_path.exists():
        print(f"❌ 檔案不存在")
        return False

    errors = []
    samples = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: JSON 解析失敗 - {e}")
                continue

            # 驗證格式
            if "messages" not in data:
                errors.append(f"Line {line_num}: 缺少 'messages' 字段")
                continue

            messages = data["messages"]
            if not isinstance(messages, list):
                errors.append(f"Line {line_num}: 'messages' 應該是列表")
                continue

            # 驗證每個 message
            for i, msg in enumerate(messages):
                if "role" not in msg or "content" not in msg:
                    errors.append(f"Line {line_num}, Message {i}: 缺少 'role' 或 'content'")
                elif msg["role"] not in ["system", "user", "assistant"]:
                    errors.append(f"Line {line_num}, Message {i}: 無效的 role '{msg['role']}'")
                elif not msg["content"]:
                    errors.append(f"Line {line_num}, Message {i}: content 為空")

            # 統計
            samples.append(data)

            # 只檢查前 100 行
            if line_num > 100:
                break

    if errors:
        print("❌ 發現錯誤:")
        for error in errors[:10]:
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... 還有 {len(errors) - 10} 個錯誤")
        return False

    print(f"✅ 格式正確！檢查了 {len(samples)} 個樣本")

    # 顯示一個樣本
    if samples:
        print("\n📄 樣本示例:")
        print(json.dumps(samples[0], ensure_ascii=False, indent=2))

    return True


def get_data_stats(file_path: Path) -> dict:
    """獲取數據統計"""
    total = 0
    total_tokens = 0
    roles_count = {"system": 0, "user": 0, "assistant": 0}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            total += 1

            for msg in data.get("messages", []):
                roles_count[msg.get("role", "unknown")] += 1
                total_tokens += len(msg.get("content", ""))

    return {
        "total_samples": total,
        "total_tokens": total_tokens,
        "roles_count": roles_count,
        "avg_tokens_per_sample": total_tokens / total if total > 0 else 0
    }


def main():
    """主函數"""
    import argparse

    parser = argparse.ArgumentParser(description="智譜 GLM-4 微調腳本")
    parser.add_argument("--prepare", action="store_true", help="先準備數據")
    parser.add_argument("--verify", action="store_true", help="驗證數據格式")
    parser.add_argument("--stats", action="store_true", help="顯示數據統計")
    parser.add_argument("--train-file", type=str, help="訓練檔案路徑")
    parser.add_argument("--val-file", type=str, help="驗證檔案路徑")
    parser.add_argument("--model", type=str, default="glm-4-flash", help="基礎模型")
    parser.add_argument("--steps", action="store_true", help="顯示操作步驟")

    args = parser.parse_args()

    # 創建數據目錄
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 顯示步驟
    if args.steps or not any([args.prepare, args.verify, args.stats]):
        print_steps()
        return

    # 準備數據
    if args.prepare:
        print("📝 準備 SFT 數據...")
        import subprocess
        subprocess.run([
            "python",
            str(BASE_DIR / "scripts" / "sft" / "prepare_sft_data.py"),
            "--output", "paiwan_sft.jsonl"
        ])
        return

    # 驗證數據
    if args.verify:
        train_file = DATA_DIR / "paiwan_sft_train.jsonl"
        val_file = DATA_DIR / "paiwan_sft_val.jsonl"

        print("\n驗證訓練集:")
        train_ok = verify_data_format(train_file)

        print("\n驗證驗證集:")
        val_ok = verify_data_format(val_file)

        if train_ok and val_ok:
            print("\n✅ 數據驗證通過！可以上傳到智譜平台進行微調。")

    # 顯示統計
    if args.stats:
        train_file = DATA_DIR / "paiwan_sft_train.jsonl"
        val_file = DATA_DIR / "paiwan_sft_val.jsonl"

        if train_file.exists():
            stats = get_data_stats(train_file)
            print("\n訓練集統計:")
            print(f"  樣本數: {stats['total_samples']}")
            print(f"  總字符數: {stats['total_tokens']}")
            print(f"  平均每樣本: {stats['avg_tokens_per_sample']:.0f} 字符")

        if val_file.exists():
            stats = get_data_stats(val_file)
            print("\n驗證集統計:")
            print(f"  樣本數: {stats['total_samples']}")
            print(f"  總字符數: {stats['total_tokens']}")


if __name__ == "__main__":
    main()
