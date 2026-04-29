"""
retrain_pipeline.py — 主動學習重訓 Pipeline（原型）

功能：
1. 從 active_learning_log.jsonl 中讀取已驗證的語料
2. 合併到主語料庫 merged_corpus.json
3. 重建 RAG 索引（FAISS）

使用方式：
    python3 retrain_pipeline.py [--dry-run]

引用：
- Synthetic Voice Data for ASR in African Languages (DeRenzi et al. 2025)
  提出 LLM→TTS→ASR 的合成數據管線，我們借鑑其「持續擴充」理念

作者: 地陪
日期: 2026-04-29
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path("/Users/sbb-mei/Desktop/paiwan_competition_2026")
DATA_DIR = PROJECT_ROOT / "data"
CORPUS_PATH = DATA_DIR / "merged_corpus.json"
FEEDBACK_LOG = DATA_DIR / "active_learning_log.json"
VERIFIED_OUTPUT = DATA_DIR / "community_verified_corpus.json"


def load_corpus():
    """載入主語料庫"""
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("entries", data) if isinstance(data, dict) else data
    return entries, data.get("metadata", {}) if isinstance(data, dict) else {}


def load_verified():
    """從學習日誌中讀取已驗證的語料"""
    verified = []
    log_path = DATA_DIR / "active_learning_log.jsonl"
    
    if not log_path.exists():
        print("📋 學習日誌不存在，沒有新語料")
        return verified

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                # 只接受用戶確認正確的翻譯
                if entry.get("user_feedback") == "✅" and entry.get("method") in ("exact", "rag_llm"):
                    verified.append({
                        "paiwan": entry["input"],
                        "chinese": entry["translation"],
                        "source": f"社區驗證_{entry.get('source', 'unknown')}",
                        "verified_at": entry.get("timestamp", ""),
                    })
            except json.JSONDecodeError:
                continue

    return verified


def merge_and_deduplicate(existing_entries, new_entries):
    """合併並去重"""
    existing_pairs = set()
    for e in existing_entries:
        pair = (e.get("paiwan", "").strip(), e.get("chinese", "").strip())
        existing_pairs.add(pair)

    added = 0
    for ne in new_entries:
        pair = (ne.get("paiwan", "").strip(), ne.get("chinese", "").strip())
        if pair not in existing_pairs and pair[0] and pair[1]:
            existing_entries.append(ne)
            existing_pairs.add(pair)
            added += 1

    return existing_entries, added


def rebuild_rag_index(corpus_entries):
    """
    重建 RAG 索引（FAISS + embedding）
    
    注意：這個步驟需要智譜 API 來計算 embedding
    """
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from translate_service import PaiwanTranslator
        
        print("🔄 正在重建 RAG 索引...")
        translator = PaiwanTranslator()
        translator.load()
        print(f"✅ RAG 索引已重建（{len(corpus_entries)} 條語料）")
        return True
    except Exception as e:
        print(f"⚠️ RAG 索引重建失敗: {e}")
        print("   可以稍後手動重建")
        return False


def run_pipeline(dry_run=False):
    """
    執行完整的重訓 pipeline
    
    Args:
        dry_run: 只顯示會做什麼，不實際修改
    """
    print("=" * 50)
    print("  🔄 主動學習重訓 Pipeline")
    print("=" * 50)
    print()

    # 1. 載入現有語料
    print("📖 步驟 1：載入現有語料庫...")
    existing, metadata = load_corpus()
    print(f"   現有語料：{len(existing)} 條")

    # 2. 載入已驗證的新語料
    print("\n📋 步驟 2：讀取已驗證的社區語料...")
    verified = load_verified()
    print(f"   已驗證的新語料：{len(verified)} 條")

    if not verified:
        print("\n✅ 沒有新的已驗證語料，無需更新。")
        return

    # 3. 合併去重
    print("\n🔀 步驟 3：合併並去重...")
    merged, added = merge_and_deduplicate(existing, verified)
    print(f"   新增：{added} 條")
    print(f"   去重跳過：{len(verified) - added} 條")
    print(f"   合併後總計：{len(merged)} 條")

    if dry_run:
        print("\n🏃 Dry run 模式，不寫入檔案。")
        print(f"   如果正式執行，將從 {len(existing)} → {len(merged)} 條")
        return

    # 4. 寫回語料庫
    print("\n💾 步驟 4：更新語料庫...")
    backup_path = DATA_DIR / f"merged_corpus_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # 備份
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        backup_data = f.read()
    with open(backup_path, "w", encoding="utf-8") as f:
        f.write(backup_data)
    print(f"   備份：{backup_path.name}")

    # 寫入
    corpus_data = {
        "entries": merged,
        "metadata": {
            **metadata,
            "last_updated": datetime.now().isoformat(),
            "community_verified_count": added,
        }
    }
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus_data, f, ensure_ascii=False, indent=2)
    print(f"   ✅ 語料庫已更新：{CORPUS_PATH.name}")

    # 5. 重建 RAG 索引
    print("\n🔍 步驟 5：重建 RAG 索引...")
    rebuild_rag_index(merged)

    # 6. 統計
    print("\n" + "=" * 50)
    print("  📊 Pipeline 完成")
    print("=" * 50)
    print(f"  原有語料：{len(existing)} 條")
    print(f"  新增語料：+{added} 條")
    print(f"  當前總計：{len(merged)} 條")
    print(f"  成長率：+{added/len(existing)*100:.1f}%")
    print()


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    run_pipeline(dry_run=dry_run)
