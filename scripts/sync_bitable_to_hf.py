#!/usr/bin/env python3
"""
sync_bitable_to_hf.py
路徑 B：從 Bitable 拉語料 → 更新 HF Space 的 paiwan_corpus.json → git push

執行方式：
  python3 scripts/sync_bitable_to_hf.py [--dry-run] [--push]

--dry-run: 只顯示會做什麼，不寫檔案不 push
--push: 自動 git push（否則只 commit）
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional

import requests
from dotenv import load_dotenv

# 載入環境變數
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

APP_ID = os.getenv("FEISHU_APP_ID", "")
APP_SECRET = os.getenv("FEISHU_APP_SECRET", "")
BITABLE_APP_TOKEN = "DBe0bLbJRaWlRQsSZgJciXrLnIW"

# 表 ID
TABLE_CORPUS = "tblxxEnL08nPkmOk"   # 語料收集表
TABLE_VOCAB = "tbljOpiUGUG8Iim6"     # 詞彙關聯表
TABLE_AFFIX = "tblb1pWYHFyevyav"     # 綴詞規則表

# HF Space 路徑
HF_DEPLOY = PROJECT_ROOT / "hf_deploy"
CORPUS_JSON = HF_DEPLOY / "modules" / "paiwan_corpus.json"


def get_tenant_token() -> str:
    """取得飛書 tenant_access_token"""
    r = requests.post(
        "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
        json={"app_id": APP_ID, "app_secret": APP_SECRET},
        timeout=10,
    )
    data = r.json()
    if data.get("code") != 0:
        raise RuntimeError(f"Token failed: {data}")
    return data["tenant_access_token"]


def fetch_all_records(token: str, table_id: str, page_size: int = 500) -> list:
    """分頁拉取 Bitable 所有記錄"""
    all_records = []
    page_token = None

    while True:
        url = (
            f"https://open.feishu.cn/open-apis/bitable/v1/apps/{BITABLE_APP_TOKEN}"
            f"/tables/{table_id}/records?page_size={page_size}"
        )
        if page_token:
            url += f"&page_token={page_token}"

        r = requests.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        )
        data = r.json()
        if data.get("code") != 0:
            print(f"⚠️ API error for {table_id}: {data.get('msg', 'unknown')}")
            break

        items = data.get("data", {}).get("items", [])
        all_records.extend(items)

        page_token = data.get("data", {}).get("page_token")
        if not page_token:
            break

    return all_records


# 簡單判斷：是否包含中文字
import re
_CJK_RE = re.compile(r'[\u4e00-\u9fff]')

def _is_chinese(text: str) -> bool:
    return bool(_CJK_RE.search(text))

def _normalize(text: str) -> str:
    return text.strip().lower().rstrip('。，！？,.!?')

def corpus_records_to_sentences(records: list) -> list:
    """將語料收集表記錄轉為 corpus sentence 格式，自動判斷中排方向"""
    sentences = []
    seen = set()  # 用 (排灣語, 中文) 去重

    for rec in records:
        fields = rec.get("fields", {})
        text_a = str(fields.get("文本", "")).strip()
        text_b = str(fields.get("中文翻譯", "")).strip()
        source = fields.get("語料來源", "")
        confidence = fields.get("置信度等級", "")

        if not text_a or not text_b:
            continue

        # 只收錄高置信度的
        if confidence and "低" in str(confidence):
            continue

        # 判斷方向：文本欄是排灣語還是中文
        if _is_chinese(text_a) and not _is_chinese(text_b):
            # 反了：文本是中文，翻譯是排灣語
            paiwan, chinese = text_b, text_a
        elif not _is_chinese(text_a) and _is_chinese(text_b):
            # 正常：文本是排灣語，翻譯是中文
            paiwan, chinese = text_a, text_b
        else:
            # 都是中文或都是排灣語，跳過
            continue

        # 品質過濾：跳過多義詞合併結果（含 / 或多個翻譯拼接）
        if "/" in chinese or "/" in paiwan:
            continue
        if "。" in chinese and chinese.count("。 ") > 0:
            continue
        # 跳過太長的翻譯（可能是 LLM 亂翻的）
        if len(paiwan) > 80 or len(chinese) > 80:
            continue

        # 去重
        key = (_normalize(paiwan), _normalize(chinese))
        if key in seen:
            continue
        seen.add(key)

        sentence = {
            "paiwan": paiwan,
            "chinese": chinese,
            "intent": "bitable_sync",
            "grammar_note": f"來源：{source}" if source else "Bitable 同步",
            "confidence": confidence,
            "synced_at": datetime.now().isoformat()[:10],
        }
        sentences.append(sentence)

    return sentences


def vocab_records_to_entries(records: list) -> dict:
    """將詞彙關聯表記錄轉為詞彙展示格式"""
    entries = {}
    for rec in records:
        fields = rec.get("fields", {})
        word = str(fields.get("多行文本", "")).strip()
        meaning = str(fields.get("中文釋義", "")).strip()
        word_type = str(fields.get("詞類", "")).strip()

        if not word or not meaning:
            continue

        entries[word] = {
            "paiwan": word,
            "chinese": meaning,
            "type": word_type or "未分類",
        }

    return entries


def update_corpus_json(
    corpus_path: Path,
    new_sentences: list,
    new_vocab: dict,
) -> dict:
    """更新 corpus JSON，增量追加新語料到社群貢獻分類"""
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    # 收集所有現有語料對（包括 community 分類）
    existing_pairs = set()
    for cat in corpus["categories"].values():
        for s in cat.get("sentences", []):
            p = _normalize(s.get("paiwan", ""))
            c = _normalize(s.get("chinese", ""))
            existing_pairs.add((p, c))

    # 增量去重：只追加不在現有語料中的
    new_unique = []
    for s in new_sentences:
        p = _normalize(s["paiwan"])
        c = _normalize(s["chinese"])
        if (p, c) not in existing_pairs:
            new_unique.append(s)
            existing_pairs.add((p, c))

    # 增量追加到 community 分類（不覆蓋已有的）
    if "community" not in corpus["categories"]:
        corpus["categories"]["community"] = {
            "name": "社群貢獻（Bitable 同步）",
            "icon": "🌐",
            "description": "",
            "sentences": [],
        }

    community = corpus["categories"]["community"]
    existing_community_count = len(community.get("sentences", []))
    community["sentences"].extend(new_unique)
    total_community = len(community["sentences"])

    community["description"] = (
        f"來自飛書 Bot 用戶對話的高品質語料，共 {total_community} 條。"
        f"最後同步：{datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )

    # 更新 metadata
    corpus["metadata"]["last_bitable_sync"] = datetime.now().isoformat()
    corpus["metadata"]["bitable_corpus_count"] = len(new_sentences)
    corpus["metadata"]["bitable_vocab_count"] = len(new_vocab)

    # 加入 Bitable 詞彙統計到 metadata
    if new_vocab:
        corpus["metadata"]["bitable_vocabulary"] = list(new_vocab.values())

    return corpus, new_unique


def main():
    parser = argparse.ArgumentParser(description="Bitable → HF Space 同步")
    parser.add_argument("--dry-run", action="store_true", help="只顯示不寫入")
    parser.add_argument("--push", action="store_true", help="自動 git push")
    args = parser.parse_args()

    print("=" * 60)
    print("🔄 路徑 B：Bitable → HF Space 同步")
    print("=" * 60)

    # 1. 取得 token
    print("\n📡 取得飛書 Token...")
    token = get_tenant_token()
    print("✅ Token 取得成功")

    # 2. 拉取語料
    print("\n📥 拉取語料收集表...")
    corpus_records = fetch_all_records(token, TABLE_CORPUS)
    print(f"   語料記錄: {len(corpus_records)} 筆")

    print("📥 拉取詞彙關聯表...")
    vocab_records = fetch_all_records(token, TABLE_VOCAB)
    print(f"   詞彙記錄: {len(vocab_records)} 筆")

    # 3. 轉換
    print("\n🔄 轉換語料格式...")
    new_sentences = corpus_records_to_sentences(corpus_records)
    new_vocab = vocab_records_to_entries(vocab_records)
    print(f"   有效語料: {len(new_sentences)} 條")
    print(f"   有效詞彙: {len(new_vocab)} 個")

    if args.dry_run:
        print("\n🔍 [DRY RUN] 預覽新語料:")
        for s in new_sentences[:10]:
            print(f"   {s['paiwan']} → {s['chinese']}")
        if len(new_sentences) > 10:
            print(f"   ... 還有 {len(new_sentences) - 10} 條")
        print(f"\n🔍 [DRY RUN] 預覽新詞彙:")
        for w, info in list(new_vocab.items())[:10]:
            print(f"   {w} ({info['type']}) → {info['chinese']}")
        return

    # 4. 更新 corpus JSON
    print("\n📝 更新 paiwan_corpus.json...")
    corpus, new_unique = update_corpus_json(CORPUS_JSON, new_sentences, new_vocab)

    total_sentences = sum(
        len(c.get("sentences", [])) for c in corpus["categories"].values()
    )
    print(f"   原有 42 句 → 總計 {total_sentences} 句")
    print(f"   新增社群貢獻: {len(new_unique)} 條（去重後）")

    with open(CORPUS_JSON, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    print(f"✅ 已寫入 {CORPUS_JSON}")

    # 5. Git commit
    print("\n📦 Git commit...")
    os.chdir(HF_DEPLOY)

    subprocess.run(["git", "add", "modules/paiwan_corpus.json"], check=True)

    commit_msg = (
        f"sync: Bitable 語料同步 ({len(new_unique)} 條社群貢獻, "
        f"總計 {total_sentences} 句, {datetime.now().strftime('%Y-%m-%d %H:%M')})"
    )
    result = subprocess.run(
        ["git", "commit", "-m", commit_msg],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"✅ Committed: {commit_msg}")
    else:
        print(f"⚠️ Commit 結果: {result.stdout.strip()} {result.stderr.strip()}")
        if "nothing to commit" in result.stdout + result.stderr:
            print("   沒有新的改動需要 commit")

    # 6. Git push
    if args.push:
        print("\n🚀 Git push 到 HF Space...")
        result = subprocess.run(
            [
                "git",
                "-c", "http.proxy=http://127.0.0.1:7897",
                "-c", "http.postBuffer=524288000",
                "-c", "http.lowSpeedLimit=0",
                "-c", "http.lowSpeedTime=999999",
                "push", "origin", "main",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            print("✅ Push 成功！HF Space 會自動重建。")
        else:
            print(f"❌ Push 失敗: {result.stderr}")
            print("   手動 push:")
            print(f"   cd {HF_DEPLOY}")
            print("   git -c http.proxy=http://127.0.0.1:7897 -c http.postBuffer=524288000 push origin main")
    else:
        print("\n📋 準備好了。要 push 的話執行：")
        print(f"   cd {HF_DEPLOY}")
        print("   git -c http.proxy=http://127.0.0.1:7897 -c http.postBuffer=524288000 push origin main")

    # 7. 總結
    print("\n" + "=" * 60)
    print("📊 同步總結")
    print("=" * 60)
    print(f"  Bitable 語料記錄: {len(corpus_records)} 筆")
    print(f"  有效語料: {len(new_sentences)} 條")
    print(f"  新增（去重）: {len(new_unique)} 條")
    print(f"  語料總計: {total_sentences} 句")
    print(f"  Bitable 詞彙: {len(new_vocab)} 個")
    print(f"  同步時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
