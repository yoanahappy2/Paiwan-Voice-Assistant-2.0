#!/usr/bin/env python3
"""
run_autonomous.py — 啟動自主運行的 Multi-Agent 系統

用法：
    # 啟動新任務
    python run_autonomous.py --goal "完善排灣語語料庫"

    # 從 checkpoint 恢復
    python run_autonomous.py --resume

    # 指定運行時長
    python run_autonomous.py --goal "學排灣語問候" --max-hours 2

    # 停止運行中的系統
    touch agent_framework/storage/STOP.signal

作者: 地陪
日期: 2026-05-12
"""

import argparse
import logging
import sys
from pathlib import Path

# 確保可以 import
sys.path.insert(0, str(Path(__file__).parent))

from agent_framework.core.loop import MainLoop, LoopConfig
from agent_framework.core.plan import Goal


def main():
    parser = argparse.ArgumentParser(description="🚀 排灣語 Multi-Agent 自主運行系統")
    parser.add_argument("--goal", type=str, help="任務目標描述")
    parser.add_argument("--resume", action="store_true", help="從 checkpoint 恢復")
    parser.add_argument("--max-hours", type=float, default=20, help="最大運行時間（小時）")
    parser.add_argument("--max-tokens", type=int, default=5000000, help="Token 預算上限")
    parser.add_argument("--base-wait", type=float, default=2.0, help="基礎等待秒數")
    parser.add_argument("--debug", action="store_true", help="開啟 DEBUG 日誌")

    args = parser.parse_args()

    # 日誌設定
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("  🏔️ 語聲同行 2.0 — Multi-Agent 自主運行系統")
    print("  Agent = LLM + Memory + Planning + Tool Use")
    print("=" * 60)
    print()

    # 建立 Goal
    if args.goal:
        goal = Goal(
            name=args.goal,
            description=args.goal,
            success_criteria={
                # 可以根據目標動態設定
                "iterations_completed": {"min": 10, "current": 0},
            },
            max_duration_hours=args.max_hours,
            max_budget_tokens=args.max_tokens,
        )
        print(f"  🎯 目標: {goal.name}")
        print(f"  ⏱ 最大時長: {goal.max_duration_hours}h")
        print(f"  💰 Token 預算: {goal.max_budget_tokens:,}")
    elif args.resume:
        goal = None  # 從 checkpoint 恢復，goal 在裡面
        print(f"  📂 從 checkpoint 恢復")
    else:
        print("  ❌ 請指定 --goal 或 --resume")
        parser.print_help()
        sys.exit(1)

    print()

    # 配置
    config = LoopConfig(
        base_wait=args.base_wait,
        checkpoint_interval=60,
    )

    # 啟動
    loop = MainLoop(goal=goal, config=config)
    loop.run(resume=args.resume)


if __name__ == "__main__":
    main()
