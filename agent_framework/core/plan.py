"""
plan.py — 長程任務的 Goal 和 Plan 數據結構

三種長程任務方法（唐杰老師）的數據基礎：
- Plan-Driven：先生成 Plan 再逐步執行
- Self-Judge / Goal-Driven：設定 Goal，循環執行直到達標
- ReAct Loop：Orchestrator 基本運作

作者: 地陪
日期: 2026-05-12
"""

import json
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Goal:
    """
    長程任務的終極目標

    範例：
        goal = Goal(
            name="完善排灣語語料庫",
            description="提升翻譯品質和語料覆蓋率",
            success_criteria={
                "corpus_coverage": {"min": 0.8, "current": 0.0},
                "translation_quality": {"min": 0.85, "current": 0.0},
            },
            max_duration_hours=20,
            max_budget_tokens=5000000,
        )
    """
    name: str
    description: str
    success_criteria: dict = field(default_factory=dict)
    max_duration_hours: float = 20.0
    max_budget_tokens: int = 5000000

    def is_achieved(self, metrics: dict) -> bool:
        """檢查是否達成目標"""
        for key, criterion in self.success_criteria.items():
            target = criterion.get("min", 0) if isinstance(criterion, dict) else criterion
            current = metrics.get(key, 0)
            if isinstance(current, dict):
                current = current.get("current", 0)
            if current < target:
                return False
        return True

    def get_progress(self, metrics: dict) -> dict:
        """獲取各指標的達成進度"""
        progress = {}
        for key, criterion in self.success_criteria.items():
            target = criterion.get("min", 0) if isinstance(criterion, dict) else criterion
            current = metrics.get(key, 0)
            if isinstance(current, dict):
                current = current.get("current", 0)
            progress[key] = {
                "current": current,
                "target": target,
                "ratio": round(current / target, 2) if target > 0 else 0,
            }
        return progress

    def should_terminate(self, metrics: dict, elapsed_hours: float,
                         tokens_used: int) -> tuple:
        """檢查是否應終止（超時/超支）"""
        if elapsed_hours >= self.max_duration_hours:
            return True, f"超過最大運行時間 ({self.max_duration_hours}h)"
        if tokens_used >= self.max_budget_tokens:
            return True, f"超過 Token 預算 ({self.max_budget_tokens})"
        return False, ""

    def to_dict(self) -> dict:
        try:
            return json.loads(json.dumps(asdict(self), default=str, ensure_ascii=False))
        except (RecursionError, TypeError):
            return {"name": self.name, "description": self.description}

    @classmethod
    def from_dict(cls, data: dict) -> "Goal":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PlanStep:
    """計劃中的單一步驟"""
    id: str = ""
    name: str = ""
    description: str = ""
    agent: str = "knowledge"        # 負責的 agent
    task_type: str = "rag_search"   # 任務類型
    params: dict = field(default_factory=dict)
    dependencies: list = field(default_factory=list)
    status: str = "pending"         # pending | in_progress | done | failed | skipped
    result: dict = field(default_factory=dict)
    error: str = ""
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> dict:
        try:
            return json.loads(json.dumps(asdict(self), default=str, ensure_ascii=False))
        except (RecursionError, TypeError):
            return {"id": self.id, "name": self.name, "status": self.status}

    @classmethod
    def from_dict(cls, data: dict) -> "PlanStep":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Plan:
    """
    完整的執行計劃

    由 Teaching Agent 生成，由 Orchestrator 逐步執行。
    支援：依賴檢查、進度追蹤、斷點恢復。
    """
    goal_name: str = ""
    steps: list = field(default_factory=list)  # list[dict] — PlanStep 的 dict 表示
    created_at: float = field(default_factory=time.time)
    status: str = "created"          # created | running | paused | completed | failed
    current_step_index: int = 0

    def get_ready_steps(self) -> list:
        """獲取當前可執行的步驟（依賴已滿足）"""
        done_ids = {s.get("id", "") for s in self.steps if s.get("status") == "done"}
        ready = []
        for s in self.steps:
            if s.get("status") != "pending":
                continue
            deps = s.get("dependencies", [])
            if all(d in done_ids for d in deps):
                ready.append(s)
        return ready

    def get_current_step(self) -> Optional[dict]:
        """獲取當前步驟"""
        for s in self.steps:
            if s.get("status") == "in_progress":
                return s
        return None

    def get_next_step(self) -> Optional[dict]:
        """獲取下一個待執行的步驟"""
        ready = self.get_ready_steps()
        return ready[0] if ready else None

    def mark_step(self, step_id: str, status: str, result: dict = None, error: str = ""):
        """更新步驟狀態"""
        for s in self.steps:
            if s.get("id") == step_id:
                s["status"] = status
                if result:
                    s["result"] = result
                if error:
                    s["error"] = error
                break

    def get_progress(self) -> dict:
        """獲取執行進度"""
        total = len(self.steps)
        done = sum(1 for s in self.steps if s.get("status") == "done")
        failed = sum(1 for s in self.steps if s.get("status") == "failed")
        return {
            "total": total,
            "done": done,
            "failed": failed,
            "progress": round(done / total, 2) if total > 0 else 0,
        }

    def is_complete(self) -> bool:
        return all(s.get("status") in ("done", "skipped") for s in self.steps)

    def to_dict(self) -> dict:
        try:
            return json.loads(json.dumps(asdict(self), default=str, ensure_ascii=False))
        except (RecursionError, TypeError):
            return {"goal_name": self.goal_name, "steps": self.steps, "status": self.status}

    @classmethod
    def from_dict(cls, data: dict) -> "Plan":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_llm_output(cls, steps: list, goal_name: str = "") -> "Plan":
        """從 LLM 生成的步驟列表建立 Plan"""
        plan_steps = []
        for i, step in enumerate(steps):
            plan_steps.append({
                "id": f"step_{i+1}",
                "name": step.get("name", f"步驟 {i+1}"),
                "description": step.get("description", ""),
                "agent": step.get("agent", "knowledge"),
                "task_type": step.get("task_type", "rag_search"),
                "params": step.get("params", {}),
                "dependencies": step.get("dependencies", []),
                "status": "pending",
                "result": {},
                "error": "",
                "retry_count": 0,
                "max_retries": 3,
            })
        return cls(goal_name=goal_name, steps=plan_steps, status="created")


# ============================================
# 測試
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("  📋 Goal & Plan 測試")
    print("=" * 60)

    # 測試 Goal
    goal = Goal(
        name="完善排灣語語料庫",
        description="提升翻譯品質和語料覆蓋率",
        success_criteria={
            "corpus_coverage": {"min": 0.8, "current": 0.0},
            "translation_quality": {"min": 0.85, "current": 0.0},
        },
    )

    print(f"\n  目標: {goal.name}")
    print(f"  已達成: {goal.is_achieved({'corpus_coverage': 0.5, 'translation_quality': 0.6})}")
    print(f"  進度: {goal.get_progress({'corpus_coverage': 0.5, 'translation_quality': 0.6})}")

    metrics = {"corpus_coverage": 0.9, "translation_quality": 0.88}
    print(f"  已達成 (高分): {goal.is_achieved(metrics)}")
    print(f"  進度 (高分): {goal.get_progress(metrics)}")

    # 測試 Plan
    plan = Plan.from_llm_output([
        {"name": "搜尋語料", "agent": "knowledge", "task_type": "rag_search"},
        {"name": "翻譯詞彙", "agent": "knowledge", "task_type": "translate"},
        {"name": "審核品質", "agent": "quality", "task_type": "review_translation"},
    ], goal_name="測試計畫")

    print(f"\n  計畫: {plan.goal_name}")
    print(f"  步驟數: {len(plan.steps)}")
    print(f"  進度: {plan.get_progress()}")
    print(f"  下一個: {plan.get_next_step()}")

    plan.mark_step("step_1", "done", {"results": "ok"})
    print(f"  進度 (step_1 done): {plan.get_progress()}")
    print(f"  下一個: {plan.get_next_step().get('name') if plan.get_next_step() else 'None'}")

    # JSON 往返
    plan_dict = plan.to_dict()
    plan_restored = Plan.from_dict(plan_dict)
    assert len(plan_restored.steps) == len(plan.steps)
    print(f"\n  ✅ JSON 往返測試通過")
