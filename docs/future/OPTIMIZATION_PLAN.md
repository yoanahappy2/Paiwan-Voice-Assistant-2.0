# Agent Framework 優化草案
## 實現 20+ 小時自主運行的長程任務架構

**目標**：讓 Agent 系統能夠自主運行 20+ 小時，無需人工干預
**依據**：唐杰老師提到的三種長程任務方法（Plan-Driven、Goal-Driven、自主 Loop）
**日期**：2026-05-12

---

## 一、整體架構設計

### 1.1 核心運作流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    MainLoop（主控制器）                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  while not should_stop():                               │   │
│  │      1. health_check()      # 自檢：死鎖、資源、異常    │   │
│  │      2. load_state()        # 從持久化載入當前狀態      │   │
│  │      3. get_next_task()     # 任務隊列/Plan 獲取        │   │
│  │      4. execute_task()      # 執行 → Self-Judge         │   │
│  │      5. save_checkpoint()   # 保存進度                  │   │
│  │      6. adaptive_wait()     # 動態調整循環間隔          │   │
│  │  end                                                       │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PlanExecutor（計劃執行器）                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Plan: {                                                 │   │
│  │    goal: "完善排灣語語料庫",                              │   │
│  │    steps: [                                              │   │
│  │      {type: "scrape", status: "done"},                  │   │
│  │      {type: "translate", status: "in_progress"},        │   │
│  │      {type: "validate", status: "pending"},             │   │
│  │    ],                                                     │   │
│  │    current_step: 1,                                      │   │
│  │    metrics: {coverage: 0.65, quality_score: 0.82},      │   │
│  │  }                                                        │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SelfJudge（自我監督）                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  evaluate(goal, current_state) → judgment               │   │
│  │                                                          │   │
│  │  judgment = {                                            │   │
│  │    achieved: false,                                     │   │
│  │    confidence: 0.65,                                    │   │
│  │    gap: "語料覆蓋率不足，需要增加日常用語",              │   │
│  │    next_action: "continue",  # continue/adjust/abort    │   │
│  │    suggested_adjustment: {...},                         │   │
│  │  }                                                       │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 檔案組織結構

```
agent_framework/
├── core/
│   ├── agent.py           # ✅ 已有：BaseAgent
│   ├── message.py         # ✅ 已有：AgentMessage, MessageBus
│   ├── rate_limiter.py    # ✅ 已有：APIGuard
│   ├── state.py           # 🆕 StateManager（持久化）
│   ├── loop.py            # 🆕 MainLoop（主控制器）
│   ├── plan.py            # 🆕 Plan, PlanExecutor
│   ├── judge.py           # 🆕 SelfJudge, Goal
│   ├── health.py          # 🆕 HealthChecker
│   └── task_queue.py      # 🆕 TaskQueue（任務排程）
├── agents/
│   ├── orchestrator.py    # 🆕 總管 Agent
│   ├── knowledge.py       # 🆕 知識 Agent
│   ├── teaching.py        # 🆕 教學 Agent
│   └── quality.py         # 🆕 品質 Agent
├── storage/
│   ├── state_db.py        # 🆕 狀態持久化（JSON/SQLite）
│   └── checkpoint.py      # 🆕 Checkpoint 管理
├── monitoring/
│   ├── metrics.py         # 🆕 指標收集
│   └── dashboard.py       # 🆕 可觀測性（可選）
└── config/
    └── loop_config.yaml   # 🆕 Loop 配置
```

---

## 二、核心數據結構

### 2.1 Goal（目標定義）

```python
@dataclass
class Goal:
    """長程任務的終極目標"""

    name: str                          # 目標名稱
    description: str                   # 自然語言描述
    success_criteria: dict             # 成功條件（可量化）
    termination_criteria: dict         # 終止條件（安全邊界）
    max_duration_hours: float          # 最長運行時間
    max_budget_tokens: int             # Token 預算上限

    # 成功條件範例：
    # {
    #     "corpus_coverage": {"min": 0.8, "current": 0.0},
    #     "translation_quality": {"min": 0.85, "current": 0.0},
    #     "active_learning_iterations": {"min": 5, "current": 0},
    # }

    def is_achieved(self, metrics: dict) -> bool:
        """檢查是否達成目標"""
        for key, criterion in self.success_criteria.items():
            if metrics.get(key, 0) < criterion["min"]:
                return False
        return True

    def should_terminate(self, metrics: dict, elapsed_hours: float) -> tuple[bool, str]:
        """檢查是否應該終止（超時/超支/不可能達成）"""
        if elapsed_hours >= self.max_duration_hours:
            return True, "超過最大運行時間"

        if metrics.get("total_tokens", 0) >= self.max_budget_tokens:
            return True, "超過 Token 預算"

        # 檢查是否不可能達成（可加入更複雜的邏輯）
        return False, ""
```

### 2.2 Plan（執行計劃）

```python
@dataclass
class PlanStep:
    """計劃中的單一步驟"""
    id: str
    name: str
    type: str                          # "task" | "milestone" | "check"
    agent: str                         # 負責的 agent
    params: dict                       # 執行參數
    dependencies: list[str]            # 依賴的 step_id
    status: str = "pending"            # pending | in_progress | done | failed | skipped
    result: dict = None                # 執行結果
    error: str = None                  # 錯誤信息
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class Plan:
    """完整的執行計劃"""

    goal: Goal                         # 對應的目標
    steps: list[PlanStep]              # 執行步驟
    metadata: dict                     # 計劃元數據
    created_at: float = field(default_factory=time.time)
    started_at: float = None
    completed_at: float = None

    # 狀態追蹤
    current_step_index: int = 0
    status: str = "created"            # created | running | paused | completed | failed

    def get_ready_steps(self) -> list[PlanStep]:
        """獲取當前可執行的步驟（依賴已滿足）"""
        ready = []
        for step in self.steps:
            if step.status == "pending":
                # 檢查依賴
                deps_met = all(
                    any(s.id == dep and s.status == "done" for s in self.steps)
                    for dep in step.dependencies
                )
                if deps_met:
                    ready.append(step)
        return ready

    def get_progress(self) -> dict:
        """獲取執行進度"""
        total = len(self.steps)
        done = sum(1 for s in self.steps if s.status == "done")
        failed = sum(1 for s in self.steps if s.status == "failed")
        return {
            "total": total,
            "done": done,
            "failed": failed,
            "progress": done / total if total > 0 else 0,
        }

    def can_be_resumed(self) -> bool:
        """檢查是否可以從中斷點恢復"""
        return any(s.status == "pending" for s in self.steps)
```

### 2.3 SystemState（系統狀態）

```python
@dataclass
class SystemState:
    """系統運行狀態（用於持久化和恢復）"""

    # 基本信息
    session_id: str
    goal: Goal
    current_plan: Plan
    started_at: float
    last_checkpoint_at: float = None

    # 運行指標
    metrics: dict = field(default_factory=dict)
    llm_calls_total: int = 0
    tokens_used_total: int = 0

    # 健康狀態
    health_status: str = "healthy"     # healthy | degraded | critical
    last_health_check: float = None
    consecutive_failures: int = 0
    last_error: str = None

    # Loop 控制
    loop_count: int = 0
    is_paused: bool = False
    should_stop: bool = False
    stop_reason: str = None

    # 增量學習狀態
    learning_history: list[dict] = field(default_factory=list)
    successful_patterns: list[dict] = field(default_factory=list)

    def to_checkpoint(self) -> dict:
        """序列化為 checkpoint"""
        return asdict(self)

    @classmethod
    def from_checkpoint(cls, data: dict) -> "SystemState":
        """從 checkpoint 還原"""
        # 需要處理 Goal 和 Plan 的還原
        ...
```

---

## 三、核心組件設計

### 3.1 StateManager（狀態持久化）

```python
class StateManager:
    """狀態持久化管理器

    職責：
    1. 保存系統狀態到持久化存儲
    2. 從 checkpoint 恢復
    3. 定期自動保存
    4. 狀態版本管理（支持回滾）
    """

    def __init__(self, storage_path: Path, auto_save_interval: int = 60):
        self.storage_path = storage_path
        self.auto_save_interval = auto_save_interval
        self._last_save_time = 0

    def save_state(self, state: SystemState) -> str:
        """保存狀態，返回 checkpoint_id"""
        checkpoint_id = f"ckpt_{state.session_id}_{int(time.time())}"
        checkpoint_path = self.storage_path / f"{checkpoint_id}.json"

        checkpoint_data = {
            "id": checkpoint_id,
            "state": state.to_checkpoint(),
            "saved_at": time.time(),
        }

        checkpoint_path.write_text(json.dumps(checkpoint_data, ensure_ascii=False, indent=2))

        # 更新 latest 指針
        latest_path = self.storage_path / "latest.json"
        latest_path.write_text(json.dumps({"checkpoint_id": checkpoint_id}))

        return checkpoint_id

    def load_latest_state(self) -> tuple[SystemState, str]:
        """載入最新的狀態"""
        latest_path = self.storage_path / "latest.json"
        if not latest_path.exists():
            return None, None

        data = json.loads(latest_path.read_text())
        checkpoint_id = data["checkpoint_id"]
        checkpoint_path = self.storage_path / f"{checkpoint_id}.json"

        checkpoint_data = json.loads(checkpoint_path.read_text())
        state = SystemState.from_checkpoint(checkpoint_data["state"])
        return state, checkpoint_id

    def list_checkpoints(self, session_id: str = None) -> list[dict]:
        """列出所有 checkpoint"""
        ...

    def cleanup_old_checkpoints(self, keep_n: int = 10):
        """清理舊的 checkpoint，只保留最近 N 個"""
        ...
```

### 3.2 MainLoop（主控制器）

```python
class MainLoop:
    """主循環控制器

    職責：
    1. 驅動整個系統持續運行
    2. 協調各個 Agent
    3. 執行健康檢查
    4. 管理系統狀態
    """

    def __init__(self, goal: Goal, config: dict = None):
        self.goal = goal
        self.config = config or self._load_default_config()

        # 組件初始化
        self.state_manager = StateManager(...)
        self.health_checker = HealthChecker(...)
        self.plan_executor = PlanExecutor(...)
        self.self_judge = SelfJudge(...)
        self.message_bus = MessageBus()

        # 當前狀態
        self.state: SystemState = None

    def run(self, resume: bool = False):
        """啟動主循環

        Args:
            resume: 是否從之前的 checkpoint 恢復
        """
        # 1. 初始化或恢復狀態
        if resume:
            self.state, _ = self.state_manager.load_latest_state()
            if not self.state:
                raise RuntimeError("無可恢復的狀態")
            logger.info(f"從 checkpoint 恢復: {self.state.session_id}")
        else:
            self.state = self._initialize_state()

        # 2. 註冊中斷處理（優雅退出）
        self._setup_signal_handlers()

        # 3. 主循環
        try:
            while not self._should_stop():
                self._loop_iteration()

        except KeyboardInterrupt:
            logger.info("收到中斷信號，保存狀態後退出...")
            self._graceful_shutdown()
        except Exception as e:
            logger.error(f"循環異常: {e}")
            self.state.last_error = str(e)
            self.state.health_status = "critical"
            self.state_manager.save_state(self.state)
            raise

    def _loop_iteration(self):
        """單次循環迭代"""
        self.state.loop_count += 1
        iteration_start = time.time()

        # 1. 健康檢查
        health = self.health_checker.check(self.state)
        if health.status == "critical":
            logger.error(f"健康檢查失敗: {health.reason}")
            self.state.should_stop = True
            self.state.stop_reason = f"Health check failed: {health.reason}"
            return

        self.state.health_status = health.status
        self.state.last_health_check = time.time()

        # 2. 檢查是否達成目標
        if self.goal.is_achieved(self.state.metrics):
            logger.info("🎉 目標達成！")
            self.state.should_stop = True
            self.state.stop_reason = "Goal achieved"
            return

        # 3. 檢查終止條件
        elapsed_hours = (time.time() - self.state.started_at) / 3600
        should_terminate, reason = self.goal.should_terminate(self.state.metrics, elapsed_hours)
        if should_terminate:
            logger.info(f"終止條件觸發: {reason}")
            self.state.should_stop = True
            self.state.stop_reason = reason
            return

        # 4. 獲取下一個任務
        task = self.plan_executor.get_next_task(self.state)
        if not task:
            logger.info("沒有更多任務，等待 Plan 更新...")
            time.sleep(self.config.get("idle_wait", 30))
            return

        # 5. 執行任務
        try:
            result = self.plan_executor.execute_task(task, self.state)
            self.state.consecutive_failures = 0

            # 6. Self-Judge
            judgment = self.self_judge.evaluate(self.goal, self.state, result)

            if judgment.next_action == "adjust":
                # 動態調整 Plan
                self.plan_executor.adjust_plan(self.state, judgment.suggested_adjustment)
            elif judgment.next_action == "abort":
                self.state.should_stop = True
                self.state.stop_reason = f"Self-judge abort: {judgment.reason}"
                return

        except Exception as e:
            logger.error(f"任務執行失敗: {e}")
            self.state.consecutive_failures += 1
            self.state.last_error = str(e)

            # 連續失敗過多，暫停
            if self.state.consecutive_failures >= self.config.get("max_consecutive_failures", 5):
                logger.error("連續失敗過多，進入降級模式")
                self.state.health_status = "degraded"
                time.sleep(self.config.get("degraded_wait", 60))

        # 7. 保存 checkpoint
        if time.time() - self.state.last_checkpoint_at >= self.config.get("checkpoint_interval", 300):
            self.state_manager.save_state(self.state)
            self.state.last_checkpoint_at = time.time()

        # 8. 自適應等待
        elapsed_ms = int((time.time() - iteration_start) * 1000)
        wait_time = self._adaptive_wait(elapsed_ms)
        if wait_time > 0:
            time.sleep(wait_time)

    def _should_stop(self) -> bool:
        """判斷是否應該停止循環"""
        if self.state.should_stop:
            return True

        # 檢查外部停止信號（檔案）
        stop_file = self.storage_path / "STOP.signal"
        if stop_file.exists():
            logger.info("檢測到外部停止信號")
            return True

        return False

    def _adaptive_wait(self, last_iteration_ms: int) -> float:
        """根據系統狀態動態調整等待時間

        策略：
        - 運行順利 → 短等待，保持高吞吐
        - API 接近限速 → 長等待，避免觸發限制
        - 資源緊張 → 長等待，讓系統恢復
        """
        base_wait = self.config.get("base_wait", 1.0)

        # API 限速檢查
        api_guard = get_api_guard()
        usage = api_guard.get_usage_summary(last_n_minutes=1)
        if usage["total_requests"] >= 50:  # 接近 60/min 限制
            return base_wait * 5

        # 系統資源檢查（簡化版）
        if last_iteration_ms > 5000:  # 上次迭代超過 5 秒
            return base_wait * 2

        return base_wait
```

### 3.3 SelfJudge（自我監督）

```python
class SelfJudge:
    """自我監督機制

    職責：
    1. 評估當前狀態與目標的差距
    2. 判斷下一步行動（繼續/調整/中止）
    3. 生成調整建議
    """

    def __init__(self, llm_client, model: str = "glm-4-plus"):
        self.client = llm_client
        self.model = model

    def evaluate(self, goal: Goal, state: SystemState,
                 last_result: dict = None) -> Judgment:
        """評估當前狀態

        Returns:
            Judgment: {
                "achieved": bool,
                "confidence": float,
                "gap": str,
                "next_action": "continue" | "adjust" | "abort",
                "reason": str,
                "suggested_adjustment": dict,
            }
        """
        # 1. 快速檢查：是否已達成
        if goal.is_achieved(state.metrics):
            return Judgment(
                achieved=True,
                confidence=1.0,
                gap="",
                next_action="complete",
                reason="所有成功條件已滿足",
            )

        # 2. 構建評估 prompt
        prompt = self._build_judgment_prompt(goal, state, last_result)

        # 3. 調用 LLM 評估
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,  # 低溫度，更穩定的判斷
            response_format={"type": "json_object"},
        )

        # 4. 解析結果
        result = json.loads(response.choices[0].message.content)
        return Judgment(**result)

    def _build_judgment_prompt(self, goal: Goal, state: SystemState,
                               last_result: dict) -> str:
        """構建評估 prompt"""
        return f"""請評估當前系統狀態，判斷下一步行動。

## 目標
名稱：{goal.name}
描述：{goal.description}

成功條件：
{json.dumps(goal.success_criteria, ensure_ascii=False, indent=2)}

## 當前狀態
指標：
{json.dumps(state.metrics, ensure_ascii=False, indent=2)}

Plan 進度：
{json.dumps(state.current_plan.get_progress(), ensure_ascii=False, indent=2)}

當前步驟：
{state.current_plan.steps[state.current_plan.current_step_index].name if state.current_plan.current_step_index < len(state.current_plan.steps) else "N/A"}

最近執行結果：
{json.dumps(last_result or {}, ensure_ascii=False, indent=2)}

## 請判斷
1. 是否已達成目標？(achieved: true/false)
2. 當前信心分數 (confidence: 0-1)
3. 與目標的差距是什麼？(gap: string)
4. 下一步應該？(next_action: "continue" | "adjust" | "abort")
5. 理由 (reason: string)
6. 如果需要調整，具體建議？(suggested_adjustment: object)

請以 JSON 格式回覆。
"""
```

### 3.4 HealthChecker（健康檢查）

```python
class HealthChecker:
    """系統健康檢查

    檢查項：
    1. API 可用性
    2. 資源使用（內存、磁盤）
    3. 連續失敗次數
    4. 死鎖檢測（長時間無進度）
    """

    def __init__(self):
        self.checks = [
            self._check_api_availability,
            self._check_resources,
            self._check_stuck_detection,
            self._check_error_rate,
        ]

    def check(self, state: SystemState) -> HealthStatus:
        """執行所有健康檢查"""
        results = []
        for check_fn in self.checks:
            try:
                result = check_fn(state)
                results.append(result)
            except Exception as e:
                results.append(HealthCheck(
                    name=check_fn.__name__,
                    status="error",
                    reason=f"檢查失敗: {e}",
                ))

        # 綜合判斷
        failed = [r for r in results if r.status == "fail"]
        warning = [r for r in results if r.status == "warning"]

        if failed:
            return HealthStatus(
                status="critical",
                reason=failed[0].reason,
                details=results,
            )
        elif warning:
            return HealthStatus(
                status="degraded",
                reason=warning[0].reason,
                details=results,
            )
        else:
            return HealthStatus(
                status="healthy",
                reason="All checks passed",
                details=results,
            )

    def _check_api_availability(self, state: SystemState) -> HealthCheck:
        """檢查 API 可用性"""
        try:
            guard = get_api_guard()
            usage = guard.get_usage_summary(last_n_minutes=1)
            # 檢查最近的錯誤率
            # ...
            return HealthCheck(name="api", status="pass", reason="")
        except Exception as e:
            return HealthCheck(name="api", status="fail", reason=str(e))

    def _check_resources(self, state: SystemState) -> HealthCheck:
        """檢查系統資源"""
        import psutil

        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent

        if memory_percent > 90 or disk_percent > 90:
            return HealthCheck(
                name="resources",
                status="fail",
                reason=f"內存 {memory_percent}%, 磁盤 {disk_percent}%",
            )
        elif memory_percent > 70 or disk_percent > 80:
            return HealthCheck(
                name="resources",
                status="warning",
                reason=f"內存 {memory_percent}%, 磁盤 {disk_percent}%",
            )
        return HealthCheck(name="resources", status="pass", reason="")

    def _check_stuck_detection(self, state: SystemState) -> HealthCheck:
        """檢測是否卡住（長時間無進度）"""
        if state.last_checkpoint_at is None:
            return HealthCheck(name="stuck", status="pass", reason="")

        idle_time = time.time() - state.last_checkpoint_at
        if idle_time > 3600:  # 1 小時無進度
            return HealthCheck(
                name="stuck",
                status="fail",
                reason=f"{idle_time/60:.0f} 分鐘無進度",
            )
        return HealthCheck(name="stuck", status="pass", reason="")

    def _check_error_rate(self, state: SystemState) -> HealthCheck:
        """檢查錯誤率"""
        if state.consecutive_failures > 10:
            return HealthCheck(
                name="error_rate",
                status="fail",
                reason=f"連續失敗 {state.consecutive_failures} 次",
            )
        return HealthCheck(name="error_rate", status="pass", reason="")
```

---

## 四、配置與啟動

### 4.1 配置文件 (loop_config.yaml)

```yaml
# 主循環配置
loop:
  base_wait: 1.0              # 基礎等待時間（秒）
  idle_wait: 30               # 無任務時等待時間
  degraded_wait: 60           # 降級模式等待時間
  checkpoint_interval: 300    # Checkpoint 間隔（秒）

# 健康檢查
health:
  check_interval: 60          # 健康檢查間隔（秒）
  max_consecutive_failures: 5 # 連續失敗上限
  stuck_threshold: 3600       # 卡住檢測閾值（秒）

# 資源限制
resources:
  max_memory_percent: 85
  max_disk_percent: 90
  max_tokens_per_session: 10000000  # 1000萬 token 上限

# Plan 執行
execution:
  max_step_retries: 3
  step_timeout: 600           # 單步超時（秒）
  parallel_steps: 1           # 並行步驟數（線 B 可增加）

# 自我監督
judge:
  evaluation_interval: 5      # 每 N 步評估一次
  adjust_threshold: 0.6       # 低於此信心分數時觸發調整

# 存儲
storage:
  state_path: "./agent_framework/storage/state"
  checkpoint_keep: 10         # 保留最近 N 個 checkpoint
```

### 4.2 啟動腳本 (run_long_running.py)

```python
#!/usr/bin/env python3
"""
長程任務啟動腳本

用法：
    python run_long_running.py --goal "完善排灣語語料庫" --resume
"""

import argparse
from pathlib import Path
from agent_framework.core.loop import MainLoop
from agent_framework.core.plan import Goal
from agent_framework.config import load_config

def main():
    parser = argparse.ArgumentParser(description="啟動長程 Agent 任務")
    parser.add_argument("--goal", type=str, help="目標描述")
    parser.add_argument("--config", type=str, default="agent_framework/config/loop_config.yaml")
    parser.add_argument("--resume", action="store_true", help="從 checkpoint 恢復")
    parser.add_argument("--max-hours", type=float, default=24, help="最大運行時間")
    parser.add_argument("--max-tokens", type=int, default=10000000, help="最大 token 預算")

    args = parser.parse_args()

    # 載入配置
    config = load_config(args.config)

    # 定義目標
    if args.goal:
        goal = Goal(
            name=args.goal,
            description=args.goal,
            success_criteria={
                "corpus_coverage": {"min": 0.8, "current": 0.0},
                "translation_quality": {"min": 0.85, "current": 0.0},
            },
            termination_criteria={},
            max_duration_hours=args.max_hours,
            max_budget_tokens=args.max_tokens,
        )
    else:
        # 從檔案載入目標定義
        ...

    # 啟動主循環
    loop = MainLoop(goal=goal, config=config)
    loop.run(resume=args.resume)

if __name__ == "__main__":
    main()
```

---

## 五、實現優先順序

### Phase 1：最小可運行 Loop（MVP）- 1-2 天
1. ✅ `state.py` - StateManager（JSON 持久化）
2. ✅ `health.py` - HealthChecker（基礎檢查）
3. ✅ `loop.py` - MainLoop（基本框架）
4. ✅ `run_long_running.py` - 啟動腳本

**驗收標準**：能跑完整個循環，可保存/恢復狀態

### Phase 2：Plan-Driven - 2-3 天
1. ✅ `plan.py` - Plan, PlanStep, Goal 類
2. ✅ `judge.py` - SelfJudge 機制
3. ✅ `agents/orchestrator.py` - 基礎總管 Agent
4. ✅ Plan 生成 prompt

**驗收標準**：能根據目標生成 Plan 並逐步執行

### Phase 3：增強可靠性 - 2-3 天
1. ✅ `task_queue.py` - 任務排程
2. ✅ 優雅重啟機制
3. ✅ 完整的健康檢查
4. ✅ 錯誤恢復策略

**驗收標準**：能在異常後自動恢復，連續運行 8+ 小時

### Phase 4：優化與監控（可選）
1. ⏳ `monitoring/metrics.py` - 指標收集
2. ⏳ `monitoring/dashboard.py` - 可視化
3. ⏳ 動態 Plan 調整
4. ⏳ 成本優化

---

## 六、風險與緩解

| 風險 | 影響 | 緩解措施 |
|------|------|---------|
| API 限速/宕機 | 任務中斷 | 指數退避、多模型備份 |
| 記憶體溢出 | 程序崩潰 | 定期 checkpoint、監控 |
| 死鎖/無限循環 | 資源浪費 | Stuck detection、超時機制 |
| Token 超支 | 成本失控 | 預算熔斷、定期報告 |
| LLM 幻覺 | Plan 執行錯誤 | Self-Judge 驗證 |

---

**下一步**：請確認這份草案的方向是否符合您的需求，我將開始實現 Phase 1 的核心組件。
