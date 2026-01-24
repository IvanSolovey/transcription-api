# Feature Feasibility Report: Task Cancellation & User History

**Date:** 2024  
**System:** FastAPI Transcription API with SQLite + asyncio.Queue  
**Current Version:** Post Phase 1-3 fixes (100% queue test success)

---

## Executive Summary

### Feature 1: Stop Transcription Task ⚠️ PARTIALLY FEASIBLE
**Queued tasks**: ✅ Safe and straightforward  
**Processing tasks**: ⚠️ Limited - cannot interrupt CPU-bound work safely  
**Recommendation**: Implement with clear UX limitations

### Feature 2: User History 🟢 FULLY FEASIBLE
**Backend ready**: ✅ Database schema + repository method exist  
**Query performance**: ✅ Indexed api_key column  
**Implementation effort**: 🟢 Low (2-3 hours)  
**Recommendation**: Implement immediately

---

## Feature 1: Task Cancellation Analysis

### Current Architecture Constraints

**Queue Structure:**
```python
task_queue = asyncio.Queue(maxsize=25)  # In-memory, volatile
tasks = {}                              # Memory cache: task_id → TaskStatus
```

**Worker Pool:**
```python
ThreadPoolExecutor(max_workers=3)       # CPU-bound transcription work
await asyncio.wait_for(
    run_in_executor(...), 
    timeout=7200.0                      # 2-hour hard timeout
)
```

**Task Lifecycle:**
1. `queued` → Task created, added to asyncio.Queue
2. Worker pulls from queue → Status changes to `processing`
3. `run_in_executor()` starts sync CPU work in thread pool
4. Transcription completes → Status becomes `completed`/`failed`

### What CAN Be Done

#### ✅ Scenario A: Cancel Queued Tasks (SAFE)

**Feasibility:** 🟢 **FULLY FEASIBLE**

**How it works:**
```python
@app.delete("/task/{task_id}")
async def cancel_task(task_id: str, api_key: str):
    task_status = tasks.get(task_id)
    
    if task_status.status == "queued":
        # Task hasn't been picked up by worker yet
        task_status.status = "cancelled"
        task_status.completed_at = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Save to database
        with get_db_session() as session:
            repo = TaskRepository(session)
            repo.update_status(task_id, "cancelled")
        
        # Task will sit in asyncio.Queue until worker pulls it
        # Worker checks memory cache status and skips cancelled tasks
        return {"status": "cancelled"}
```

**Required Changes:**
1. **Worker loop modification** - Skip cancelled tasks:
```python
async def worker():
    while True:
        task_data = await task_queue.get()
        
        # ✅ NEW: Check if task was cancelled before starting
        if tasks[task_data['task_id']].status == "cancelled":
            logger.info(f"Пропускаємо скасовану задачу {task_data['task_id']}")
            task_queue.task_done()
            continue
        
        # Continue normal processing...
```

2. **DELETE endpoint enhancement** - Remove stub, add real logic
3. **Temp file cleanup** - Delete file if task cancelled before processing

**Risk Level:** 🟢 LOW  
**Complexity:** Easy  
**Estimated Time:** 1-2 hours

---

#### ⚠️ Scenario B: Cancel Processing Tasks (LIMITED)

**Feasibility:** 🟡 **PARTIALLY FEASIBLE**

**Problem:** Python's `ThreadPoolExecutor` does **NOT** support thread cancellation. The `run_in_executor()` call cannot be interrupted once started.

**Why it's hard:**
```python
# This is running in a separate thread:
def process_transcription_task_sync(task_id, file_path, ...):
    # 1. Load Whisper model (30-60 seconds)
    # 2. Transcribe audio (1-120 minutes)
    # 3. Diarization (if enabled, +30-60 seconds)
    # 4. Save results
    
    # 🔴 CANNOT be interrupted from outside the thread
    # 🔴 No safe way to kill a thread in Python
```

**Technical Limitations:**
- `ThreadPoolExecutor` has no `.cancel()` method for running tasks
- `faster-whisper` library has no callback mechanism for interruption
- Killing threads forcefully = undefined behavior (memory leaks, corrupted DB)
- Current 2-hour timeout is the only hard stop

**What COULD Work (with major refactoring):**

**Option 1: Cooperative Cancellation Flag (LOW IMPACT)**
```python
cancellation_flags = {}  # task_id → threading.Event()

def process_transcription_task_sync(task_id, file_path, ...):
    cancel_event = cancellation_flags.get(task_id)
    
    # Check before each major step
    if cancel_event and cancel_event.is_set():
        raise TaskCancelledException()
    
    transcription_service.transcribe(...)  # ⚠️ Still can't interrupt THIS
    
    if cancel_event and cancel_event.is_set():
        raise TaskCancelledException()
```

**Effectiveness:** ~20-30% (only interrupts between steps, not during CPU work)

**Option 2: ProcessPoolExecutor (MAJOR REFACTORING)**
```python
# Replace ThreadPoolExecutor with ProcessPoolExecutor
executor = ProcessPoolExecutor(max_workers=3)

# Can kill processes with:
future.cancel()  # Works if not started yet
process.terminate()  # Force kill running process
```

**Pros:**
- ✅ Can actually terminate running work
- ✅ Proper isolation (no shared memory corruption)

**Cons:**
- ⚠️ Must serialize all data (file_path, model_size, etc.)
- ⚠️ Model loading happens 3x (once per process) → 3x slower startup
- ⚠️ Higher memory usage (3 separate Python interpreters)
- ⚠️ Cannot share loaded Whisper models between processes
- ⚠️ Need IPC for progress updates
- 🔴 BREAKS current architecture (2-3 days refactoring)

**Option 3: Gradual Timeout Reduction (COMPROMISE)**
```python
@app.delete("/task/{task_id}")
async def cancel_task(task_id: str):
    if task_status.status == "processing":
        # Mark as "cancelling" - reduce timeout to 5 minutes
        task_status.status = "cancelling"
        # Worker will hit timeout sooner
        return {"status": "cancelling", "message": "Task will stop within 5 minutes"}
```

**Effectiveness:** 40-50% (eventual cancellation, not immediate)

---

### Recommended Implementation for Feature 1

**Phase 1: Quick Win (Implement Now)**
✅ Cancel queued tasks only (safe, easy)  
✅ Return 400 error for processing tasks: "Cannot cancel running transcription"  
✅ Add worker check to skip cancelled tasks  
✅ Add temp file cleanup for cancelled tasks  

**Phase 2: Future Enhancement (Post-MVP)**
⚠️ Add cooperative cancellation flags (20-30% effectiveness)  
⚠️ Document limitations clearly in API docs  
⚠️ Consider ProcessPoolExecutor migration (major effort)  

**Code Changes Required:**

```python
# 1. Worker enhancement (main.py line ~400)
async def worker():
    while True:
        task_data = await task_queue.get()
        task_id = task_data['task_id']
        
        # NEW: Skip cancelled tasks
        if tasks[task_id].status == "cancelled":
            logger.info(f"Пропускаємо скасовану задачу {task_id}")
            
            # Cleanup temp file
            file_path = task_data['file_path']
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.info(f"Видалено файл скасованої задачі: {file_path}")
            
            task_queue.task_done()
            continue
        
        # Normal processing...

# 2. Enhanced DELETE endpoint (main.py line ~755)
@app.delete("/task/{task_id}")
async def cancel_task(task_id: str, api_key: str = Depends(verify_api_key)):
    """Скасування задачі (тільки в статусі 'queued')"""
    
    # Load from DB if not in memory
    if task_id not in tasks:
        task_status = load_task_status(task_id)
        if not task_status:
            raise HTTPException(status_code=404, detail="Задача не знайдена")
        tasks[task_id] = task_status
    
    task_status = tasks[task_id]
    
    # Authorization check
    if task_status.api_key != api_key:
        raise HTTPException(status_code=403, detail="Доступ заборонено")
    
    # Status validation
    if task_status.status == "completed":
        raise HTTPException(status_code=400, detail="Задача вже завершена")
    
    if task_status.status == "failed":
        raise HTTPException(status_code=400, detail="Задача вже провалена")
    
    if task_status.status == "cancelled":
        raise HTTPException(status_code=400, detail="Задача вже скасована")
    
    # CRITICAL: Cannot cancel running transcription
    if task_status.status == "processing":
        raise HTTPException(
            status_code=400, 
            detail="Неможливо скасувати задачу, яка вже обробляється. "
                   "Транскрипція виконується в окремому потоці і не може бути перервана безпечно. "
                   "Задача завершиться автоматично (максимум 2 години)."
        )
    
    # Cancel queued task
    if task_status.status == "queued":
        task_status.status = "cancelled"
        task_status.completed_at = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Save to database
        with get_db_session() as session:
            repo = TaskRepository(session)
            repo.update_status(task_id, "cancelled", error_message="Скасовано користувачем")
        
        logger.info(f"Задача {task_id} скасована користувачем")
        
        return {
            "task_id": task_id,
            "status": "cancelled",
            "message": "Задача успішно скасована (ще не почала обробку)"
        }
```

**Database:** Already supports `cancelled` status (TaskStatus enum has it)  
**Testing:** Add to test_queue.py - submit 10 tasks, cancel 5 queued, verify only 5 complete

---

## Feature 2: User History Analysis

### Current Architecture Support

**Database Schema (app/db/models.py):**
```python
class Task(SQLModel, table=True):
    id: str = Field(primary_key=True)
    api_key: str = Field(foreign_key="apikey.key", index=True)  # ✅ INDEXED
    status: str
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    filename: str
    duration_sec: Optional[float]
    result_json: Optional[str]
    error_message: Optional[str]
```

**Existing Repository Method (app/db/repositories/tasks.py:48):**
```python
def get_by_api_key(self, api_key: str, limit: int = 100) -> List[Task]:
    """Get all tasks for specific API key."""
    statement = (
        select(Task)
        .where(Task.api_key == api_key)
        .order_by(Task.created_at.desc())
        .limit(limit)
    )
    return list(self.session.exec(statement).all())
```

**✅ THE METHOD ALREADY EXISTS!** Just need to expose via API endpoint.

### Feasibility Assessment

**Pros:**
- ✅ Database schema ready (indexed api_key)
- ✅ Repository method implemented
- ✅ Query is performant (index scan)
- ✅ Pagination support (limit parameter)
- ✅ No volatile state (pure database read)
- ✅ Thread-safe (read-only query)

**Cons:**
- ⚠️ No offset/cursor pagination (only limit)
- ⚠️ No date range filtering
- ⚠️ No status filtering within user's tasks

**Risk Level:** 🟢 ZERO  
**Complexity:** Trivial  
**Estimated Time:** 30 minutes - 1 hour

### Recommended Implementation

**New Endpoint:**
```python
@app.get("/my-tasks")
async def get_my_tasks(
    api_key: str = Depends(verify_api_key),
    limit: int = Query(default=50, le=200, description="Максимальна кількість задач"),
    status: Optional[str] = Query(default=None, description="Фільтр за статусом"),
    offset: int = Query(default=0, ge=0, description="Зміщення для пагінації")
):
    """
    Отримання історії транскрипцій поточного користувача.
    
    Повертає всі задачі, створені з поточним API ключем, 
    відсортовані за датою створення (нові спочатку).
    
    Args:
        limit: Кількість задач на сторінці (макс. 200)
        status: Фільтр за статусом (queued/processing/completed/failed/cancelled)
        offset: Зміщення для пагінації (default 0)
    
    Returns:
        {
            "tasks": [...],
            "total": int,
            "limit": int,
            "offset": int,
            "has_more": bool
        }
    """
    try:
        with get_db_session() as session:
            repo = TaskRepository(session)
            
            # Enhanced query with offset support
            from sqlmodel import select
            from app.db.models import Task
            
            statement = (
                select(Task)
                .where(Task.api_key == api_key)
                .order_by(Task.created_at.desc())
            )
            
            # Add status filter if specified
            if status:
                statement = statement.where(Task.status == status)
            
            # Apply pagination
            statement = statement.offset(offset).limit(limit + 1)
            
            db_tasks = list(session.exec(statement).all())
            
            # Check if more results exist
            has_more = len(db_tasks) > limit
            if has_more:
                db_tasks = db_tasks[:limit]
            
            # Convert to TaskStatus objects
            tasks_list = []
            for task in db_tasks:
                result = None
                if task.result_json:
                    try:
                        result = json.loads(task.result_json)
                    except:
                        pass
                
                tasks_list.append(TaskStatus(
                    task_id=task.id,
                    status=task.status,
                    created_at=task.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    started_at=task.started_at.strftime("%Y-%m-%d %H:%M:%S") if task.started_at else None,
                    completed_at=task.completed_at.strftime("%Y-%m-%d %H:%M:%S") if task.completed_at else None,
                    progress=100 if task.status == "completed" else 0,
                    result=result,
                    error=task.error_message,
                    file_name=task.filename,
                    language="uk",
                    model_size=task.model_size,
                    use_diarization=task.has_diarization,
                    api_key=task.api_key
                ))
            
            # Get total count for this user (expensive, can be optimized later)
            count_statement = (
                select(Task)
                .where(Task.api_key == api_key)
            )
            if status:
                count_statement = count_statement.where(Task.status == status)
            
            total_count = len(list(session.exec(count_statement).all()))
            
            return {
                "tasks": tasks_list,
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": has_more,
                "status_filter": status
            }
        
    except Exception as e:
        logger.error(f"Помилка отримання історії задач: {e}")
        raise HTTPException(status_code=500, detail=f"Помилка отримання історії: {str(e)}")
```

**Enhanced Repository Method (optional optimization):**
```python
def get_by_api_key_paginated(
    self, 
    api_key: str, 
    limit: int = 50, 
    offset: int = 0,
    status: Optional[str] = None
) -> tuple[List[Task], int]:
    """Get paginated tasks for API key with total count."""
    statement = (
        select(Task)
        .where(Task.api_key == api_key)
    )
    
    if status:
        statement = statement.where(Task.status == status)
    
    # Get total count
    count_statement = statement
    total = len(list(self.session.exec(count_statement).all()))
    
    # Get paginated results
    statement = statement.order_by(Task.created_at.desc()).offset(offset).limit(limit)
    tasks = list(self.session.exec(statement).all())
    
    return tasks, total
```

### Performance Considerations

**Query Performance:**
```sql
-- With index on api_key (already exists)
SELECT * FROM task 
WHERE api_key = 'abc123' 
ORDER BY created_at DESC 
LIMIT 50 OFFSET 0;

-- Expected execution time: < 10ms for 10,000 rows
-- Index scan → Sort → Limit (efficient)
```

**Scaling:**
- 1,000 tasks/user: Instant (< 5ms)
- 10,000 tasks/user: Fast (< 20ms)
- 100,000 tasks/user: Acceptable (< 100ms)
- 1M+ tasks/user: Consider archival strategy

**Optimization for COUNT (future):**
```python
# Instead of len(list(...)), use SQLAlchemy count:
from sqlalchemy import func

count_query = select(func.count()).select_from(Task).where(Task.api_key == api_key)
total = session.exec(count_query).one()
```

### Privacy & Security Considerations

**✅ Already Handled:**
- API key verified via `Depends(verify_api_key)`
- User can only see their own tasks (filtered by api_key)
- No cross-user data leakage possible

**⚠️ Additional Considerations:**
- Completed transcriptions contain sensitive audio data in `result_json`
- Should user be able to delete old tasks?
- GDPR compliance: Right to be forgotten (add DELETE /my-tasks/{id})

### Testing Strategy

**Test Cases:**
```python
# test_user_history.py

def test_empty_history():
    """New API key has no tasks"""
    response = client.get("/my-tasks", headers={"X-API-Key": new_key})
    assert response.json()["total"] == 0

def test_paginated_history():
    """Pagination works correctly"""
    # Create 25 tasks
    for i in range(25):
        submit_task(f"file_{i}.mp3")
    
    # Page 1
    page1 = client.get("/my-tasks?limit=10&offset=0", headers=auth)
    assert len(page1.json()["tasks"]) == 10
    assert page1.json()["has_more"] == True
    
    # Page 2
    page2 = client.get("/my-tasks?limit=10&offset=10", headers=auth)
    assert len(page2.json()["tasks"]) == 10
    
    # Page 3
    page3 = client.get("/my-tasks?limit=10&offset=20", headers=auth)
    assert len(page3.json()["tasks"]) == 5
    assert page3.json()["has_more"] == False

def test_status_filtering():
    """Can filter by status"""
    # Create mixed tasks
    submit_task("file1.mp3")  # Will be completed
    submit_task("file2.mp3")  # Will fail
    
    # Wait for completion
    time.sleep(5)
    
    # Filter completed
    completed = client.get("/my-tasks?status=completed", headers=auth)
    assert all(t["status"] == "completed" for t in completed.json()["tasks"])
    
    # Filter failed
    failed = client.get("/my-tasks?status=failed", headers=auth)
    assert all(t["status"] == "failed" for t in failed.json()["tasks"])

def test_chronological_order():
    """Tasks ordered by newest first"""
    # Create 5 tasks with delay
    task_ids = []
    for i in range(5):
        response = submit_task(f"file_{i}.mp3")
        task_ids.append(response.json()["task_id"])
        time.sleep(0.5)
    
    # Get history
    history = client.get("/my-tasks", headers=auth)
    returned_ids = [t["task_id"] for t in history.json()["tasks"]]
    
    # Should be in reverse order (newest first)
    assert returned_ids == list(reversed(task_ids))

def test_cross_user_isolation():
    """User A cannot see User B's tasks"""
    user_a_key = create_api_key("user_a")
    user_b_key = create_api_key("user_b")
    
    # User A creates task
    submit_task("user_a_file.mp3", api_key=user_a_key)
    
    # User B checks history
    response = client.get("/my-tasks", headers={"X-API-Key": user_b_key})
    assert response.json()["total"] == 0  # Cannot see User A's task
```

---

## Implementation Priority Recommendation

### 🟢 HIGH PRIORITY: User History (Feature 2)
**Why implement first:**
- ✅ Zero risk, pure read-only operation
- ✅ Already 90% implemented (just need endpoint)
- ✅ Instant user value (see past transcriptions)
- ✅ Foundation for future features (analytics, billing)
- ⏱️ 30-60 minutes implementation time

**Implementation Order:**
1. Add `/my-tasks` endpoint (20 min)
2. Enhance repository with pagination helper (15 min)
3. Add tests (30 min)
4. Update API documentation (15 min)

**Total Time:** 1-2 hours

---

### 🟡 MEDIUM PRIORITY: Task Cancellation - Phase 1 (Feature 1 - Limited)
**Why implement second:**
- ⚠️ Only works for queued tasks (not processing)
- ✅ Low risk, well-defined behavior
- ✅ Improves UX (mistakes happen)
- ⏱️ 2-3 hours implementation time

**Implementation Order:**
1. Enhance worker to skip cancelled tasks (30 min)
2. Implement DELETE endpoint with status checks (45 min)
3. Add temp file cleanup logic (20 min)
4. Add tests (45 min)
5. Update API docs with limitations (30 min)

**Total Time:** 2-3 hours

---

### ⚠️ LOW PRIORITY: Task Cancellation - Phase 2 (Processing Tasks)
**Why defer:**
- 🔴 Requires major architecture changes (ProcessPoolExecutor)
- 🔴 2-3 days refactoring effort
- 🔴 Testing complexity (race conditions, memory leaks)
- 🔴 Limited effectiveness (20-30% with cooperative flags)
- ⚠️ Breaking change risk

**Only consider if:**
- User complaints about long-running tasks are common
- Willing to invest 1 week for proper implementation
- Can accept process-based architecture (higher memory usage)

---

## Summary Table

| Feature | Feasibility | Risk | Effort | User Value | Priority |
|---------|-------------|------|--------|------------|----------|
| **User History** | 🟢 100% | 🟢 None | 🟢 1-2h | 🟢 High | **1st** |
| **Cancel Queued** | 🟢 100% | 🟢 Low | 🟡 2-3h | 🟡 Medium | **2nd** |
| **Cancel Processing** | 🟡 20-30% | 🔴 High | 🔴 3-5d | 🟡 Medium | **Defer** |

---

## Next Steps

**Immediate (This Sprint):**
1. ✅ Implement Feature 2: User History (`/my-tasks` endpoint)
2. ✅ Implement Feature 1 Phase 1: Cancel queued tasks only
3. ✅ Add comprehensive tests for both features
4. ✅ Update API documentation with clear limitations

**Future Considerations:**
- Add task deletion endpoint (`DELETE /my-tasks/{id}` for GDPR)
- Add date range filtering to history
- Add statistics endpoint (`GET /my-stats`)
- Research ProcessPoolExecutor migration for true cancellation

**Documentation Required:**
- API endpoint specs (OpenAPI/Swagger)
- User-facing limitations ("Cannot cancel running tasks")
- Error handling guide
- Testing guide

---

## Questions for Product Decision

1. **User History Privacy:** Should completed tasks auto-delete after 30/60/90 days?
2. **Cancellation UX:** Is "cannot cancel processing" acceptable, or must we invest in ProcessPoolExecutor?
3. **History Pagination:** Is offset-based pagination sufficient, or need cursor-based?
4. **Statistics:** Should `/my-tasks` include summary stats (total completed, avg duration)?
5. **Billing Integration:** Will history be used for usage-based billing calculations?

---

**Report Prepared By:** GitHub Copilot  
**Review Status:** Ready for stakeholder review  
**Implementation Ready:** Feature 2 (100%), Feature 1 Phase 1 (90%)
