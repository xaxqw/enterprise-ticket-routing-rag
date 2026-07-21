"""
[已废弃] 早期任务定义位置。
真正的 Celery 任务已统一迁移到 app/services/tasks.py（被 celery_app.autodiscover 发现）。
此处仅做向后兼容的再导出，避免旧代码 `from app.core.tasks import process_document_task` 失效。
"""
from app.services.tasks import (  # noqa: F401
    process_document_task,
    process_url_task,
    rebuild_tenant_index_task,
)
