"""
Tenant Context Management for Multi-Tenant Data Isolation

This module provides a robust, automatic way to enforce tenant (admin) data isolation
across all database operations. It uses Python's contextvars for async-safe tenant
context propagation.

ARCHITECTURE:
- TenantContext: Holds the current tenant (admin_id) for the request
- TENANT_SCOPED_COLLECTIONS: Collections that require tenant filtering
- TenantAwareDB: Wrapper that automatically injects tenant filters

USAGE:
1. Set tenant context at the start of each request (middleware)
2. Use TenantAwareDB methods instead of raw DatabaseManager methods
3. All queries to tenant-scoped collections automatically include admin_id filter

This prevents the "forgot to add admin_id" class of bugs entirely.
"""

import logging
from contextvars import ContextVar
from typing import Optional, Dict, Any, List, Set
from bson import ObjectId
from functools import wraps

logger = logging.getLogger(__name__)

# Async-safe context variable for tenant
_tenant_context: ContextVar[Optional[Dict[str, Any]]] = ContextVar('tenant_context', default=None)


# Collections that MUST be filtered by admin_id (tenant-scoped data)
TENANT_SCOPED_COLLECTIONS: Set[str] = {
    "students",
    "documents",
    "tutors",
    "questions",
    "question_attempts",
    "student_activity_log",
    "chat_sessions",
    "student_test_attempts",
    "assignments",
    "meetings",
    "notifications",
    "class_schedules",
    "smartboard_sessions",  # SmartBoard whiteboard sessions
    "class_timetables",     # Class timetable definitions
    "timetable_periods",    # Individual period entries
    "academic_calendar",    # Holidays, exams, events
    "pre_reads",            # Pre-read material for periods
}

# Collections that are global (not tenant-scoped)
GLOBAL_COLLECTIONS: Set[str] = {
    "admins",           # Admin accounts themselves
    "system_settings",  # Global system config
    "audit_logs",       # System-wide audit trail
}

# Collections where tenant filter is optional (mixed use)
OPTIONAL_TENANT_COLLECTIONS: Set[str] = {
    "mcq_questions",    # Can be global or tenant-specific
}


class TenantContext:
    """
    Manages tenant context for the current request/async task.

    Usage:
        # Set at request start (in middleware or dependency)
        TenantContext.set(admin_id="abc123", user_type="admin")

        # Get current tenant
        ctx = TenantContext.get()
        admin_id = ctx.admin_id

        # Clear at request end
        TenantContext.clear()
    """

    @staticmethod
    def set(
        admin_id: Optional[str],
        user_type: str = "admin",
        user_id: Optional[str] = None,
        tutor_id: Optional[str] = None,
        db_name: Optional[str] = None,
        tenant_id: Optional[str] = None,
        institution_id: Optional[str] = None,
        **extra,
    ):
        """Set tenant context for current async context"""
        ctx = {
            "admin_id": admin_id,
            "user_type": user_type,
            "user_id": user_id,
            "tutor_id": tutor_id,
            "db_name": db_name,
            "tenant_id": tenant_id,
            "institution_id": institution_id,
            **extra
        }
        _tenant_context.set(ctx)
        logger.debug(
            "Tenant context set: admin_id=%s user_type=%s db_name=%s",
            admin_id,
            user_type,
            db_name,
        )

    @staticmethod
    def get() -> Optional[Dict[str, Any]]:
        """Get current tenant context"""
        return _tenant_context.get()

    @staticmethod
    def get_admin_id() -> Optional[str]:
        """Get current admin_id from context"""
        ctx = _tenant_context.get()
        return ctx.get("admin_id") if ctx else None

    @staticmethod
    def get_admin_oid() -> Optional[ObjectId]:
        """Get current admin_id as ObjectId"""
        admin_id = TenantContext.get_admin_id()
        if admin_id:
            try:
                return ObjectId(admin_id)
            except Exception:
                return None
        return None

    @staticmethod
    def get_db_name() -> Optional[str]:
        """Get current tenant database name from context"""
        ctx = _tenant_context.get()
        return ctx.get("db_name") if ctx else None

    @staticmethod
    def clear():
        """Clear tenant context"""
        _tenant_context.set(None)

    @staticmethod
    def require() -> Dict[str, Any]:
        """Get tenant context, raise if not set"""
        ctx = _tenant_context.get()
        if not ctx or not ctx.get("admin_id"):
            raise TenantContextError("Tenant context not set - cannot perform tenant-scoped operation")
        return ctx


class TenantContextError(Exception):
    """Raised when tenant context is required but not set"""
    pass


class TenantIsolationError(Exception):
    """Raised when a tenant isolation violation is detected"""
    pass


class TenantAwareDB:
    """
    Tenant-aware database wrapper that automatically enforces data isolation.

    This class wraps DatabaseManager and automatically injects admin_id filters
    for tenant-scoped collections. It's impossible to accidentally query
    another tenant's data when using this wrapper.

    Usage:
        db = DatabaseManager()
        tenant_db = TenantAwareDB(db)

        # This automatically adds {"admin_id": current_admin_id} to the filter
        students = await tenant_db.find("students", {"grade": "10"})

        # For global collections, no filter is added
        admins = await tenant_db.find("admins", {"email": "foo@bar.com"})
    """

    def __init__(self, db_manager):
        self.db = db_manager
        self._bypass_tenant_filter = False

    def _get_tenant_filter(self, collection_name: str) -> Dict[str, Any]:
        """Get the tenant filter to apply for this collection"""
        if self._bypass_tenant_filter:
            return {}

        if collection_name in GLOBAL_COLLECTIONS:
            return {}

        if collection_name in TENANT_SCOPED_COLLECTIONS:
            admin_oid = TenantContext.get_admin_oid()
            if admin_oid is None:
                raise TenantContextError(
                    f"Cannot query tenant-scoped collection '{collection_name}' without tenant context. "
                    f"Ensure TenantContext.set() was called at request start."
                )
            return {"admin_id": admin_oid}

        # Optional tenant collections - add filter if context exists
        if collection_name in OPTIONAL_TENANT_COLLECTIONS:
            admin_oid = TenantContext.get_admin_oid()
            if admin_oid:
                return {"admin_id": admin_oid}
            return {}

        # Unknown collection - log warning and proceed without filter
        logger.warning(f"Unknown collection '{collection_name}' - no tenant filter applied. "
                      f"Consider adding it to TENANT_SCOPED_COLLECTIONS or GLOBAL_COLLECTIONS.")
        return {}

    def _merge_filter(self, collection_name: str, user_filter: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user filter with tenant filter"""
        tenant_filter = self._get_tenant_filter(collection_name)

        if not tenant_filter:
            return user_filter

        # Check if user already included admin_id (for backward compatibility)
        if "admin_id" in user_filter:
            # Verify it matches the tenant context (security check)
            user_admin_id = user_filter["admin_id"]
            tenant_admin_id = tenant_filter["admin_id"]

            # Normalize to string for comparison
            user_str = str(user_admin_id)
            tenant_str = str(tenant_admin_id)

            if user_str != tenant_str:
                raise TenantIsolationError(
                    f"Tenant isolation violation: filter admin_id ({user_str}) "
                    f"does not match tenant context ({tenant_str})"
                )
            return user_filter

        # Merge filters
        if "$and" in user_filter:
            # Already has $and, append tenant filter
            user_filter["$and"].append(tenant_filter)
            return user_filter
        else:
            # Simple merge
            return {**user_filter, **tenant_filter}

    async def find_one(self, collection_name: str, filter_dict: Dict[str, Any],
                       projection: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Find one document with automatic tenant filtering"""
        merged_filter = self._merge_filter(collection_name, filter_dict)
        return await self.db.mongo_find_one(collection_name, merged_filter, projection)

    async def find(self, collection_name: str, filter_dict: Dict[str, Any],
                   projection: Optional[Dict[str, Any]] = None,
                   sort: Optional[List[tuple]] = None,
                   limit: Optional[int] = None,
                   skip: Optional[int] = None) -> List[Dict[str, Any]]:
        """Find documents with automatic tenant filtering"""
        merged_filter = self._merge_filter(collection_name, filter_dict)
        return await self.db.mongo_find(collection_name, merged_filter, projection, sort, limit, skip)

    async def insert_one(self, collection_name: str, document: Dict[str, Any]) -> Optional[str]:
        """Insert document with automatic tenant ID injection"""
        if collection_name in TENANT_SCOPED_COLLECTIONS:
            admin_oid = TenantContext.get_admin_oid()
            if admin_oid is None:
                raise TenantContextError(
                    f"Cannot insert into tenant-scoped collection '{collection_name}' without tenant context"
                )
            # Inject admin_id if not present
            if "admin_id" not in document:
                document["admin_id"] = admin_oid
            elif str(document["admin_id"]) != str(admin_oid):
                raise TenantIsolationError(
                    f"Tenant isolation violation: document admin_id does not match tenant context"
                )

        return await self.db.mongo_insert_one(collection_name, document)

    async def insert_many(self, collection_name: str, documents: List[Dict[str, Any]]) -> Optional[List[str]]:
        """Insert documents with automatic tenant ID injection"""
        if collection_name in TENANT_SCOPED_COLLECTIONS:
            admin_oid = TenantContext.get_admin_oid()
            if admin_oid is None:
                raise TenantContextError(
                    f"Cannot insert into tenant-scoped collection '{collection_name}' without tenant context"
                )
            for doc in documents:
                if "admin_id" not in doc:
                    doc["admin_id"] = admin_oid
                elif str(doc["admin_id"]) != str(admin_oid):
                    raise TenantIsolationError(
                        f"Tenant isolation violation: document admin_id does not match tenant context"
                    )

        return await self.db.mongo_insert_many(collection_name, documents)

    async def update_one(self, collection_name: str, filter_dict: Dict[str, Any],
                         update_dict: Dict[str, Any], upsert: bool = False) -> bool:
        """Update document with automatic tenant filtering"""
        merged_filter = self._merge_filter(collection_name, filter_dict)
        return await self.db.mongo_update_one(collection_name, merged_filter, update_dict, upsert)

    async def update_many(self, collection_name: str, filter_dict: Dict[str, Any],
                          update_dict: Dict[str, Any]) -> int:
        """Update documents with automatic tenant filtering"""
        merged_filter = self._merge_filter(collection_name, filter_dict)
        return await self.db.mongo_update_many(collection_name, merged_filter, update_dict)

    async def delete_one(self, collection_name: str, filter_dict: Dict[str, Any]) -> bool:
        """Delete document with automatic tenant filtering"""
        merged_filter = self._merge_filter(collection_name, filter_dict)
        return await self.db.mongo_delete_one(collection_name, merged_filter)

    async def delete_many(self, collection_name: str, filter_dict: Dict[str, Any]) -> int:
        """Delete documents with automatic tenant filtering"""
        merged_filter = self._merge_filter(collection_name, filter_dict)
        return await self.db.mongo_delete_many(collection_name, merged_filter)

    async def count(self, collection_name: str, filter_dict: Dict[str, Any] = None) -> int:
        """Count documents with automatic tenant filtering"""
        filter_dict = filter_dict or {}
        merged_filter = self._merge_filter(collection_name, filter_dict)
        return await self.db.mongo_count(collection_name, merged_filter)

    async def aggregate(self, collection_name: str, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run aggregation pipeline with automatic tenant filtering.
        Injects $match stage at the beginning if collection is tenant-scoped.
        """
        if collection_name in TENANT_SCOPED_COLLECTIONS:
            tenant_filter = self._get_tenant_filter(collection_name)
            if tenant_filter:
                # Inject tenant filter as first $match stage
                pipeline = [{"$match": tenant_filter}] + pipeline

        return await self.db.mongo_aggregate(collection_name, pipeline)

    def bypass_tenant_filter(self):
        """
        Context manager to temporarily bypass tenant filtering.
        USE WITH EXTREME CAUTION - only for admin/system operations.

        Usage:
            with tenant_db.bypass_tenant_filter():
                # This query won't have tenant filter
                all_students = await tenant_db.find("students", {})
        """
        return _TenantBypassContext(self)


class _TenantBypassContext:
    """Context manager for bypassing tenant filter"""

    def __init__(self, tenant_db: TenantAwareDB):
        self.tenant_db = tenant_db
        self._previous_state = None

    def __enter__(self):
        self._previous_state = self.tenant_db._bypass_tenant_filter
        self.tenant_db._bypass_tenant_filter = True
        logger.warning("SECURITY: Tenant filter bypass enabled - ensure this is intentional")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tenant_db._bypass_tenant_filter = self._previous_state
        return False


def require_tenant_context(func):
    """
    Decorator to require tenant context for a function.
    Raises TenantContextError if context is not set.

    Usage:
        @require_tenant_context
        async def get_students():
            # TenantContext is guaranteed to be set here
            ...
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        TenantContext.require()
        return await func(*args, **kwargs)
    return wrapper
