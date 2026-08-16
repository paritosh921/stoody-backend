"""
Async Database Manager for SkillBot
Handles MongoDB connections with connection pooling
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List, Tuple
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ServerSelectionTimeoutError, OperationFailure, ConfigurationError
import certifi

from config_async import (
    MONGODB_URL,
    MONGODB_DB_MASTER,
    MONGODB_DB_STOODY,
    DISABLE_MONGODB,
    settings
)

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages MongoDB connections with async support"""

    def __init__(self):
        self.mongo_client: Optional[AsyncIOMotorClient] = None
        self.mongo_master_db: Optional[AsyncIOMotorDatabase] = None
        self.mongo_tenant_dbs: Dict[str, AsyncIOMotorDatabase] = {}
        self.mongo_db_b2c: Optional[AsyncIOMotorDatabase] = None  # B2C database
        self._mongo_lock = asyncio.Lock()
        self._indexed_dbs: set = set()

    async def initialize(self) -> bool:
        """Initialize database connections"""
        try:
            # Initialize MongoDB (if configured and not disabled)
            if MONGODB_URL and not DISABLE_MONGODB:
                await self._init_mongodb()

            logger.info("✅ Database manager initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize database manager: {str(e)}")
            raise

    async def _init_mongodb(self) -> None:
        """Initialize MongoDB with connection pooling"""
        async with self._mongo_lock:
            try:
                # Skip MongoDB entirely if disabled or not configured
                if not MONGODB_URL or DISABLE_MONGODB:
                    logger.info("MongoDB disabled or not configured; skipping initialization")
                    self.mongo_client = None
                    self.mongo_master_db = None
                    self.mongo_db_b2c = None
                    self.mongo_tenant_dbs.clear()
                    return
                # Create MongoDB client with async motor
                # Reduced timeouts to fail fast and not hang the UI
                min_pool_size = max(0, settings.MONGODB_MIN_POOL_SIZE)
                max_pool_size = max(1, settings.MONGODB_MAX_POOL_SIZE)

                if min_pool_size > max_pool_size:
                    logger.warning(
                        "Configured MongoDB min pool size (%d) exceeds max pool size (%d); trimming min to match max",
                        min_pool_size,
                        max_pool_size
                    )
                    min_pool_size = max_pool_size

                self.mongo_client = AsyncIOMotorClient(
                    MONGODB_URL,
                    minPoolSize=min_pool_size,
                    maxPoolSize=max_pool_size,
                    maxIdleTimeMS=30000,
                    serverSelectionTimeoutMS=5000,  # Reduced from 15s to 5s
                    connectTimeoutMS=5000,  # Reduced from 15s to 5s
                    socketTimeoutMS=10000,  # Reduced from 45s to 10s
                    waitQueueTimeoutMS=5000,  # Reduced from 20s to 5s
                    tls=True,
                    tlsCAFile=certifi.where(),
                )

                # Get database instances (master + B2C only — tenant DBs resolved per-request)
                self.mongo_master_db = self.mongo_client[MONGODB_DB_MASTER]

                # Test connection
                await self.mongo_client.admin.command('ping')

                logger.info(f"✅ MongoDB initialized - Master DB: {MONGODB_DB_MASTER}")

                # Also initialize B2C database using same client
                self.mongo_db_b2c = self.mongo_client[MONGODB_DB_STOODY]
                logger.info(f"✅ B2C MongoDB initialized - Database: {MONGODB_DB_STOODY}")

                # Ensure indexes on B2C database
                try:
                    await self.ensure_indexes_for_db(self.mongo_db_b2c)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to ensure B2C indexes: {str(e)}")

            except (ServerSelectionTimeoutError, ConfigurationError) as e:
                logger.warning(f"⚠️ MongoDB connection failed ({type(e).__name__}) - continuing without MongoDB")
                self.mongo_client = None
                self.mongo_master_db = None
                self.mongo_db_b2c = None
                self.mongo_tenant_dbs.clear()
            except Exception as e:
                logger.error(f"❌ MongoDB initialization failed: {str(e)}")
                raise

    async def get_mongo_db(self) -> None:
        """REMOVED: This method no longer returns a database.

        All callers must use get_tenant_db(db_name) or get_master_db() instead.
        Raises RuntimeError to surface any remaining callers.
        """
        raise RuntimeError(
            "get_mongo_db() is removed. Use get_tenant_db(db_name) for tenant data "
            "or get_master_db() for the tenant registry."
        )

    async def get_master_db(self) -> Optional[AsyncIOMotorDatabase]:
        """Get master MongoDB database (tenant registry) with lazy initialization"""
        if self.mongo_master_db is None and MONGODB_URL and not DISABLE_MONGODB:
            await self._init_mongodb()
        return self.mongo_master_db

    async def get_tenant_db(self, db_name: str) -> Optional[AsyncIOMotorDatabase]:
        """Get tenant MongoDB database by name (cached per process)."""
        if not db_name:
            return None
        if self.mongo_client is None and MONGODB_URL and not DISABLE_MONGODB:
            await self._init_mongodb()
        if self.mongo_client is None:
            return None
        if db_name in self.mongo_tenant_dbs:
            return self.mongo_tenant_dbs[db_name]
        tenant_db = self.mongo_client[db_name]
        self.mongo_tenant_dbs[db_name] = tenant_db
        # Lazily ensure indexes on first access per database
        try:
            await self.ensure_indexes_for_db(tenant_db)
        except Exception as e:
            logger.warning(f"Failed to ensure indexes on {db_name}: {str(e)}")
        return tenant_db

    async def get_mongo_collection(self, collection_name: str):
        """REMOVED: This method no longer returns a collection.

        All callers must use get_tenant_db(db_name)[collection_name] instead.
        Raises RuntimeError to surface any remaining callers.
        """
        raise RuntimeError(
            f"get_mongo_collection('{collection_name}') is removed. "
            "Use get_tenant_db(db_name) to get the tenant database first."
        )

    async def _get_context_db(self) -> Optional[AsyncIOMotorDatabase]:
        """Resolve MongoDB database for the current request context."""
        if self.mongo_client is None and MONGODB_URL and not DISABLE_MONGODB:
            await self._init_mongodb()
        if self.mongo_client is None:
            return None

        db_name = None
        try:
            from core.tenant import TenantContext
            db_name = TenantContext.get_db_name()
        except Exception:
            db_name = None

        if db_name:
            return await self.get_tenant_db(db_name)

        logger.warning("No tenant context set — returning None (strict mode, no default DB)", stack_info=True)
        return None

    async def _get_context_collection(self, collection_name: str):
        """Get MongoDB collection for the current request context."""
        db = await self._get_context_db()
        if db is None:
            return None
        return db[collection_name]

    async def get_context_db(self) -> Optional[AsyncIOMotorDatabase]:
        """Public accessor for the context-resolved MongoDB database."""
        return await self._get_context_db()

    async def get_master_collection(self, collection_name: str):
        """Get MongoDB collection from master database (skb_master)"""
        db = await self.get_master_db()
        if db is None:
            return None
        return db[collection_name]

    async def get_tenant_collection(self, db_name: str, collection_name: str):
        """Get MongoDB collection from tenant database by name."""
        db = await self.get_tenant_db(db_name)
        if db is None:
            return None
        return db[collection_name]

    async def get_b2c_db(self) -> Optional[AsyncIOMotorDatabase]:
        """Get B2C MongoDB database (stoody-b2c) with lazy initialization"""
        if self.mongo_db_b2c is None and MONGODB_URL and not DISABLE_MONGODB:
            await self._init_mongodb()
        return self.mongo_db_b2c

    async def get_b2c_collection(self, collection_name: str):
        """Get MongoDB collection from B2C database (stoody-b2c)"""
        db = await self.get_b2c_db()
        if db is None:
            return None
        return db[collection_name]

    @staticmethod
    def _normalize_index_keys(keys: List[Tuple[str, int]]) -> Tuple[Tuple[str, int], ...]:
        """Normalize index keys to a stable tuple representation for comparisons."""
        return tuple((str(field), int(direction)) for field, direction in keys)

    async def _ensure_index_with_spec_check(
        self,
        collection,
        keys: List[Tuple[str, int]],
        *,
        name: str,
        unique: bool = False,
        sparse: bool = False,
        partial_filter_expression: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create index only when an equivalent spec does not already exist under another name."""
        target_keys = self._normalize_index_keys(keys)
        index_info = await collection.index_information()

        for existing_name, existing_def in index_info.items():
            existing_keys = self._normalize_index_keys(existing_def.get("key", []))
            if existing_keys != target_keys:
                continue

            existing_unique = bool(existing_def.get("unique", False))
            existing_sparse = bool(existing_def.get("sparse", False))
            existing_partial = existing_def.get("partialFilterExpression")
            if (
                existing_unique == unique
                and existing_sparse == sparse
                and existing_partial == partial_filter_expression
            ):
                if existing_name != name:
                    logger.info(
                        "Equivalent index already exists on %s as '%s'; skipping '%s'",
                        collection.full_name,
                        existing_name,
                        name,
                    )
                return

        create_kwargs: Dict[str, Any] = {"name": name}
        if unique:
            create_kwargs["unique"] = True
        if sparse:
            create_kwargs["sparse"] = True
        if partial_filter_expression is not None:
            create_kwargs["partialFilterExpression"] = partial_filter_expression
        await collection.create_index(keys, **create_kwargs)

    async def _drop_matching_indexes(
        self,
        collection,
        *,
        key_specs: List[List[Tuple[str, int]]],
        names: Optional[List[str]] = None,
        unique: Optional[bool] = None,
    ) -> int:
        """Drop indexes matching any provided name or normalized key spec."""
        index_info = await collection.index_information()
        target_specs = {self._normalize_index_keys(spec) for spec in key_specs}
        dropped = 0

        for existing_name, existing_def in index_info.items():
            if existing_name == "_id_":
                continue

            existing_keys = self._normalize_index_keys(existing_def.get("key", []))
            name_match = bool(names and existing_name in names)
            spec_match = existing_keys in target_specs if target_specs else False
            unique_match = unique is None or bool(existing_def.get("unique", False)) == unique

            if (name_match or spec_match) and unique_match:
                try:
                    await collection.drop_index(existing_name)
                    dropped += 1
                    logger.info("Dropped legacy index %s on %s", existing_name, collection.full_name)
                except OperationFailure as exc:
                    if getattr(exc, "code", None) == 27 or "IndexNotFound" in str(exc):
                        logger.debug(
                            "Legacy index %s already absent on %s",
                            existing_name,
                            collection.full_name,
                        )
                    else:
                        logger.warning(
                            "Failed to drop legacy index %s on %s: %s",
                            existing_name,
                            collection.full_name,
                            exc,
                        )
                except Exception as exc:
                    logger.warning(
                        "Failed to drop legacy index %s on %s: %s",
                        existing_name,
                        collection.full_name,
                        exc,
                    )
        return dropped

    async def ensure_indexes_for_db(self, db: AsyncIOMotorDatabase) -> None:
        """Create necessary indexes on the given database (idempotent)."""
        db_name = db.name
        if db_name in self._indexed_dbs:
            return
        try:
            students = db["students"]
            await self._ensure_index_with_spec_check(
                students,
                [("student_id", 1)],
                unique=True,
                name="uniq_student_id"
            )
            await self._ensure_index_with_spec_check(
                students,
                [("username_lower", 1)],
                unique=True,
                sparse=True,
                name="uniq_students_username_lower"
            )

            tutors = db["tutors"]
            await self._ensure_index_with_spec_check(
                tutors,
                [("username", 1)],
                unique=True,
                name="uniq_tutors_username"
            )
            await self._ensure_index_with_spec_check(
                tutors,
                [("username_lower", 1)],
                unique=True,
                sparse=True,
                name="uniq_tutors_username_lower"
            )
            await self._ensure_index_with_spec_check(
                tutors,
                [("tutor_id", 1)],
                unique=True,
                name="uniq_tutors_tutor_id"
            )

            # Student contribution credits collections (append-only ledger + async jobs).
            # MongoDB already creates a unique ``_id_`` index.  A second named
            # unique index on ``_id`` is invalid on Atlas and made every
            # operational script report a misleading startup warning.
            student_credit_jobs = db["student_credit_jobs"]
            await self._ensure_index_with_spec_check(
                student_credit_jobs,
                [("source_type", 1), ("source_id", 1), ("source_version", 1)],
                unique=True,
                name="uniq_credit_source_version",
            )
            await self._ensure_index_with_spec_check(
                student_credit_jobs,
                [("status", 1), ("next_attempt_at", 1), ("lease_expires_at", 1)],
                name="idx_credit_job_dispatch",
            )
            await self._ensure_index_with_spec_check(
                student_credit_jobs,
                [("status", 1), ("updated_at", -1)],
                name="idx_credit_job_status_updated",
            )

            student_credit_judgments = db["student_credit_judgments"]
            await self._ensure_index_with_spec_check(
                student_credit_judgments,
                [("source_type", 1), ("source_id", 1), ("source_version", 1)],
                unique=True,
                name="uniq_credit_judgment_source",
            )
            await self._ensure_index_with_spec_check(
                student_credit_judgments,
                [("student_record_id", 1), ("decided_at", -1)],
                name="idx_credit_judgment_student",
            )

            student_credit_ledger = db["student_credit_ledger"]
            await self._ensure_index_with_spec_check(
                student_credit_ledger,
                [("judgment_key", 1)],
                unique=True,
                name="uniq_credit_ledger_judgment",
            )
            await self._ensure_index_with_spec_check(
                student_credit_ledger,
                [("student_record_id", 1), ("created_at", -1)],
                name="idx_credit_ledger_student",
            )
            await self._ensure_index_with_spec_check(
                student_credit_ledger,
                [("group_key", 1), ("created_at", 1)],
                name="idx_credit_ledger_group",
            )

            student_credit_locks = db["student_credit_locks"]
            await self._ensure_index_with_spec_check(
                student_credit_locks,
                [("lease_expires_at", 1)],
                name="idx_credit_lock_expiry",
            )

            # Note classification lookup indexes. The legacy unique page index
            # omitted ``copy_id`` and is incompatible with multiple copies of
            # the same notebook page. Do not recreate it here: the copy-aware
            # unique index is migrated below once every row has ``copy_id``.
            note_cls = db["note_classifications"]
            await self._ensure_index_with_spec_check(
                note_cls,
                [("user_id", 1), ("subject", 1)],
                name="idx_note_user_subject"
            )
            await self._ensure_index_with_spec_check(
                note_cls,
                [("user_id", 1), ("subject", 1), ("topic", 1)],
                name="idx_note_user_subject_topic"
            )

            note_fc = db["note_flashcards"]
            await self._ensure_index_with_spec_check(
                note_fc,
                [("user_id", 1), ("subject", 1), ("topic", 1)],
                unique=True,
                name="uniq_flashcard_topic"
            )

            note_ps = db["note_practice_sets"]
            await self._ensure_index_with_spec_check(
                note_ps,
                [("user_id", 1), ("subject", 1), ("topic", 1)],
                unique=True,
                name="uniq_practice_topic"
            )

            # Canvas pages: merge-then-deduplicate before unique index
            canvas_pages = db["canvas_pages"]
            try:
                pipeline = [
                    {"$group": {
                        "_id": {
                            "user_id": "$user_id",
                            "copy_id": "$copy_id",
                            "book_type": "$book_type",
                            "page_number": "$page_number",
                        },
                        "doc_ids": {"$push": "$_id"},
                        "count": {"$sum": 1}
                    }},
                    {"$match": {"count": {"$gt": 1}}}
                ]
                async for group in canvas_pages.aggregate(pipeline):
                    doc_ids = group["doc_ids"]
                    # Fetch full docs to merge strokes
                    full_docs = []
                    async for doc in canvas_pages.find({"_id": {"$in": doc_ids}}):
                        full_docs.append(doc)

                    if not full_docs:
                        continue

                    # Pick the winner: highest stroke_count, then latest last_modified
                    full_docs.sort(
                        key=lambda d: (d.get("stroke_count", 0), d.get("last_modified", "")),
                        reverse=True,
                    )
                    winner = full_docs[0]
                    losers = full_docs[1:]

                    # Merge strokes from losers into winner (deduplicate by stroke id)
                    winner_strokes = winner.get("strokes") or []
                    seen_ids = {s.get("id") for s in winner_strokes if s.get("id")}
                    merged_count = 0
                    for loser in losers:
                        for stroke in (loser.get("strokes") or []):
                            sid = stroke.get("id")
                            if sid:
                                if sid in seen_ids:
                                    continue
                                seen_ids.add(sid)
                            # No id → always keep (cannot deduplicate)
                            winner_strokes.append(stroke)
                            merged_count += 1

                    # Update winner with merged strokes if any were added
                    if merged_count > 0:
                        await canvas_pages.update_one(
                            {"_id": winner["_id"]},
                            {"$set": {
                                "strokes": winner_strokes,
                                "stroke_count": len(winner_strokes),
                            }},
                        )

                    # Delete losers
                    loser_ids = [d["_id"] for d in losers]
                    if loser_ids:
                        await canvas_pages.delete_many({"_id": {"$in": loser_ids}})
                        logger.info(
                            "Deduped canvas_pages: merged %d strokes, removed %d docs for (%s, %s, %s, %s)",
                            merged_count,
                            len(loser_ids),
                            group["_id"]["user_id"],
                            group["_id"].get("copy_id"),
                            group["_id"]["book_type"],
                            group["_id"]["page_number"],
                        )
            except Exception as e:
                logger.warning(f"canvas_pages dedup failed (non-fatal): {e}")

            # Target index includes copy_id.  Once copy_id has been backfilled,
            # proactively drop any legacy unique 3-field canvas_pages index
            # regardless of its historical name.
            try:
                missing_copy_id = await canvas_pages.count_documents(
                    {"copy_id": {"$exists": False}},
                    limit=1,
                )
                if missing_copy_id == 0:
                    await self._drop_matching_indexes(
                        canvas_pages,
                        key_specs=[[("user_id", 1), ("book_type", 1), ("page_number", 1)]],
                        names=["uniq_canvas_page", "uniq_canvas_pages_user_book_page"],
                        unique=True,
                    )
                else:
                    logger.error(
                        "Canvas copy schema is not migration-ready on %s: "
                        "canvas_pages rows are missing copy_id. Run "
                        "scripts/migrations/backfill_copy_sets.py before deployment.",
                        db_name,
                    )
                await self._ensure_index_with_spec_check(
                    canvas_pages,
                    [("user_id", 1), ("copy_id", 1), ("book_type", 1), ("page_number", 1)],
                    unique=True,
                    name="uniq_canvas_page_v2"
                )
            except Exception as exc:
                # Never recreate the legacy three-field unique index: it makes
                # two copy sets collide and is the direct cause of HTTP 409s.
                logger.error(
                    "Copy-aware canvas index setup failed on %s; run the "
                    "copy-set migration before accepting canvas writes: %s",
                    db_name,
                    exc,
                )

            # copy_sets collection
            copy_sets = db["copy_sets"]
            await self._ensure_index_with_spec_check(
                copy_sets,
                [("user_id", 1), ("is_archived", 1), ("created_at", 1)],
                name="idx_copy_sets_user"
            )
            await self._ensure_index_with_spec_check(
                copy_sets,
                [("user_id", 1), ("purpose", 1)],
                unique=True,
                partial_filter_expression={"purpose": {"$type": "string"}},
                name="uniq_copy_sets_user_purpose",
            )

            # note_classifications: target index includes copy_id
            try:
                missing_cls_copy_id = await note_cls.count_documents(
                    {"copy_id": {"$exists": False}},
                    limit=1,
                )
                if missing_cls_copy_id == 0:
                    await self._drop_matching_indexes(
                        note_cls,
                        key_specs=[[("user_id", 1), ("pen_mac", 1), ("book_type", 1), ("page_number", 1)]],
                        names=["uniq_note_page"],
                        unique=True,
                    )
                await self._ensure_index_with_spec_check(
                    note_cls,
                    [("user_id", 1), ("copy_id", 1), ("book_type", 1), ("page_number", 1)],
                    unique=True,
                    name="uniq_note_page_v2"
                )
            except Exception:
                pass  # legacy index still in place

            cls_queue = db["classification_queue"]
            try:
                missing_queue_copy_id = await cls_queue.count_documents(
                    {"copy_id": {"$exists": False}},
                    limit=1,
                )
                if missing_queue_copy_id == 0:
                    await self._drop_matching_indexes(
                        cls_queue,
                        key_specs=[[("user_id", 1), ("pen_mac", 1), ("book_type", 1), ("page_number", 1), ("db_name", 1)]],
                        names=["uniq_cls_queue_page"],
                        unique=True,
                    )
                await self._ensure_index_with_spec_check(
                    cls_queue,
                    [("user_id", 1), ("copy_id", 1), ("pen_mac", 1), ("book_type", 1), ("page_number", 1), ("db_name", 1)],
                    unique=True,
                    name="uniq_cls_queue_page_v2"
                )
            except Exception:
                await self._ensure_index_with_spec_check(
                    cls_queue,
                    [("user_id", 1), ("pen_mac", 1), ("book_type", 1), ("page_number", 1), ("db_name", 1)],
                    unique=True,
                    name="uniq_cls_queue_page"
                )
            # TTL index: auto-cleanup completed/failed jobs after 24h
            try:
                await cls_queue.create_index(
                    "created_at", expireAfterSeconds=86400, name="ttl_cls_queue"
                )
            except OperationFailure:
                pass  # index may already exist with different options

            # Notification indexes
            notifications = db["notifications"]
            await self._ensure_index_with_spec_check(
                notifications,
                [("recipient_id", 1), ("is_read", 1), ("created_at", -1)],
                name="idx_notif_recipient_unread"
            )
            try:
                await notifications.create_index(
                    "created_at", expireAfterSeconds=7776000, name="ttl_notifications_90d"
                )
            except OperationFailure:
                pass  # index may already exist with different options

            ai_usage_events = db["ai_usage_events"]
            await self._ensure_index_with_spec_check(
                ai_usage_events,
                [("user_id", 1), ("created_at", -1)],
                name="idx_ai_usage_user_time"
            )
            await self._ensure_index_with_spec_check(
                ai_usage_events,
                [("document_id", 1), ("region_id", 1), ("stage", 1), ("created_at", -1)],
                name="idx_ai_usage_document_region_stage"
            )
            await self._ensure_index_with_spec_check(
                ai_usage_events,
                [("provider", 1), ("model", 1), ("stage", 1), ("created_at", -1)],
                name="idx_ai_usage_provider_model_stage"
            )

            ai_usage_counters = db["ai_usage_counters"]
            await self._ensure_index_with_spec_check(
                ai_usage_counters,
                [("counter_id", 1)],
                unique=True,
                name="uniq_ai_usage_counter"
            )
            await self._ensure_index_with_spec_check(
                ai_usage_counters,
                [("scope", 1), ("subject_id", 1), ("period", 1), ("period_key", 1)],
                name="idx_ai_usage_counter_scope_period"
            )

            ai_usage_limits = db["ai_usage_limits"]
            await self._ensure_index_with_spec_check(
                ai_usage_limits,
                [("scope", 1), ("subject_id", 1)],
                unique=True,
                name="uniq_ai_usage_limit_subject"
            )

            answer_question_mappings = db["answer_question_mappings"]
            await self._ensure_index_with_spec_check(
                answer_question_mappings,
                [("document_id", 1), ("question_id", 1)],
                name="idx_answer_mapping_question"
            )
            await self._ensure_index_with_spec_check(
                answer_question_mappings,
                [("document_id", 1), ("answer_region_id", 1)],
                unique=True,
                name="uniq_answer_mapping_answer_region"
            )

            # Notices collection indexes
            notices = db["notices"]
            await self._ensure_index_with_spec_check(
                notices,
                [("admin_id", 1), ("created_by", 1), ("created_at", -1)],
                name="idx_notices_sent"
            )
            await self._ensure_index_with_spec_check(
                notices,
                [("admin_id", 1), ("resolved_recipients", 1), ("created_at", -1)],
                name="idx_notices_received"
            )
            try:
                await notices.create_index(
                    "created_at", expireAfterSeconds=15552000, name="ttl_notices_180d"
                )
            except OperationFailure:
                pass

            self._indexed_dbs.add(db_name)
            logger.info(f"Ensured indexes on database: {db_name}")
        except OperationFailure as e:
            logger.warning(f"Could not create one or more indexes on {db_name}: {str(e)}")

    # MongoDB async operations wrapper
    async def mongo_find_one(self, collection_name: str, filter_dict: Dict[str, Any],
                            projection: Optional[Dict[str, Any]] = None,
                            collation: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Find one document in MongoDB"""
        try:
            # Quick check if MongoDB is available
            if self.mongo_client is None:
                logger.warning(f"MongoDB not connected - cannot query {collection_name} collection")
                return None

            collection = await self._get_context_collection(collection_name)
            if collection is None:
                return None
            if collation:
                return await collection.find_one(filter_dict, projection, collation=collation)
            return await collection.find_one(filter_dict, projection)
        except Exception as e:
            logger.error(f"MongoDB find_one failed: {str(e)}")
            return None

    async def mongo_find(self, collection_name: str, filter_dict: Dict[str, Any],
                        projection: Optional[Dict[str, Any]] = None,
                        sort: Optional[List[tuple]] = None,
                        limit: Optional[int] = None,
                        skip: Optional[int] = None) -> List[Dict[str, Any]]:
        """Find documents in MongoDB"""
        try:
            # Quick check if MongoDB is available
            if self.mongo_client is None:
                logger.warning(f"MongoDB not connected - cannot query {collection_name} collection")
                return []

            collection = await self._get_context_collection(collection_name)
            if collection is None:
                return []

            cursor = collection.find(filter_dict, projection)

            if sort:
                cursor = cursor.sort(sort)
            if skip:
                cursor = cursor.skip(skip)
            if limit:
                cursor = cursor.limit(limit)

            # Use a reasonable max limit to avoid hanging on large collections
            max_length = limit if limit is not None else 1000
            return await cursor.to_list(length=max_length)
        except Exception as e:
            logger.error(f"MongoDB find failed: {str(e)}")
            return []

    async def mongo_insert_one(self, collection_name: str, document: Dict[str, Any]) -> Optional[str]:
        """Insert one document to MongoDB"""
        try:
            collection = await self._get_context_collection(collection_name)
            if collection is None:
                return None
            result = await collection.insert_one(document)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"MongoDB insert_one failed: {str(e)}")
            return None

    async def mongo_insert_many(self, collection_name: str, documents: List[Dict[str, Any]]) -> Optional[List[str]]:
        """Insert many documents to MongoDB"""
        try:
            collection = await self._get_context_collection(collection_name)
            if collection is None:
                return None
            result = await collection.insert_many(documents)
            return [str(id_) for id_ in result.inserted_ids]
        except Exception as e:
            logger.error(f"MongoDB insert_many failed: {str(e)}")
            return None

    async def mongo_update_one(self, collection_name: str, filter_dict: Dict[str, Any],
                              update_dict: Dict[str, Any], upsert: bool = False) -> bool:
        """Update one document in MongoDB"""
        try:
            collection = await self._get_context_collection(collection_name)
            if collection is None:
                return False
            result = await collection.update_one(filter_dict, update_dict, upsert=upsert)
            return result.modified_count > 0 or (upsert and result.upserted_id is not None)
        except Exception as e:
            logger.error(f"MongoDB update_one failed: {str(e)}")
            return False

    async def mongo_update_many(self, collection_name: str, filter_dict: Dict[str, Any],
                               update_dict: Dict[str, Any]) -> int:
        """Update multiple documents in MongoDB"""
        try:
            collection = await self._get_context_collection(collection_name)
            if collection is None:
                return 0
            result = await collection.update_many(filter_dict, update_dict)
            return result.modified_count
        except Exception as e:
            logger.error(f"MongoDB update_many failed: {str(e)}")
            return 0

    async def mongo_delete_one(self, collection_name: str, filter_dict: Dict[str, Any]) -> bool:
        """Delete one document from MongoDB"""
        try:
            collection = await self._get_context_collection(collection_name)
            if collection is None:
                return False
            result = await collection.delete_one(filter_dict)
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"MongoDB delete_one failed: {str(e)}")
            return False

    async def mongo_delete_many(self, collection_name: str, filter_dict: Dict[str, Any]) -> int:
        """Delete multiple documents from MongoDB"""
        try:
            collection = await self._get_context_collection(collection_name)
            if collection is None:
                return 0
            result = await collection.delete_many(filter_dict)
            return result.deleted_count
        except Exception as e:
            logger.error(f"MongoDB delete_many failed: {str(e)}")
            return 0

    async def mongo_count(self, collection_name: str, filter_dict: Dict[str, Any] = None) -> int:
        """Count documents in MongoDB collection"""
        try:
            collection = await self._get_context_collection(collection_name)
            if collection is None:
                return 0
            filter_dict = filter_dict or {}
            return await collection.count_documents(filter_dict)
        except Exception as e:
            logger.error(f"MongoDB count failed: {str(e)}")
            return 0

    async def mongo_aggregate(self, collection_name: str, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run aggregation pipeline on MongoDB collection"""
        try:
            collection = await self._get_context_collection(collection_name)
            if collection is None:
                return []
            cursor = collection.aggregate(pipeline)
            return await cursor.to_list(length=1000)
        except Exception as e:
            logger.error(f"MongoDB aggregate failed: {str(e)}")
            return []

    async def health_check(self) -> bool:
        """Check health of MongoDB connection"""
        try:
            mongo_healthy = False
            if not DISABLE_MONGODB and self.mongo_client:
                try:
                    await asyncio.wait_for(
                        self.mongo_client.admin.command('ping'),
                        timeout=5.0
                    )
                    mongo_healthy = True
                except Exception as e:
                    logger.error(f"MongoDB health check failed: {str(e)}")
                    # Close client to stop background threads from retrying
                    try:
                        self.mongo_client.close()
                    except Exception:
                        pass
                    self.mongo_client = None
                    self.mongo_master_db = None
                    self.mongo_db_b2c = None
                    self.mongo_tenant_dbs.clear()
                    mongo_healthy = False

            return mongo_healthy

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False

    async def close(self) -> None:
        """Close all database connections"""
        try:
            # Close MongoDB connection
            if self.mongo_client:
                self.mongo_client.close()
                logger.info("🔌 MongoDB connection closed")

            logger.info("🔌 Database connections closed")

        except Exception as e:
            logger.error(f"Error closing database connections: {str(e)}")

    # ==================== B2C Database Operations ====================
    # These methods operate on the stoody-b2c database for B2C users
    # They mirror the main methods but use get_b2c_collection instead

    async def b2c_find_one(self, collection_name: str, filter_dict: Dict[str, Any],
                           projection: Optional[Dict[str, Any]] = None,
                           collation: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Find one document in B2C MongoDB database"""
        try:
            if self.mongo_client is None or self.mongo_db_b2c is None:
                logger.warning(f"B2C MongoDB not connected - cannot query {collection_name}")
                return None
            collection = await self.get_b2c_collection(collection_name)
            if collection is None:
                return None
            if collation:
                return await collection.find_one(filter_dict, projection, collation=collation)
            return await collection.find_one(filter_dict, projection)
        except Exception as e:
            logger.error(f"B2C MongoDB find_one failed: {str(e)}")
            return None

    async def b2c_find(self, collection_name: str, filter_dict: Dict[str, Any],
                       projection: Optional[Dict[str, Any]] = None,
                       sort: Optional[List[tuple]] = None,
                       limit: Optional[int] = None,
                       skip: Optional[int] = None) -> List[Dict[str, Any]]:
        """Find documents in B2C MongoDB database"""
        try:
            if self.mongo_client is None or self.mongo_db_b2c is None:
                logger.warning(f"B2C MongoDB not connected - cannot query {collection_name}")
                return []
            collection = await self.get_b2c_collection(collection_name)
            if collection is None:
                return []
            cursor = collection.find(filter_dict, projection)
            if sort:
                cursor = cursor.sort(sort)
            if skip:
                cursor = cursor.skip(skip)
            if limit:
                cursor = cursor.limit(limit)
            max_length = limit if limit is not None else 1000
            return await cursor.to_list(length=max_length)
        except Exception as e:
            logger.error(f"B2C MongoDB find failed: {str(e)}")
            return []

    async def b2c_insert_one(self, collection_name: str, document: Dict[str, Any]) -> Optional[str]:
        """Insert one document to B2C MongoDB database"""
        try:
            collection = await self.get_b2c_collection(collection_name)
            if collection is None:
                return None
            result = await collection.insert_one(document)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"B2C MongoDB insert_one failed: {str(e)}")
            return None

    async def b2c_update_one(self, collection_name: str, filter_dict: Dict[str, Any],
                             update_dict: Dict[str, Any], upsert: bool = False) -> bool:
        """Update one document in B2C MongoDB database"""
        try:
            collection = await self.get_b2c_collection(collection_name)
            if collection is None:
                return False
            result = await collection.update_one(filter_dict, update_dict, upsert=upsert)
            return result.modified_count > 0 or (upsert and result.upserted_id is not None)
        except Exception as e:
            logger.error(f"B2C MongoDB update_one failed: {str(e)}")
            return False

    async def b2c_count(self, collection_name: str, filter_dict: Dict[str, Any] = None) -> int:
        """Count documents in a B2C MongoDB collection"""
        try:
            collection = await self.get_b2c_collection(collection_name)
            if collection is None:
                return 0
            filter_dict = filter_dict or {}
            return await collection.count_documents(filter_dict)
        except Exception as e:
            logger.error(f"B2C MongoDB count failed: {str(e)}")
            return 0

    async def b2c_aggregate(self, collection_name: str, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run aggregation pipeline on a B2C MongoDB collection"""
        try:
            collection = await self.get_b2c_collection(collection_name)
            if collection is None:
                return []
            cursor = collection.aggregate(pipeline)
            return await cursor.to_list(length=1000)
        except Exception as e:
            logger.error(f"B2C MongoDB aggregate failed: {str(e)}")
            return []

    async def b2c_delete_one(self, collection_name: str, filter_dict: Dict[str, Any]) -> bool:
        """Delete one document from B2C MongoDB database"""
        try:
            collection = await self.get_b2c_collection(collection_name)
            if collection is None:
                return False
            result = await collection.delete_one(filter_dict)
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"B2C MongoDB delete_one failed: {str(e)}")
            return False

    async def b2c_delete_many(self, collection_name: str, filter_dict: Dict[str, Any]) -> bool:
        """Delete multiple documents from B2C MongoDB database"""
        try:
            collection = await self.get_b2c_collection(collection_name)
            if collection is None:
                return False
            result = await collection.delete_many(filter_dict)
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"B2C MongoDB delete_many failed: {str(e)}")
            return False
