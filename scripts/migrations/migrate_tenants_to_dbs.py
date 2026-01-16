"""
Migrate tenant data from shared DB into per-tenant databases.

Usage:
    cd backend
    python scripts/migrations/migrate_tenants_to_dbs.py --dry-run
    python scripts/migrations/migrate_tenants_to_dbs.py --apply-indexes
    python scripts/migrations/migrate_tenants_to_dbs.py --bootstrap-from-admins
"""

import argparse
import asyncio
import logging
import os
import random
import string
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ReplaceOne
from pymongo.errors import BulkWriteError

# Add parent directory to path for imports
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, backend_dir)

# Load .env file from backend directory
try:
    from dotenv import load_dotenv
    env_path = os.path.join(backend_dir, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
except ImportError:
    pass

from scripts.migrations.apply_tenant_indexes import apply_indexes

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_REGION = "in"
DEFAULT_INSTANCE = "001"

TENANT_COLLECTIONS = [
    "admins",
    "canvas_history",
    "chat_analytics",
    "chat_conversations",
    "chat_documents",
    "chat_sessions",
    "devices",
    "document_annotations",
    "document_regions",
    "students",
    "student_activities",
    "student_activity_log",
    "student_sessions",
    "student_test_attempts",
    "tutors",
    "tutor_activity_log",
    "documents",
    "images",
    "mcq_solutions",
    "online_classes",
    "pen_calibrations",
    "pens",
    "practice_sessions",
    "questions",
    "question_attempts",
    "school_settings",
    "session_promotions",
    "sessions",
    "smartboard_sessions",
    "strokes",
    "video_views",
    "videos",
    "pen_tokens",
    "assignments",
    "meetings",
    "notifications",
    "class_schedules",
]

EXCLUDED_COLLECTIONS = {
    "tenants",
    "system_settings",
    "audit_logs",
    "billing",
    "feature_flags",
}


def generate_institution_id(existing_ids: Set[str]) -> str:
    """Generate a unique institution_id in the XXXXX-XXX-XX format."""
    while True:
        parts = [
            "".join(random.choices(string.ascii_uppercase + string.digits, k=5)),
            "".join(random.choices(string.ascii_uppercase + string.digits, k=3)),
            "".join(random.choices(string.ascii_uppercase + string.digits, k=2)),
        ]
        institution_id = "-".join(parts)
        if institution_id not in existing_ids:
            existing_ids.add(institution_id)
            return institution_id


def normalize_institution_id(institution_id: str) -> str:
    return institution_id.strip().upper()


def normalize_instance(instance: Optional[Any]) -> str:
    if instance is None:
        return DEFAULT_INSTANCE
    if isinstance(instance, int):
        return f"{instance:03d}"
    value = str(instance).strip()
    if value.isdigit():
        return f"{int(value):03d}"
    return value


def build_db_name(region: str, institution_id: str, instance: str) -> str:
    institution_segment = institution_id.replace("-", "_")
    return f"skb_{region}_{institution_segment}_{instance}"


def build_admin_conditions(admin_oid: ObjectId) -> List[Dict[str, Any]]:
    return [
        {"admin_id": admin_oid},
        {"admin_id": str(admin_oid)},
        {"created_by": admin_oid},
        {"created_by": str(admin_oid)},
    ]


def _build_in_filter(field: str, values: List[Any]) -> Dict[str, Any]:
    if not values:
        return {"_id": None}
    return {field: {"$in": values}}


def _build_or(filters: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [f for f in filters if f]
    if not valid:
        return {"_id": None}
    if len(valid) == 1:
        return valid[0]
    return {"$or": valid}


class TenantMigrator:
    def __init__(self, mongo_uri: str, source_db: str, master_db: str, dry_run: bool = False,
                 bootstrap_from_admins: bool = False, apply_indexes_flag: bool = False,
                 delete_source: bool = False, collections: Optional[Set[str]] = None):
        self.mongo_uri = mongo_uri
        self.source_db_name = source_db
        self.master_db_name = master_db
        self.dry_run = dry_run
        self.bootstrap_from_admins = bootstrap_from_admins
        self.apply_indexes_flag = apply_indexes_flag
        self.delete_source = delete_source
        self.collections = collections
        self.client: Optional[AsyncIOMotorClient] = None
        self.source_db = None
        self.master_db = None
        self.existing_institution_ids: Set[str] = set()
        self.existing_db_names: Set[str] = set()

    async def connect(self) -> None:
        self.client = AsyncIOMotorClient(self.mongo_uri)
        self.source_db = self.client[self.source_db_name]
        self.master_db = self.client[self.master_db_name]

    async def close(self) -> None:
        if self.client:
            self.client.close()

    async def load_existing_registry(self) -> List[Dict[str, Any]]:
        tenants = await self.master_db["tenants"].find({}).to_list(length=None)
        for tenant in tenants:
            inst_id = tenant.get("institution_id")
            if inst_id:
                self.existing_institution_ids.add(normalize_institution_id(inst_id))
            db_name = tenant.get("db_name")
            if db_name:
                self.existing_db_names.add(db_name)
        return tenants

    async def bootstrap_registry_from_admins(self) -> List[Dict[str, Any]]:
        logger.info("Bootstrapping tenant registry from existing admins")
        admins = await self.source_db["admins"].find({}).to_list(length=None)
        tenants: List[Dict[str, Any]] = []

        for admin in admins:
            inst_id = generate_institution_id(self.existing_institution_ids)
            region = DEFAULT_REGION
            instance = DEFAULT_INSTANCE
            db_name = build_db_name(region, inst_id, instance)
            while db_name in self.existing_db_names:
                instance = normalize_instance(int(instance) + 1)
                db_name = build_db_name(region, inst_id, instance)
            self.existing_db_names.add(db_name)

            tenant_doc = {
                "tenant_id": db_name,
                "db_name": db_name,
                "institution_id": inst_id,
                "region": region,
                "instance": instance,
                "subdomain": admin.get("subdomain"),
                "admin_id": admin.get("_id"),
                "admin_email": admin.get("email"),
                "admin_full_name": admin.get("full_name") or admin.get("name"),
                "status": "active",
                "created_at": datetime.utcnow(),
            }

            if not self.dry_run:
                result = await self.master_db["tenants"].insert_one(tenant_doc)
                tenant_doc["_id"] = result.inserted_id
            else:
                tenant_doc["_id"] = ObjectId()
            tenants.append(tenant_doc)

            logger.info("Registered tenant for admin %s -> %s", admin.get("email"), db_name)

        return tenants

    async def ensure_tenant_fields(self, tenant: Dict[str, Any]) -> Dict[str, Any]:
        updates: Dict[str, Any] = {}

        inst_id = tenant.get("institution_id")
        if inst_id:
            normalized = normalize_institution_id(inst_id)
            if normalized != inst_id:
                updates["institution_id"] = normalized
            inst_id = normalized
        else:
            inst_id = generate_institution_id(self.existing_institution_ids)
            updates["institution_id"] = inst_id

        region = tenant.get("region") or DEFAULT_REGION
        region = str(region).lower()
        if region != tenant.get("region"):
            updates["region"] = region

        instance = normalize_instance(tenant.get("instance"))
        if instance != tenant.get("instance"):
            updates["instance"] = instance

        db_name = tenant.get("db_name")
        if db_name and "-" in db_name:
            normalized_db_name = build_db_name(region, inst_id, instance)
            if normalized_db_name != db_name:
                db_name = normalized_db_name
                updates["db_name"] = db_name
                self.existing_db_names.add(db_name)
        if not db_name:
            db_name = build_db_name(region, inst_id, instance)
            while db_name in self.existing_db_names:
                instance = normalize_instance(int(instance) + 1)
                updates["instance"] = instance
                db_name = build_db_name(region, inst_id, instance)
            updates["db_name"] = db_name
            self.existing_db_names.add(db_name)

        tenant_id = tenant.get("tenant_id")
        if not tenant_id:
            updates["tenant_id"] = db_name

        admin_id = tenant.get("admin_id")
        if not admin_id and tenant.get("admin_email"):
            admin = await self.source_db["admins"].find_one({"email": tenant.get("admin_email")})
            if admin:
                updates["admin_id"] = admin.get("_id")

        if updates and not self.dry_run:
            await self.master_db["tenants"].update_one({"_id": tenant["_id"]}, {"$set": updates})

        return {**tenant, **updates}

    async def migrate_collection(self, source_collection, target_collection,
                                 filter_dict: Dict[str, Any],
                                 transform=None) -> int:
        cursor = source_collection.find(filter_dict)
        batch: List[Dict[str, Any]] = []
        total = 0

        async for doc in cursor:
            if transform:
                doc = transform(doc)
            batch.append(doc)
            if len(batch) >= 500:
                total += await self.write_batch(target_collection, batch)
                batch = []

        if batch:
            total += await self.write_batch(target_collection, batch)

        return total

    async def maybe_delete_source(self, source_collection, filter_dict: Dict[str, Any]) -> int:
        if self.dry_run or not self.delete_source:
            return 0
        result = await source_collection.delete_many(filter_dict)
        return result.deleted_count

    async def write_batch(self, collection, docs: List[Dict[str, Any]]) -> int:
        if not docs:
            return 0
        if self.dry_run:
            return len(docs)

        ops = [ReplaceOne({"_id": doc["_id"]}, doc, upsert=True) for doc in docs]
        try:
            result = await collection.bulk_write(ops, ordered=False)
            return result.upserted_count + result.modified_count + result.inserted_count
        except BulkWriteError as exc:
            logger.warning("Bulk write error on %s: %s", collection.name, exc.details)
            details = exc.details or {}
            return details.get("nInserted", 0) + details.get("nUpserted", 0) + details.get("nModified", 0)

    async def migrate_tenant(self, tenant: Dict[str, Any]) -> None:
        tenant = await self.ensure_tenant_fields(tenant)
        db_name = tenant.get("db_name")
        if not db_name:
            logger.warning("Skipping tenant without db_name: %s", tenant.get("_id"))
            return

        admin_id = tenant.get("admin_id")
        if not admin_id:
            logger.warning("Skipping tenant without admin_id: %s", tenant.get("_id"))
            return
        try:
            admin_oid = ObjectId(admin_id) if not isinstance(admin_id, ObjectId) else admin_id
        except Exception:
            logger.warning("Skipping tenant with invalid admin_id: %s", tenant.get("_id"))
            return
        admin_conditions = build_admin_conditions(admin_oid)

        tenant_db = self.client[db_name]
        logger.info("Migrating tenant %s -> %s", tenant.get("subdomain") or tenant.get("tenant_id"), db_name)

        student_cursor = self.source_db["students"].find({"admin_id": admin_oid}, {"_id": 1, "username": 1})
        student_ids: List[ObjectId] = []
        student_usernames: List[str] = []
        async for student in student_cursor:
            student_ids.append(student["_id"])
            username = student.get("username")
            if username:
                student_usernames.append(username)
        student_id_values = list({*student_ids, *[str(sid) for sid in student_ids]})
        user_id_values = list({*student_id_values, *student_usernames})

        tutor_cursor = self.source_db["tutors"].find(
            {"admin_id": admin_oid},
            {"_id": 1, "username": 1, "tutor_id": 1},
        )
        tutor_ids: List[ObjectId] = []
        tutor_usernames: List[str] = []
        tutor_custom_ids: List[str] = []
        async for tutor in tutor_cursor:
            tutor_ids.append(tutor["_id"])
            tname = tutor.get("username")
            if tname:
                tutor_usernames.append(tname)
            tid = tutor.get("tutor_id")
            if tid:
                tutor_custom_ids.append(str(tid))
        tutor_id_values = list({*tutor_ids, *[str(tid) for tid in tutor_ids], *tutor_usernames, *tutor_custom_ids})

        document_cursor = self.source_db["documents"].find(
            {"admin_id": admin_oid},
            {"_id": 1, "document_id": 1, "filename": 1},
        )
        document_ids: List[ObjectId] = []
        document_custom_ids: List[str] = []
        document_filenames: List[str] = []
        async for doc in document_cursor:
            document_ids.append(doc["_id"])
            doc_id = doc.get("document_id")
            if doc_id:
                document_custom_ids.append(str(doc_id))
            filename = doc.get("filename")
            if filename:
                document_filenames.append(str(filename))
        document_id_values = list({*document_ids, *[str(did) for did in document_ids], *document_custom_ids})

        question_cursor = self.source_db["questions"].find(
            {"$or": admin_conditions},
            {"_id": 1, "id": 1, "question_id": 1},
        )
        question_ids: List[ObjectId] = []
        question_custom_ids: List[str] = []
        async for q in question_cursor:
            question_ids.append(q["_id"])
            qid = q.get("id") or q.get("question_id")
            if qid:
                question_custom_ids.append(str(qid))
        question_id_values = list({*question_ids, *[str(qid) for qid in question_ids], *question_custom_ids})

        collections_to_migrate = self.collections or set(TENANT_COLLECTIONS)
        for collection_name in TENANT_COLLECTIONS:
            if collection_name not in collections_to_migrate:
                continue
            source_collection = self.source_db[collection_name]
            target_collection = tenant_db[collection_name]

            if collection_name == "admins":
                filter_dict = {"_id": admin_oid}

                def transform_admin(doc: Dict[str, Any]) -> Dict[str, Any]:
                    updated = dict(doc)
                    updated["role"] = "master_admin"
                    updated.setdefault("permissions", [])
                    updated.setdefault("created_by", None)
                    return updated

                count = await self.migrate_collection(source_collection, target_collection, filter_dict, transform_admin)
            elif collection_name in {"strokes", "sessions", "pens", "devices", "pen_calibrations"}:
                filter_dict = _build_in_filter("user_id", user_id_values)
                count = await self.migrate_collection(source_collection, target_collection, filter_dict)
            elif collection_name in {"student_activities", "student_sessions", "practice_sessions", "video_views"}:
                filter_dict = _build_or([
                    _build_in_filter("student_id", student_id_values),
                    _build_in_filter("user_id", user_id_values),
                    _build_in_filter("username", student_usernames),
                ])
                count = await self.migrate_collection(source_collection, target_collection, filter_dict)
            elif collection_name in {"tutor_activity_log"}:
                conditions = list(admin_conditions)
                conditions.append(_build_in_filter("tutor_id", tutor_id_values))
                conditions.append(_build_in_filter("username", tutor_usernames))
                filter_dict = _build_or(conditions)
                count = await self.migrate_collection(source_collection, target_collection, filter_dict)
            elif collection_name in {"online_classes", "videos"}:
                conditions = list(admin_conditions)
                conditions.append(_build_in_filter("tutor_id", tutor_id_values))
                filter_dict = _build_or(conditions)
                count = await self.migrate_collection(source_collection, target_collection, filter_dict)
            elif collection_name in {"documents", "school_settings"}:
                filter_dict = _build_or(list(admin_conditions))
                count = await self.migrate_collection(source_collection, target_collection, filter_dict)
            elif collection_name in {"document_annotations", "document_regions", "images", "chat_documents"}:
                conditions = list(admin_conditions)
                conditions.append(_build_in_filter("document_id", document_id_values))
                conditions.append(_build_in_filter("doc_id", document_id_values))
                conditions.append(_build_in_filter("source_pdf", document_filenames))
                filter_dict = _build_or(conditions)
                count = await self.migrate_collection(source_collection, target_collection, filter_dict)
            elif collection_name in {"questions", "question_attempts", "mcq_solutions"}:
                conditions = list(admin_conditions)
                conditions.append(_build_in_filter("question_id", question_id_values))
                conditions.append(_build_in_filter("id", question_id_values))
                conditions.append(_build_in_filter("student_id", student_id_values))
                filter_dict = _build_or(conditions)
                count = await self.migrate_collection(source_collection, target_collection, filter_dict)
            elif collection_name in {"chat_sessions", "chat_conversations", "chat_analytics"}:
                conditions = list(admin_conditions)
                conditions.append(_build_in_filter("student_id", student_id_values))
                conditions.append(_build_in_filter("user_id", user_id_values))
                conditions.append(_build_in_filter("tutor_id", tutor_id_values))
                filter_dict = _build_or(conditions)
                count = await self.migrate_collection(source_collection, target_collection, filter_dict)
            elif collection_name == "canvas_history":
                conditions = list(admin_conditions)
                conditions.append(_build_in_filter("user_id", user_id_values))
                conditions.append(_build_in_filter("student_id", student_id_values))
                filter_dict = _build_or(conditions)
                count = await self.migrate_collection(source_collection, target_collection, filter_dict)
            elif collection_name == "pen_tokens":
                filter_dict = {"student_id": {"$in": student_ids}} if student_ids else {"student_id": None}
                count = await self.migrate_collection(source_collection, target_collection, filter_dict)
            elif collection_name in {"student_activity_log", "student_test_attempts"}:
                conditions = list(admin_conditions)
                if student_id_values:
                    conditions.append({"student_id": {"$in": student_id_values}})
                filter_dict = {"$or": conditions}
                count = await self.migrate_collection(source_collection, target_collection, filter_dict)
            else:
                filter_dict = {"$or": admin_conditions}
                count = await self.migrate_collection(source_collection, target_collection, filter_dict)

            logger.info("  %s: %s documents", collection_name, count)
            deleted = await self.maybe_delete_source(source_collection, filter_dict)
            if deleted:
                logger.info("  %s: deleted %s legacy documents", collection_name, deleted)

        if self.apply_indexes_flag:
            await apply_indexes(tenant_db, dry_run=self.dry_run)

    async def run(self) -> None:
        await self.connect()
        try:
            tenants = await self.load_existing_registry()

            if not tenants and self.bootstrap_from_admins:
                tenants = await self.bootstrap_registry_from_admins()

            if not tenants:
                logger.warning("No tenants found in master registry")
                return

            for tenant in tenants:
                await self.migrate_tenant(tenant)
        finally:
            await self.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate tenants to per-tenant databases")
    parser.add_argument("--dry-run", action="store_true", help="Log changes without writing data")
    parser.add_argument("--bootstrap-from-admins", action="store_true", help="Create tenant registry from admins")
    parser.add_argument("--apply-indexes", action="store_true", help="Apply tenant index template after copy")
    parser.add_argument("--delete-source", action="store_true", help="Delete migrated docs from source DB")
    parser.add_argument(
        "--collections",
        default=None,
        help="Comma-separated list of collections to migrate (default: all tenant collections)",
    )
    parser.add_argument("--mongo-uri", default=None, help="MongoDB URI (default: from env)")
    parser.add_argument("--source-db", default=None, help="Source DB name (default: from env)")
    parser.add_argument("--master-db", default=None, help="Master DB name (default: from env)")
    args = parser.parse_args()

    collections: Optional[Set[str]] = None
    if args.collections:
        collections = {c.strip() for c in args.collections.split(",") if c.strip()}
        unknown = collections - set(TENANT_COLLECTIONS)
        if unknown:
            logger.warning("Unknown collections requested: %s", ", ".join(sorted(unknown)))

    mongo_uri = args.mongo_uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    source_db = args.source_db or os.getenv("MONGODB_DB_NAME", "skillbot_db")
    master_db = args.master_db or os.getenv("MONGODB_DB_MASTER", "skb_master")

    logger.info("MongoDB URI: %s...", mongo_uri[:50])
    logger.info("Source DB: %s", source_db)
    logger.info("Master DB: %s", master_db)

    migrator = TenantMigrator(
        mongo_uri=mongo_uri,
        source_db=source_db,
        master_db=master_db,
        dry_run=args.dry_run,
        bootstrap_from_admins=args.bootstrap_from_admins,
        apply_indexes_flag=args.apply_indexes,
        delete_source=args.delete_source,
        collections=collections,
    )

    asyncio.run(migrator.run())


if __name__ == "__main__":
    main()
