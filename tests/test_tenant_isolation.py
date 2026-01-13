"""
Tenant Isolation Tests

Tests to verify that the multi-tenant data isolation is working correctly.
These tests ensure that:
1. Tenant context is required for tenant-scoped collections
2. Cross-tenant data access is blocked
3. admin_id is automatically injected into queries
4. TenantAwareDB properly filters data

Usage:
    cd backend
    pytest tests/test_tenant_isolation.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from bson import ObjectId

# Import tenant isolation components
from core.tenant import (
    TenantContext,
    TenantAwareDB,
    TenantContextError,
    TenantIsolationError,
    TENANT_SCOPED_COLLECTIONS,
    GLOBAL_COLLECTIONS,
)


class TestTenantContext:
    """Tests for TenantContext class"""

    def setup_method(self):
        """Clear context before each test"""
        TenantContext.clear()

    def teardown_method(self):
        """Clear context after each test"""
        TenantContext.clear()

    def test_set_and_get_context(self):
        """Test setting and getting tenant context"""
        admin_id = "admin123"
        TenantContext.set(admin_id=admin_id, user_type="admin")

        ctx = TenantContext.get()
        assert ctx is not None
        assert ctx["admin_id"] == admin_id
        assert ctx["user_type"] == "admin"

    def test_get_admin_id(self):
        """Test getting admin_id from context"""
        admin_id = "admin456"
        TenantContext.set(admin_id=admin_id, user_type="admin")

        assert TenantContext.get_admin_id() == admin_id

    def test_get_admin_oid(self):
        """Test getting admin_id as ObjectId"""
        admin_id = str(ObjectId())
        TenantContext.set(admin_id=admin_id, user_type="admin")

        oid = TenantContext.get_admin_oid()
        assert isinstance(oid, ObjectId)
        assert str(oid) == admin_id

    def test_get_admin_oid_invalid(self):
        """Test getting admin_oid with invalid ObjectId returns None"""
        TenantContext.set(admin_id="invalid-oid", user_type="admin")
        assert TenantContext.get_admin_oid() is None

    def test_clear_context(self):
        """Test clearing tenant context"""
        TenantContext.set(admin_id="admin123", user_type="admin")
        TenantContext.clear()

        assert TenantContext.get() is None
        assert TenantContext.get_admin_id() is None

    def test_require_raises_without_context(self):
        """Test that require() raises when context not set"""
        with pytest.raises(TenantContextError):
            TenantContext.require()

    def test_require_raises_without_admin_id(self):
        """Test that require() raises when admin_id is None"""
        TenantContext.set(admin_id=None, user_type="admin")

        with pytest.raises(TenantContextError):
            TenantContext.require()

    def test_require_returns_context(self):
        """Test that require() returns context when properly set"""
        admin_id = "admin789"
        TenantContext.set(admin_id=admin_id, user_type="admin")

        ctx = TenantContext.require()
        assert ctx["admin_id"] == admin_id


class TestTenantAwareDB:
    """Tests for TenantAwareDB wrapper"""

    def setup_method(self):
        """Setup mock database and clear context"""
        TenantContext.clear()
        self.mock_db = MagicMock()
        self.mock_db.mongo_find_one = AsyncMock(return_value=None)
        self.mock_db.mongo_find = AsyncMock(return_value=[])
        self.mock_db.mongo_insert_one = AsyncMock(return_value="inserted_id")
        self.mock_db.mongo_update_one = AsyncMock(return_value=True)
        self.mock_db.mongo_delete_one = AsyncMock(return_value=True)
        self.mock_db.mongo_count = AsyncMock(return_value=0)
        self.mock_db.mongo_aggregate = AsyncMock(return_value=[])

        self.tenant_db = TenantAwareDB(self.mock_db)

    def teardown_method(self):
        """Clear context after each test"""
        TenantContext.clear()

    @pytest.mark.asyncio
    async def test_find_on_tenant_scoped_requires_context(self):
        """Test that querying tenant-scoped collection requires context"""
        with pytest.raises(TenantContextError):
            await self.tenant_db.find("students", {})

    @pytest.mark.asyncio
    async def test_find_on_global_collection_no_context_needed(self):
        """Test that global collections don't require context"""
        # Should not raise
        await self.tenant_db.find("admins", {"email": "test@example.com"})

        # Verify no admin_id filter was added
        self.mock_db.mongo_find.assert_called_once()
        call_args = self.mock_db.mongo_find.call_args
        assert "admin_id" not in call_args[0][1]

    @pytest.mark.asyncio
    async def test_find_auto_injects_admin_id(self):
        """Test that admin_id is automatically injected into queries"""
        admin_id = str(ObjectId())
        TenantContext.set(admin_id=admin_id, user_type="admin")

        await self.tenant_db.find("students", {"grade": "10"})

        # Verify admin_id was added to filter
        self.mock_db.mongo_find.assert_called_once()
        call_args = self.mock_db.mongo_find.call_args
        filter_dict = call_args[0][1]

        assert "admin_id" in filter_dict
        assert str(filter_dict["admin_id"]) == admin_id
        assert filter_dict["grade"] == "10"

    @pytest.mark.asyncio
    async def test_insert_auto_sets_admin_id(self):
        """Test that admin_id is automatically added to inserts"""
        admin_id = str(ObjectId())
        TenantContext.set(admin_id=admin_id, user_type="admin")

        doc = {"name": "Test Student", "grade": "10"}
        await self.tenant_db.insert_one("students", doc)

        # Verify admin_id was added to document
        self.mock_db.mongo_insert_one.assert_called_once()
        call_args = self.mock_db.mongo_insert_one.call_args
        inserted_doc = call_args[0][1]

        assert "admin_id" in inserted_doc
        assert str(inserted_doc["admin_id"]) == admin_id

    @pytest.mark.asyncio
    async def test_cross_tenant_access_blocked(self):
        """Test that accessing another tenant's data raises error"""
        admin_a_id = str(ObjectId())
        admin_b_id = str(ObjectId())

        TenantContext.set(admin_id=admin_a_id, user_type="admin")

        # Try to query with a different admin_id
        with pytest.raises(TenantIsolationError):
            await self.tenant_db.find("students", {"admin_id": admin_b_id})

    @pytest.mark.asyncio
    async def test_same_admin_id_in_filter_allowed(self):
        """Test that including same admin_id in filter is allowed"""
        admin_id = str(ObjectId())
        TenantContext.set(admin_id=admin_id, user_type="admin")

        # Should not raise - same admin_id
        await self.tenant_db.find("students", {"admin_id": ObjectId(admin_id)})

    @pytest.mark.asyncio
    async def test_bypass_tenant_filter(self):
        """Test bypass_tenant_filter context manager"""
        admin_id = str(ObjectId())
        TenantContext.set(admin_id=admin_id, user_type="admin")

        with self.tenant_db.bypass_tenant_filter():
            await self.tenant_db.find("students", {})

        # Verify no admin_id filter was added
        self.mock_db.mongo_find.assert_called_once()
        call_args = self.mock_db.mongo_find.call_args
        filter_dict = call_args[0][1]

        assert "admin_id" not in filter_dict

    @pytest.mark.asyncio
    async def test_aggregate_injects_match_stage(self):
        """Test that aggregation pipelines get tenant filter injected"""
        admin_id = str(ObjectId())
        TenantContext.set(admin_id=admin_id, user_type="admin")

        pipeline = [{"$group": {"_id": "$grade", "count": {"$sum": 1}}}]
        await self.tenant_db.aggregate("students", pipeline)

        # Verify $match stage was prepended
        self.mock_db.mongo_aggregate.assert_called_once()
        call_args = self.mock_db.mongo_aggregate.call_args
        modified_pipeline = call_args[0][1]

        assert len(modified_pipeline) == 2
        assert "$match" in modified_pipeline[0]
        assert "admin_id" in modified_pipeline[0]["$match"]


class TestCollectionClassification:
    """Tests for collection classification"""

    def test_students_is_tenant_scoped(self):
        """Test that students collection is tenant-scoped"""
        assert "students" in TENANT_SCOPED_COLLECTIONS

    def test_admins_is_global(self):
        """Test that admins collection is global"""
        assert "admins" in GLOBAL_COLLECTIONS

    def test_smartboard_sessions_is_tenant_scoped(self):
        """Test that smartboard_sessions is tenant-scoped"""
        assert "smartboard_sessions" in TENANT_SCOPED_COLLECTIONS

    def test_expected_tenant_collections(self):
        """Test that all expected collections are classified"""
        expected = {
            "students", "documents", "tutors", "questions",
            "question_attempts", "student_activity_log", "chat_sessions",
            "student_test_attempts", "assignments", "meetings",
            "notifications", "class_schedules", "smartboard_sessions"
        }
        assert expected.issubset(TENANT_SCOPED_COLLECTIONS)


class TestIntegration:
    """Integration tests for tenant isolation"""

    @pytest.mark.asyncio
    async def test_tutor_sees_only_their_admin_students(self):
        """Test that tutors can only see students from their admin"""
        # This is a placeholder for integration tests
        # In a real test, you would:
        # 1. Create two admins
        # 2. Create students for each admin
        # 3. Login as tutor of admin A
        # 4. Verify tutor only sees admin A's students
        pass

    @pytest.mark.asyncio
    async def test_admin_cannot_access_other_admin_data(self):
        """Test that admins cannot access each other's data"""
        # Placeholder for integration test
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
