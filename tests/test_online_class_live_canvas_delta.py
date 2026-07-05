import uuid

import pytest


def test_live_canvas_delta_page_payload_is_scoped_to_online_class_copy():
    from api.v1.online_class.router import (
        TeacherLiveCanvasDeltaRequest,
        TeacherLiveCanvasDeltaPage,
        _decode_monitoring_page_key,
        _build_teacher_live_delta_page_payload,
    )

    page = TeacherLiveCanvasDeltaPage(
        book_type="ms",
        page_number=2,
        copy_id="online-meeting-1",
        stroke_count=3,
    )

    payload = _build_teacher_live_delta_page_payload("meeting-1", page)

    assert payload["copy_id"] == "online-meeting-1"
    assert payload["book_type"] == "MS"
    assert _decode_monitoring_page_key(payload["page_key"]) == {
        "copy_id": "online-meeting-1",
        "book_type": "MS",
        "page_number": 2,
    }

    bad_page = TeacherLiveCanvasDeltaPage(
        book_type="MS",
        page_number=2,
        copy_id="default",
    )
    with pytest.raises(ValueError, match="Live canvas deltas must use this class copy scope"):
        _build_teacher_live_delta_page_payload("meeting-1", bad_page)

    request = TeacherLiveCanvasDeltaRequest(
        client_batch_id="batch-1",
        page=page,
        strokes=[],
        sent_at=1234,
    )
    assert request.client_batch_id == "batch-1"


@pytest.mark.asyncio
async def test_live_canvas_hub_broadcasts_and_replays_deltas():
    from api.v1.online_class.router import teacher_live_canvas_hub

    class FakeWebSocket:
        def __init__(self):
            self.messages = []

        async def send_json(self, payload):
            self.messages.append(payload)

    meeting_id = f"test-live-canvas-{uuid.uuid4()}"
    student_socket = FakeWebSocket()

    latest_seq = await teacher_live_canvas_hub.add(meeting_id, "student", student_socket)
    assert latest_seq == 0

    event = await teacher_live_canvas_hub.publish(
        meeting_id,
        {
            "type": "teacher_canvas_delta",
            "page": {
                "page_key": "online-test:MS:0",
                "copy_id": "online-test",
                "book_type": "MS",
                "page_number": 0,
                "stroke_count": 1,
            },
            "strokes": [{"id": "stroke-1", "points": [[1, 2, 0.5]]}],
            "sent_at": 1234,
        },
    )

    assert event["seq"] == 1
    assert student_socket.messages == [event]

    replay = await teacher_live_canvas_hub.replay_after(meeting_id, 0)
    assert replay == [event]
    assert await teacher_live_canvas_hub.replay_after(meeting_id, 1) == []

    await teacher_live_canvas_hub.remove(meeting_id, student_socket)


@pytest.mark.asyncio
async def test_live_canvas_redis_publish_stores_replay_and_global_sequence():
    from api.v1.online_class.router import (
        _get_teacher_live_canvas_redis_latest_seq,
        _publish_teacher_live_canvas_redis,
        _replay_teacher_live_canvas_redis,
        _teacher_live_canvas_redis_channel,
    )

    class FakePipeline:
        def __init__(self, redis):
            self.redis = redis
            self.ops = []

        def lpush(self, key, value):
            self.ops.append(("lpush", key, value))
            return self

        def ltrim(self, key, start, end):
            self.ops.append(("ltrim", key, start, end))
            return self

        def expire(self, key, ttl):
            self.ops.append(("expire", key, ttl))
            return self

        def publish(self, channel, payload):
            self.ops.append(("publish", channel, payload))
            return self

        async def execute(self):
            for op in self.ops:
                if op[0] == "lpush":
                    _, key, value = op
                    self.redis.lists.setdefault(key, []).insert(0, value)
                elif op[0] == "ltrim":
                    _, key, start, end = op
                    self.redis.lists[key] = self.redis.lists.get(key, [])[start:end + 1]
                elif op[0] == "expire":
                    continue
                elif op[0] == "publish":
                    _, channel, payload = op
                    self.redis.published.append((channel, payload))

    class FakeRedis:
        def __init__(self):
            self.values = {}
            self.lists = {}
            self.published = []

        async def incr(self, key):
            self.values[key] = int(self.values.get(key, 0)) + 1
            return self.values[key]

        async def get(self, key):
            value = self.values.get(key)
            return str(value).encode("utf-8") if value is not None else None

        async def lrange(self, key, start, end):
            return self.lists.get(key, [])[start:end + 1]

        def pipeline(self):
            return FakePipeline(self)

    redis = FakeRedis()
    meeting_id = f"test-live-canvas-redis-{uuid.uuid4()}"
    payload = {
        "type": "teacher_canvas_delta",
        "page": {
            "page_key": "online-test:MS:0",
            "copy_id": "online-test",
            "book_type": "MS",
            "page_number": 0,
            "stroke_count": 1,
        },
        "strokes": [{"id": "stroke-redis", "points": [[1, 2, 0.5]]}],
        "sent_at": 1234,
    }

    event = await _publish_teacher_live_canvas_redis(redis, meeting_id, payload)

    assert event["seq"] == 1
    assert event["meeting_id"] == meeting_id
    assert await _get_teacher_live_canvas_redis_latest_seq(redis, meeting_id) == 1
    assert redis.published[0][0] == _teacher_live_canvas_redis_channel(meeting_id)

    replay = await _replay_teacher_live_canvas_redis(redis, meeting_id, 0)
    assert replay == [event]
    assert await _replay_teacher_live_canvas_redis(redis, meeting_id, 1) == []
