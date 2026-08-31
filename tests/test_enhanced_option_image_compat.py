from __future__ import annotations

import pytest


class _ImageDatabase:
    async def mongo_find_one(self, collection_name, query):
        assert collection_name == "images"
        assert query == {"_id": "stored-option-image"}
        return {
            "_id": "stored-option-image",
            "content_type": "image/png",
            "base64Data": "iVBORw0KGgo=",
        }


@pytest.mark.asyncio
async def test_durable_option_is_projected_as_data_image_for_released_clients():
    from utils.enhanced_option_images import enrich_enhanced_option_images

    stored_options = [
        {
            "id": "stored-option-image",
            "image_id": "stored-option-image",
            "label": "A",
            "type": "image",
            "content": "/api/v1/images/stored-option-image",
            "url": "/api/v1/images/stored-option-image",
        }
    ]

    projected = await enrich_enhanced_option_images(
        stored_options,
        db=_ImageDatabase(),
        is_b2c=False,
    )

    assert projected[0]["content"] == "data:image/png;base64,iVBORw0KGgo="
    assert projected[0]["image_id"] == "stored-option-image"
    assert stored_options[0]["content"] == "/api/v1/images/stored-option-image"
