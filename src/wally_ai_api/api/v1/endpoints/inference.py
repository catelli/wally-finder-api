from __future__ import annotations

import base64
from typing import Literal

from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import Response

from wally_ai_api.api.v1.schemas.inference import (
    AnnotatedImageSchema,
    BoundingBoxSchema,
    DetectionSchema,
    InferenceResponseSchema,
)
from wally_ai_api.app.dependencies import get_detect_wally_use_case
from wally_ai_api.application.dto.inference_dto import InferenceResultDto
from wally_ai_api.application.use_cases.detect_wally import DetectWallyUseCase
from wally_ai_api.domain.entities.image import ImageInput

router = APIRouter(prefix="/inference", tags=["inference"])


def _to_json_response(result: InferenceResultDto) -> InferenceResponseSchema:
    return InferenceResponseSchema(
        request_id=result.request_id,
        detection_count=len(result.detections),
        detections=[
            DetectionSchema(
                class_id=item.class_id,
                label=item.label,
                confidence=item.confidence,
                bbox=BoundingBoxSchema(
                    x1=item.bbox.x1,
                    y1=item.bbox.y1,
                    x2=item.bbox.x2,
                    y2=item.bbox.y2,
                ),
            )
            for item in result.detections
        ],
        annotated_image=AnnotatedImageSchema(
            content_type=result.content_type,
            data_base64=base64.b64encode(result.annotated_image).decode("ascii"),
        ),
    )


@router.post("", response_model=None)
async def run_inference(
    file: UploadFile = File(...),
    response_format: Literal["json", "image"] = Query(default="json"),
    use_case: DetectWallyUseCase = Depends(get_detect_wally_use_case),
) -> InferenceResponseSchema | Response:
    content = await file.read()
    image = ImageInput(
        content=content,
        content_type=file.content_type or "application/octet-stream",
        filename=file.filename or "upload.jpg",
    )
    result = use_case.execute(image)

    if response_format == "image":
        return Response(
            content=result.annotated_image,
            media_type=result.content_type,
            headers={
                "X-Request-Id": result.request_id,
                "X-Detection-Count": str(len(result.detections)),
            },
        )

    return _to_json_response(result)
