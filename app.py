
import base64
import json
import math
import threading
import time
import uuid
from collections import defaultdict, deque
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import JSONResponse, Response


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TOTAL_ORDERS = 50
RATE_LIMIT = 19
WINDOW_SECONDS = 10

app = FastAPI(title="Orders API Engineering")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Retry-After"],
)


# --------------------------------------------------
# In-memory stores
# --------------------------------------------------

# Idempotency key -> original response.
idempotency_store: dict[str, dict[str, Any]] = {}

# Client ID -> timestamps of requests within the window.
client_requests: dict[str, deque[float]] = defaultdict(deque)

# Protect shared dictionaries and queues.
store_lock = threading.Lock()


# --------------------------------------------------
# Request model
# --------------------------------------------------

class OrderRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    item: str = "order"
    quantity: int = Field(default=1, ge=1)


# --------------------------------------------------
# Per-client rate limiting
# --------------------------------------------------

@app.middleware("http")
async def per_client_rate_limit(
    request: Request,
    call_next,
):
    # CORS preflight must not consume rate-limit capacity.
    if request.method == "OPTIONS":
        return await call_next(request)

    # Do not put all unidentified clients into one shared bucket.
    client_id = request.headers.get("X-Client-Id")

    if client_id is None or not client_id.strip():
        return await call_next(request)

    client_id = client_id.strip()
    now = time.monotonic()

    with store_lock:
        timestamps = client_requests[client_id]

        # Remove requests outside the sliding time window.
        while timestamps and now - timestamps[0] >= WINDOW_SECONDS:
            timestamps.popleft()

        if len(timestamps) >= RATE_LIMIT:
            retry_after = max(
                1,
                math.ceil(
                    WINDOW_SECONDS - (now - timestamps[0])
                ),
            )

            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": str(retry_after)},
            )

        timestamps.append(now)

    return await call_next(request)


# --------------------------------------------------
# Cursor encoding and decoding
# --------------------------------------------------

def encode_cursor(last_id: int) -> str:
    payload = json.dumps(
        {"after": last_id},
        separators=(",", ":"),
    ).encode("utf-8")

    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def decode_cursor(cursor: str) -> int:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = base64.urlsafe_b64decode(padded.encode("ascii"))
        data = json.loads(payload.decode("utf-8"))

        last_id = data["after"]

        if isinstance(last_id, bool) or not isinstance(last_id, int):
            raise ValueError("Invalid cursor position")

        if last_id < 0 or last_id > TOTAL_ORDERS:
            raise ValueError("Cursor position out of range")

        return last_id

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid cursor",
        )


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "orders-api-engineering",
        "total_orders": TOTAL_ORDERS,
        "rate_limit": RATE_LIMIT,
        "window_seconds": WINDOW_SECONDS,
    }


# --------------------------------------------------
# Create order with idempotency
# --------------------------------------------------

@app.post("/orders", status_code=201)
async def create_order(
    order: OrderRequest,
    idempotency_key: Optional[str] = Header(
        default=None,
        alias="Idempotency-Key",
    ),
):
    if idempotency_key is None or not idempotency_key.strip():
        raise HTTPException(
            status_code=400,
            detail="Idempotency-Key header is required",
        )

    key = idempotency_key.strip()
    order_data = order.model_dump()

    with store_lock:
        # A repeated key returns the exact original response.
        if key in idempotency_store:
            return JSONResponse(
                status_code=201,
                content=idempotency_store[key],
            )

        result = {
            "id": str(uuid.uuid4()),
            **order_data,
        }

        idempotency_store[key] = result

    return JSONResponse(
        status_code=201,
        content=result,
    )


# --------------------------------------------------
# Cursor-paginated fixed catalog, IDs 1 through 50
# --------------------------------------------------

@app.get("/orders")
async def list_orders(
    limit: int = Query(default=10, ge=1),
    cursor: Optional[str] = Query(default=None),
):
    if cursor:
        last_id = decode_cursor(cursor)
    else:
        last_id = 0

    # The fixed catalog consists of IDs 1 through 50.
    start_id = last_id + 1
    end_id = min(last_id + limit, TOTAL_ORDERS)

    items = (
        [{"id": order_id} for order_id in range(start_id, end_id + 1)]
        if start_id <= TOTAL_ORDERS
        else []
    )

    if items and end_id < TOTAL_ORDERS:
        next_cursor = encode_cursor(end_id)
    else:
        next_cursor = None

    return {
        "items": items,
        "next_cursor": next_cursor,
    }
