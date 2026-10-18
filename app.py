import base64
import json
import math
import time
import uuid
from collections import defaultdict, deque
from threading import Lock
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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

# In-memory state for idempotency and per-client rate limits.
idempotency_store: dict[str, dict[str, Any]] = {}
client_requests = defaultdict(deque)
state_lock = Lock()


class OrderRequest(BaseModel):
    model_config = {"extra": "allow"}
    item: str = "order"
    quantity: int = Field(default=1, ge=1)


def encode_cursor(last_id: int) -> str:
    raw = json.dumps({"after": last_id}, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_cursor(cursor: str) -> int:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded).decode())
        after = payload["after"]
        if not isinstance(after, int) or not 0 <= after <= TOTAL_ORDERS:
            raise ValueError
        return after
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid cursor")


@app.middleware("http")
async def per_client_rate_limit(request: Request, call_next):
    # CORS preflight must be allowed through without consuming a request.
    if request.method == "OPTIONS":
        return await call_next(request)

    client_id = request.headers.get("X-Client-Id", "anonymous").strip()
    if not client_id:
        client_id = "anonymous"

    now = time.monotonic()
    with state_lock:
        bucket = client_requests[client_id]
        while bucket and now - bucket[0] >= WINDOW_SECONDS:
            bucket.popleft()

        if len(bucket) >= RATE_LIMIT:
            retry_after = max(
                1, math.ceil(WINDOW_SECONDS - (now - bucket[0]))
            )
            return Response(
                content=json.dumps({"detail": "Rate limit exceeded"}),
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": str(retry_after)},
            )

        bucket.append(now)

    return await call_next(request)


@app.get("/")
def home():
    return {
        "status": "ok",
        "service": "orders-api-engineering",
        "total_orders": TOTAL_ORDERS,
        "rate_limit": RATE_LIMIT,
        "window_seconds": WINDOW_SECONDS,
    }


@app.post("/orders", status_code=201)
def create_order(
    order: OrderRequest,
    idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=1),
):
    key = idempotency_key.strip()
    if not key:
        raise HTTPException(status_code=422, detail="Idempotency-Key is required")

    with state_lock:
        if key in idempotency_store:
            return idempotency_store[key]

        result = {
            "id": str(uuid.uuid4()),
            **order.model_dump(),
        }
        idempotency_store[key] = result
        return result


@app.get("/orders")
def list_orders(limit: int = 10, cursor: str | None = None):
    if not 1 <= limit <= TOTAL_ORDERS:
        raise HTTPException(
            status_code=422,
            detail=f"limit must be between 1 and {TOTAL_ORDERS}",
        )

    after = decode_cursor(cursor) if cursor else 0
    ids = list(range(after + 1, min(after + limit, TOTAL_ORDERS) + 1))
    items = [{"id": order_id} for order_id in ids]

    next_cursor = encode_cursor(ids[-1]) if ids and ids[-1] < TOTAL_ORDERS else None

    return {
        "items": items,
        "next_cursor": next_cursor,
    }
