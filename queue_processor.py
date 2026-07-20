import asyncio
import gc
import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("facturas-bot")

MAX_QUEUE_SIZE = 20
COOLDOWN = 2.0
TIMEOUT = 90


@dataclass
class FacturaJob:
    user_id: int = 0
    username: str = ""
    image_bytes: bytes = b""
    media_type: str = "image/jpeg"
    update: Any = None
    status_msg: Any = None
    context: Any = None


class FacturaQueue:
    def __init__(self):
        self._queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        self._processing = set()
        self._started = False
        self._stats = {"ok": 0, "err": 0}

    @property
    def stats(self):
        return {**self._stats, "en_cola": self._queue.qsize(), "procesando": len(self._processing)}

    async def start(self):
        if self._started:
            return
        self._started = True
        asyncio.create_task(self._worker())
        log.info("Cola de facturas iniciada")

    async def add(self, job):
        if job.user_id in self._processing:
            return False, "Ya tienes una factura en proceso. Espera."
        if self._queue.full():
            return False, "Cola llena. Intenta luego."
        await self._queue.put(job)
        pos = self._queue.qsize()
        if pos > 1:
            return True, f"En cola (posicion {pos})."
        return True, ""

    async def _worker(self):
        while True:
            job = await self._queue.get()
            self._processing.add(job.user_id)
            try:
                await asyncio.wait_for(self._process(job), timeout=TIMEOUT)
                self._stats["ok"] += 1
            except asyncio.TimeoutError:
                self._stats["err"] += 1
                try:
                    await job.status_msg.edit_text("Timeout. Intenta de nuevo.")
                except Exception:
                    pass
            except Exception as e:
                self._stats["err"] += 1
                log.error(f"Worker error: {e}", exc_info=True)
                try:
                    await job.status_msg.edit_text(f"Error: {e}")
                except Exception:
                    pass
            finally:
                self._processing.discard(job.user_id)
                job.image_bytes = None
                gc.collect()
                self._queue.task_done()
                await asyncio.sleep(COOLDOWN)

    async def _process(self, job):
        from bot import process_queued_invoice
        await process_queued_invoice(job)


cola = FacturaQueue()
