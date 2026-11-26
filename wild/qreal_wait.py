"""Wait for the Modal call started by modal_qreal.py::launch.

Usage: qreal_wait.py CALL_ID [MAX_HOURS]
Exits 0 when the call has finished, 1 if it failed or MAX_HOURS (default 3)
passed.
"""

import sys
import time

import modal


def main():
    call = modal.FunctionCall.from_id(sys.argv[1])
    limit = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    start = time.time()
    while time.time() - start < limit * 3600:
        try:
            call.get(timeout=0)
            print("call finished", flush=True)
            return 0
        except TimeoutError:
            time.sleep(60)
        except Exception as e:  # The call itself failed (after its retries).
            print("call failed:", repr(e), flush=True)
            return 1
    print("gave up waiting", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
