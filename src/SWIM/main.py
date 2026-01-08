import os, time, queue, threading

LOG = "./log/messages.log"

def follow(path: str, poll=0.2):
    while True:
        try:
            f = open(path, "r", encoding="utf-8", errors="replace")
            break
        except FileNotFoundError:
            time.sleep(poll)

    f.seek(0, os.SEEK_END)
    ino = os.fstat(f.fileno()).st_ino

    while True:
        line = f.readline()
        if line:
            yield line
            continue

        time.sleep(poll)
        try:
            if os.stat(path).st_ino != ino:
                f.close()
                f = open(path, "r", encoding="utf-8", errors="replace")
                ino = os.fstat(f.fileno()).st_ino
        except FileNotFoundError:
            pass

def reader_thread(q: queue.Queue):
    for line in follow(LOG):
        q.put(line)  # blocks if full => no drops

def worker_thread(q: queue.Queue):
    while True:
        line = q.get()
        try:
            # do your slow processing here
            process(line)
        finally:
            q.task_done()

def process(line: str):
    # placeholder
    print(line)

q = queue.Queue(maxsize=100_000)  # tune for memory
threading.Thread(target=reader_thread, args=(q,), daemon=True).start()
threading.Thread(target=worker_thread, args=(q,), daemon=True).start()

q.join()  # optional; keeps main alive if you want
