import os, time, queue, threading
print(os.environ["ENVIRONMENT"])
import os

def print_tree(root="/app", max_depth=3):
    root = os.path.abspath(root)
    root_depth = root.count(os.sep)

    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.count(os.sep) - root_depth
        if depth > max_depth:
            dirnames[:] = []  # stop descending
            continue

        indent = "    " * depth
        print(f"{indent}{os.path.basename(dirpath)}/")

        if depth < max_depth:
            for name in filenames:
                print(f"{indent}    {name}")

print_tree("/app", max_depth=3)
if os.environ["ENVIRONMENT"] == "production":
    SWIM_SFDPS_PATH = "/app/sfdps"
    SWIM_TFMS_PATH = "/app/tfms"
else:
    SWIM_SFDPS_PATH = "/Users/jonahgoode/Documents/PlaneQuery/Code/jumpstart-latest/log"
    SWIM_TFMS_PATH = "/Users/jonahgoode/Documents/PlaneQuery/Code/jumpstart-latest/tfms_log"
MESSAGES_LOG_PATH = os.path.join(SWIM_SFDPS_PATH, "messages.log")

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
    """Read and assemble complete XML documents without data loss"""
    buffer = ""
    
    for line in follow(MESSAGES_LOG_PATH):
        buffer += line  # Keep original spacing/newlines
        
        # Look for complete documents
        while True:
            start = buffer.find("<?xml")
            if start == -1:
                break
                
            end = buffer.find("</ns5:MessageCollection>", start)
            if end == -1:
                break
                
            # Extract complete document
            end += len("</ns5:MessageCollection>")
            doc = buffer[start:end]
            q.put(doc)
            buffer = buffer[end:]  # Remove processed document

def worker_thread(q: queue.Queue):
    while True:
        xml_doc = q.get()
        # do your slow processing here
        process(xml_doc)
        q.task_done()

from read_messages import parse_message

def process(xml_doc: str):
    # placeholder
    parse_message(xml_doc)

if os.environ["ENVIRONMENT"] == "production":
    from s3_watcher import start_s3_uploader
    print("Running in production mode")
    # Start S3 uploader to watch for .gz files and upload them (SFDPS)
    s3_observer_sfdps = start_s3_uploader(
        log_path=SWIM_SFDPS_PATH,
        bucket="planequery-swim-logs-prod",
        prefix="sfdps-logs"
    )
    # Start S3 uploader for TFMS logs
    s3_observer_tfms = start_s3_uploader(
        log_path=SWIM_TFMS_PATH,
        bucket="planequery-swim-logs-prod",
        prefix="tfms-logs"
    )

q = queue.Queue(maxsize=100_000)  # tune for memory
threading.Thread(target=reader_thread, args=(q,), daemon=True).start()
threading.Thread(target=worker_thread, args=(q,), daemon=True).start()

q.join()  # optional; keeps main alive if you want

try:
    while True:
        print(f"Queue size: {q.qsize()}, Unfinished: {q.unfinished_tasks}")
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down...")
    s3_observer_sfdps.stop()
    s3_observer_sfdps.join()
    s3_observer_tfms.stop()
    s3_observer_tfms.join()
