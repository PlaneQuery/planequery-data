import os, time, queue, threading

SWIM_LOG_PATH = "/Users/jonahgoode/Documents/PlaneQuery/Other-Code/jumpstart-latest/log"
MESSAGES_LOG_PATH = SWIM_LOG_PATH + "/" + "messages.log"

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
    """Read lines and assemble complete XML documents"""
    current_doc = []

    for line in follow(MESSAGES_LOG_PATH):
        line = line.strip()
        
        # Start of new XML document
        if line.startswith("<?xml"):
            # If we have a previous document, send it to the queue
            if current_doc:
                q.put("\n".join(current_doc))
            current_doc = [line]
        else:
            # Continue building current document
            current_doc.append(line)
            
            # Check if document is complete (ends with closing tag)
            if line.endswith("</ns5:MessageCollection>"):
                q.put("\n".join(current_doc))
                current_doc = []

def worker_thread(q: queue.Queue):
    while True:
        xml_doc = q.get()
        try:
            # do your slow processing here
            process(xml_doc)
        finally:
            q.task_done()

from read_messages import parse_message
def process(xml_doc: str):
    # placeholder
    parse_message(xml_doc)

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