import os
import time

SWIM_LOG_PATH = os.environ["SWIM_LOG_PATH"]
MESSAGES_LOG_PATH = SWIM_LOG_PATH + "/" + "messages.log"

def follow(path: str):
    # open (wait until it exists)
    while True:
        try:
            f = open(path, "r", encoding="utf-8", errors="replace")
            break
        except FileNotFoundError:
            time.sleep(0.2)

    # start at end
    f.seek(0, os.SEEK_END)
    ino = os.fstat(f.fileno()).st_ino

    while True:
        line = f.readline()
        if line:
            yield line
            continue

        time.sleep(0.2)

        # if rollover happened (messages.log replaced), reopen
        try:
            if os.stat(path).st_ino != ino:
                f.close()
                f = open(path, "r", encoding="utf-8", errors="replace")
                ino = os.fstat(f.fileno()).st_ino
        except FileNotFoundError:
            # brief gap during rollover
            pass

for line in follow(MESSAGES_LOG_PATH):
    print(line, end="")
