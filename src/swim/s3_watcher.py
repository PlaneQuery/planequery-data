import os
import time
import threading
import boto3
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent

s3_client = boto3.client("s3")


class GzFileHandler(FileSystemEventHandler):
    """Watch for new .gz files and upload them to S3"""
    
    def __init__(self, base_path: str, bucket: str, prefix: str = ""):
        self.base_path = Path(base_path)
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.pending_files = set()  # Track files we've seen but haven't uploaded
        self.lock = threading.Lock()
    
    def on_created(self, event):
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Only process .gz files
        if not file_path.suffix == ".gz":
            return
        
        with self.lock:
            self.pending_files.add(file_path)
        
        # Wait a moment for the file to be fully written
        self._wait_for_file_complete(file_path)
        
        # Upload to S3
        self._upload_to_s3(file_path)
        
        with self.lock:
            self.pending_files.discard(file_path)
    
    def on_deleted(self, event):
        """Detect if a file was deleted before we uploaded it"""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        if not file_path.suffix == ".gz":
            return
        
        with self.lock:
            if file_path in self.pending_files:
                print(f"ERROR: File deleted before upload completed: {file_path}")
                print("FALLING BEHIND - files are being deleted faster than we can upload them")
                os._exit(1)  # Hard exit
    
    def _wait_for_file_complete(self, file_path: Path, timeout: float = 30.0):
        """Wait for file to stop being written to"""
        last_size = -1
        stable_count = 0
        start = time.time()
        
        while time.time() - start < timeout:
            try:
                current_size = file_path.stat().st_size
                if current_size == last_size:
                    stable_count += 1
                    if stable_count >= 3:  # File size stable for 1.5 seconds
                        return
                else:
                    stable_count = 0
                    last_size = current_size
            except FileNotFoundError:
                # File was deleted while we were waiting
                print(f"ERROR: File deleted during upload wait: {file_path}")
                print("FALLING BEHIND - files are being deleted faster than we can upload them")
                os._exit(1)
            time.sleep(0.5)
    
    def _upload_to_s3(self, file_path: Path):
        """Upload file to S3 preserving the partition structure"""
        # Get relative path from base (e.g., "year=2026/month=01/day=13/messages-1.log.gz")
        relative_path = file_path.relative_to(self.base_path)
        
        # Build S3 key with optional prefix
        if self.prefix:
            s3_key = f"{self.prefix}/{relative_path}"
        else:
            s3_key = str(relative_path)
        
        print(f"Uploading {file_path} -> s3://{self.bucket}/{s3_key}")
        
        try:
            s3_client.upload_file(
                str(file_path),
                self.bucket,
                s3_key,
                ExtraArgs={
                    "ContentType": "application/gzip",
                    "ContentEncoding": "gzip",
                }
            )
            print(f"Successfully uploaded: s3://{self.bucket}/{s3_key}")
        except FileNotFoundError:
            # File was deleted during upload
            print(f"ERROR: File deleted during S3 upload: {file_path}")
            print("FALLING BEHIND - files are being deleted faster than we can upload them")
            os._exit(1)
            
def start_s3_uploader(log_path, bucket, prefix):
    """Start watching for new .gz files and uploading to S3"""
    handler = GzFileHandler(log_path, bucket, prefix)
    observer = Observer()
    observer.schedule(handler, log_path, recursive=True)
    observer.start()
    print(f"Watching {log_path} for new .gz files...")
    return observer
