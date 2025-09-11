#!/usr/bin/env python3
"""
File Watcher Service
Monitors the input directory for new files and logs file system events.
Demonstrates volume monitoring and event-driven processing.
"""

import os
import time
import logging
from datetime import datetime
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/file_watcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FileEventHandler(FileSystemEventHandler):
    """Handle file system events"""
    
    def __init__(self, watch_dir):
        self.watch_dir = Path(watch_dir)
        self.event_count = 0
        logger.info(f"FileEventHandler initialized for directory: {self.watch_dir}")
    
    def on_created(self, event):
        """Handle file creation events"""
        if not event.is_directory:
            self.event_count += 1
            logger.info(f"File created: {event.src_path}")
            self._log_file_info(event.src_path, "created")
    
    def on_modified(self, event):
        """Handle file modification events"""
        if not event.is_directory:
            self.event_count += 1
            logger.info(f"File modified: {event.src_path}")
            self._log_file_info(event.src_path, "modified")
    
    def on_deleted(self, event):
        """Handle file deletion events"""
        if not event.is_directory:
            self.event_count += 1
            logger.info(f"File deleted: {event.src_path}")
            self._log_file_info(event.src_path, "deleted")
    
    def on_moved(self, event):
        """Handle file move/rename events"""
        if not event.is_directory:
            self.event_count += 1
            logger.info(f"File moved: {event.src_path} -> {event.dest_path}")
            self._log_file_info(event.dest_path, "moved")
    
    def _log_file_info(self, filepath, action):
        """Log detailed file information"""
        try:
            path = Path(filepath)
            if path.exists():
                stat = path.stat()
                logger.info(f"  Action: {action}")
                logger.info(f"  File: {path.name}")
                logger.info(f"  Size: {stat.st_size} bytes")
                logger.info(f"  Modified: {datetime.fromtimestamp(stat.st_mtime)}")
                logger.info(f"  Total events: {self.event_count}")
            else:
                logger.info(f"  Action: {action}")
                logger.info(f"  File: {path.name} (no longer exists)")
        except Exception as e:
            logger.error(f"Error getting file info for {filepath}: {e}")

class FileWatcher:
    """File system watcher service"""
    
    def __init__(self, watch_dir='/app/data/input'):
        self.watch_dir = Path(watch_dir)
        self.observer = Observer()
        self.event_handler = FileEventHandler(watch_dir)
        self.running = False
        
        # Ensure watch directory exists
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"FileWatcher initialized")
        logger.info(f"Watching directory: {self.watch_dir}")
    
    def start(self):
        """Start watching the directory"""
        try:
            self.observer.schedule(self.event_handler, str(self.watch_dir), recursive=False)
            self.observer.start()
            self.running = True
            logger.info("File watcher started successfully")
            
            # Log initial directory contents
            self._log_directory_contents()
            
        except Exception as e:
            logger.error(f"Failed to start file watcher: {e}")
            raise
    
    def stop(self):
        """Stop watching the directory"""
        if self.running:
            self.observer.stop()
            self.observer.join()
            self.running = False
            logger.info("File watcher stopped")
    
    def _log_directory_contents(self):
        """Log current directory contents"""
        try:
            files = list(self.watch_dir.glob('*'))
            logger.info(f"Initial directory contents ({len(files)} files):")
            for filepath in files:
                if filepath.is_file():
                    stat = filepath.stat()
                    logger.info(f"  {filepath.name} ({stat.st_size} bytes)")
        except Exception as e:
            logger.error(f"Error listing directory contents: {e}")
    
    def get_status(self):
        """Get watcher status"""
        return {
            'running': self.running,
            'watch_directory': str(self.watch_dir),
            'events_detected': self.event_handler.event_count,
            'files_in_directory': len(list(self.watch_dir.glob('*')))
        }

def main():
    """Main function"""
    watch_dir = os.getenv('WATCH_DIR', '/app/data/input')
    
    logger.info("Starting File Watcher Service")
    logger.info(f"Watch directory: {watch_dir}")
    
    watcher = FileWatcher(watch_dir)
    
    try:
        watcher.start()
        
        # Keep the service running
        while True:
            time.sleep(10)
            
            # Log status every minute
            if int(time.time()) % 60 == 0:
                status = watcher.get_status()
                logger.info(f"Watcher status: {status}")
    
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        watcher.stop()
        logger.info("File Watcher Service stopped")

if __name__ == '__main__':
    main()
