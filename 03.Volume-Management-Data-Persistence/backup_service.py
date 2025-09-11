#!/usr/bin/env python3
"""
Backup Service
Demonstrates automated backup strategies and data persistence.
Creates backups of processed data and manages backup retention.
"""

import os
import time
import shutil
import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/backup_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BackupService:
    """Automated backup service for data persistence"""
    
    def __init__(self):
        self.source_dir = Path(os.getenv('SOURCE_DIR', '/app/data/output'))
        self.backup_dir = Path(os.getenv('BACKUP_DIR', '/app/data/backup'))
        self.backup_interval = int(os.getenv('BACKUP_INTERVAL', '300'))  # 5 minutes default
        self.retention_days = int(os.getenv('RETENTION_DAYS', '7'))
        
        # Create backup directory if it doesn't exist
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup metadata file
        self.metadata_file = self.backup_dir / 'backup_metadata.json'
        self.metadata = self._load_metadata()
        
        logger.info(f"BackupService initialized")
        logger.info(f"Source directory: {self.source_dir}")
        logger.info(f"Backup directory: {self.backup_dir}")
        logger.info(f"Backup interval: {self.backup_interval} seconds")
        logger.info(f"Retention period: {self.retention_days} days")
    
    def _load_metadata(self):
        """Load backup metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        
        return {
            'backups': [],
            'last_backup': None,
            'total_backups': 0,
            'total_size': 0
        }
    
    def _save_metadata(self):
        """Save backup metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _calculate_file_hash(self, filepath):
        """Calculate MD5 hash of a file"""
        hash_md5 = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {filepath}: {e}")
            return None
    
    def _get_backup_name(self, timestamp=None):
        """Generate backup name with timestamp"""
        if timestamp is None:
            timestamp = datetime.now()
        return f"backup_{timestamp.strftime('%Y%m%d_%H%M%S')}"
    
    def create_backup(self):
        """Create a new backup"""
        try:
            if not self.source_dir.exists():
                logger.warning(f"Source directory does not exist: {self.source_dir}")
                return False
            
            # Check if there are files to backup
            source_files = list(self.source_dir.glob('*'))
            if not source_files:
                logger.info("No files to backup")
                return False
            
            # Create backup directory
            backup_timestamp = datetime.now()
            backup_name = self._get_backup_name(backup_timestamp)
            backup_path = self.backup_dir / backup_name
            backup_path.mkdir(exist_ok=True)
            
            logger.info(f"Creating backup: {backup_name}")
            
            # Copy files and calculate total size
            total_size = 0
            files_backed_up = []
            
            for source_file in source_files:
                if source_file.is_file():
                    dest_file = backup_path / source_file.name
                    shutil.copy2(source_file, dest_file)
                    
                    file_size = source_file.stat().st_size
                    total_size += file_size
                    
                    file_hash = self._calculate_file_hash(source_file)
                    
                    files_backed_up.append({
                        'name': source_file.name,
                        'size': file_size,
                        'hash': file_hash,
                        'modified': datetime.fromtimestamp(source_file.stat().st_mtime).isoformat()
                    })
                    
                    logger.info(f"  Backed up: {source_file.name} ({file_size} bytes)")
            
            # Update metadata
            backup_info = {
                'name': backup_name,
                'timestamp': backup_timestamp.isoformat(),
                'files': files_backed_up,
                'total_size': total_size,
                'file_count': len(files_backed_up)
            }
            
            self.metadata['backups'].append(backup_info)
            self.metadata['last_backup'] = backup_timestamp.isoformat()
            self.metadata['total_backups'] += 1
            self.metadata['total_size'] += total_size
            
            self._save_metadata()
            
            logger.info(f"Backup completed: {backup_name}")
            logger.info(f"  Files: {len(files_backed_up)}")
            logger.info(f"  Size: {total_size} bytes")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return False
    
    def cleanup_old_backups(self):
        """Remove backups older than retention period"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            backups_to_remove = []
            
            for backup_info in self.metadata['backups']:
                backup_date = datetime.fromisoformat(backup_info['timestamp'])
                if backup_date < cutoff_date:
                    backups_to_remove.append(backup_info)
            
            if not backups_to_remove:
                logger.info("No old backups to clean up")
                return 0
            
            logger.info(f"Cleaning up {len(backups_to_remove)} old backups")
            
            removed_count = 0
            for backup_info in backups_to_remove:
                backup_path = self.backup_dir / backup_info['name']
                if backup_path.exists():
                    shutil.rmtree(backup_path)
                    logger.info(f"  Removed: {backup_info['name']}")
                    removed_count += 1
                
                # Remove from metadata
                self.metadata['backups'].remove(backup_info)
                self.metadata['total_size'] -= backup_info['total_size']
            
            self.metadata['total_backups'] -= removed_count
            self._save_metadata()
            
            logger.info(f"Cleanup completed: {removed_count} backups removed")
            return removed_count
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0
    
    def get_backup_status(self):
        """Get backup service status"""
        return {
            'service_status': 'running',
            'source_directory': str(self.source_dir),
            'backup_directory': str(self.backup_dir),
            'backup_interval': self.backup_interval,
            'retention_days': self.retention_days,
            'total_backups': self.metadata['total_backups'],
            'total_size': self.metadata['total_size'],
            'last_backup': self.metadata['last_backup'],
            'available_backups': len(self.metadata['backups'])
        }
    
    def list_backups(self):
        """List all available backups"""
        return {
            'backups': self.metadata['backups'],
            'count': len(self.metadata['backups'])
        }
    
    def restore_backup(self, backup_name):
        """Restore a specific backup"""
        try:
            backup_path = self.backup_dir / backup_name
            if not backup_path.exists():
                logger.error(f"Backup not found: {backup_name}")
                return False
            
            logger.info(f"Restoring backup: {backup_name}")
            
            # Clear source directory
            if self.source_dir.exists():
                for file in self.source_dir.glob('*'):
                    if file.is_file():
                        file.unlink()
            
            # Copy files from backup
            restored_files = []
            for backup_file in backup_path.glob('*'):
                if backup_file.is_file():
                    dest_file = self.source_dir / backup_file.name
                    shutil.copy2(backup_file, dest_file)
                    restored_files.append(backup_file.name)
                    logger.info(f"  Restored: {backup_file.name}")
            
            logger.info(f"Restore completed: {len(restored_files)} files restored")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring backup: {e}")
            return False

def main():
    """Main function"""
    logger.info("Starting Backup Service")
    
    backup_service = BackupService()
    
    try:
        # Create initial backup
        backup_service.create_backup()
        
        # Main service loop
        while True:
            time.sleep(backup_service.backup_interval)
            
            # Create new backup
            backup_service.create_backup()
            
            # Cleanup old backups
            backup_service.cleanup_old_backups()
            
            # Log status
            status = backup_service.get_backup_status()
            logger.info(f"Backup service status: {status}")
    
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        logger.info("Backup Service stopped")

if __name__ == '__main__':
    main()
