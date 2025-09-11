#!/usr/bin/env python3
"""
Log Aggregator Service
Demonstrates log collection, aggregation, and analysis from multiple services.
Shows how to work with volume-mounted log files and temporary storage.
"""

import os
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, Counter
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/log_aggregator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LogAggregator:
    """Log aggregation and analysis service"""
    
    def __init__(self):
        self.log_dir = Path(os.getenv('LOG_DIR', '/app/logs'))
        self.aggregation_interval = int(os.getenv('AGGREGATION_INTERVAL', '60'))
        self.temp_dir = Path('/tmp/log_analysis')
        
        # Create temp directory
        self.temp_dir.mkdir(exist_ok=True)
        
        # Log patterns for parsing
        self.log_pattern = re.compile(
            r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - '
            r'(?P<logger>\S+) - '
            r'(?P<level>\w+) - '
            r'(?P<message>.*)'
        )
        
        logger.info(f"LogAggregator initialized")
        logger.info(f"Log directory: {self.log_dir}")
        logger.info(f"Aggregation interval: {self.aggregation_interval} seconds")
        logger.info(f"Temp directory: {self.temp_dir}")
    
    def _parse_log_line(self, line):
        """Parse a single log line"""
        match = self.log_pattern.match(line.strip())
        if match:
            return {
                'timestamp': match.group('timestamp'),
                'logger': match.group('logger'),
                'level': match.group('level'),
                'message': match.group('message')
            }
        return None
    
    def _read_log_file(self, filepath):
        """Read and parse a log file"""
        logs = []
        try:
            with open(filepath, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    parsed = self._parse_log_line(line)
                    if parsed:
                        parsed['file'] = filepath.name
                        parsed['line'] = line_num
                        logs.append(parsed)
        except Exception as e:
            logger.error(f"Error reading log file {filepath}: {e}")
        
        return logs
    
    def collect_logs(self):
        """Collect logs from all log files"""
        all_logs = []
        
        if not self.log_dir.exists():
            logger.warning(f"Log directory does not exist: {self.log_dir}")
            return all_logs
        
        log_files = list(self.log_dir.glob('*.log'))
        logger.info(f"Found {len(log_files)} log files")
        
        for log_file in log_files:
            logger.info(f"Processing log file: {log_file.name}")
            logs = self._read_log_file(log_file)
            all_logs.extend(logs)
            logger.info(f"  Parsed {len(logs)} log entries")
        
        return all_logs
    
    def analyze_logs(self, logs):
        """Analyze collected logs"""
        if not logs:
            return {}
        
        # Time range
        timestamps = [log['timestamp'] for log in logs]
        time_range = {
            'start': min(timestamps),
            'end': max(timestamps),
            'total_entries': len(logs)
        }
        
        # Level distribution
        level_counts = Counter(log['level'] for log in logs)
        
        # Logger distribution
        logger_counts = Counter(log['logger'] for log in logs)
        
        # File distribution
        file_counts = Counter(log['file'] for log in logs)
        
        # Error analysis
        error_logs = [log for log in logs if log['level'] in ['ERROR', 'CRITICAL']]
        error_analysis = {
            'total_errors': len(error_logs),
            'error_rate': len(error_logs) / len(logs) * 100,
            'error_messages': Counter(log['message'] for log in error_logs)
        }
        
        # Recent activity (last hour)
        recent_cutoff = datetime.now() - timedelta(hours=1)
        recent_logs = [
            log for log in logs 
            if datetime.strptime(log['timestamp'], '%Y-%m-%d %H:%M:%S,%f') > recent_cutoff
        ]
        
        analysis = {
            'time_range': time_range,
            'level_distribution': dict(level_counts),
            'logger_distribution': dict(logger_counts),
            'file_distribution': dict(file_counts),
            'error_analysis': error_analysis,
            'recent_activity': {
                'entries_last_hour': len(recent_logs),
                'recent_levels': dict(Counter(log['level'] for log in recent_logs))
            }
        }
        
        return analysis
    
    def save_analysis(self, analysis, timestamp=None):
        """Save analysis results to temp directory"""
        if timestamp is None:
            timestamp = datetime.now()
        
        filename = f"log_analysis_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.temp_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            logger.info(f"Analysis saved to: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving analysis: {e}")
            return None
    
    def cleanup_temp_files(self, max_age_hours=24):
        """Clean up old temporary analysis files"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            removed_count = 0
            
            for filepath in self.temp_dir.glob('*.json'):
                file_time = datetime.fromtimestamp(filepath.stat().st_mtime)
                if file_time < cutoff_time:
                    filepath.unlink()
                    removed_count += 1
                    logger.info(f"Removed old analysis file: {filepath.name}")
            
            if removed_count > 0:
                logger.info(f"Cleanup completed: {removed_count} files removed")
            else:
                logger.info("No old files to clean up")
            
            return removed_count
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0
    
    def get_aggregation_status(self):
        """Get aggregation service status"""
        log_files = list(self.log_dir.glob('*.log')) if self.log_dir.exists() else []
        temp_files = list(self.temp_dir.glob('*.json'))
        
        return {
            'service_status': 'running',
            'log_directory': str(self.log_dir),
            'temp_directory': str(self.temp_dir),
            'aggregation_interval': self.aggregation_interval,
            'log_files_found': len(log_files),
            'analysis_files': len(temp_files),
            'log_files': [f.name for f in log_files],
            'analysis_files': [f.name for f in temp_files]
        }

def main():
    """Main function"""
    logger.info("Starting Log Aggregator Service")
    
    aggregator = LogAggregator()
    
    try:
        # Initial analysis
        logger.info("Performing initial log analysis")
        logs = aggregator.collect_logs()
        analysis = aggregator.analyze_logs(logs)
        aggregator.save_analysis(analysis)
        
        # Main service loop
        while True:
            time.sleep(aggregator.aggregation_interval)
            
            # Collect and analyze logs
            logger.info("Performing scheduled log analysis")
            logs = aggregator.collect_logs()
            analysis = aggregator.analyze_logs(logs)
            
            # Save analysis
            aggregator.save_analysis(analysis)
            
            # Log summary
            logger.info(f"Analysis complete:")
            logger.info(f"  Total log entries: {analysis.get('time_range', {}).get('total_entries', 0)}")
            logger.info(f"  Error rate: {analysis.get('error_analysis', {}).get('error_rate', 0):.2f}%")
            logger.info(f"  Recent activity: {analysis.get('recent_activity', {}).get('entries_last_hour', 0)} entries")
            
            # Cleanup old files
            aggregator.cleanup_temp_files()
            
            # Log status
            status = aggregator.get_aggregation_status()
            logger.info(f"Aggregator status: {status}")
    
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        logger.info("Log Aggregator Service stopped")

if __name__ == '__main__':
    main()
