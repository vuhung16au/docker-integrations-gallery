#!/usr/bin/env python3
"""
Data Processor Service
Demonstrates volume management and data persistence concepts.
Processes files from input directory and saves results to output directory.
"""

import os
import time
import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/data_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class DataProcessor:
    def __init__(self):
        self.input_dir = Path(os.getenv('INPUT_DIR', '/app/data/input'))
        self.output_dir = Path(os.getenv('OUTPUT_DIR', '/app/data/output'))
        self.backup_dir = Path(os.getenv('BACKUP_DIR', '/app/data/backup'))
        
        # Create directories if they don't exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.processed_count = 0
        self.start_time = datetime.now()
        
        logger.info(f"DataProcessor initialized")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Backup directory: {self.backup_dir}")

    def generate_sample_data(self, filename="sample_data.csv", rows=100):
        """Generate sample data for demonstration"""
        logger.info(f"Generating sample data: {filename}")
        
        # Create sample dataset
        data = {
            'id': range(1, rows + 1),
            'timestamp': [datetime.now().isoformat()] * rows,
            'value': np.random.normal(100, 15, rows),
            'category': np.random.choice(['A', 'B', 'C'], rows),
            'processed': [False] * rows
        }
        
        df = pd.DataFrame(data)
        filepath = self.input_dir / filename
        df.to_csv(filepath, index=False)
        
        logger.info(f"Generated {rows} rows of sample data in {filepath}")
        return filepath

    def process_file(self, filepath):
        """Process a single file and save results"""
        try:
            logger.info(f"Processing file: {filepath}")
            
            # Read the file
            if filepath.suffix == '.csv':
                df = pd.read_csv(filepath)
            else:
                logger.warning(f"Unsupported file type: {filepath.suffix}")
                return False
            
            # Process the data
            df['processed'] = True
            df['processed_at'] = datetime.now().isoformat()
            df['value_squared'] = df['value'] ** 2
            df['category_encoded'] = pd.Categorical(df['category']).codes
            
            # Generate summary statistics
            summary = {
                'file': filepath.name,
                'rows': len(df),
                'columns': len(df.columns),
                'mean_value': float(df['value'].mean()),
                'std_value': float(df['value'].std()),
                'categories': df['category'].value_counts().to_dict(),
                'processed_at': datetime.now().isoformat()
            }
            
            # Save processed data
            output_file = self.output_dir / f"processed_{filepath.name}"
            df.to_csv(output_file, index=False)
            
            # Save summary
            summary_file = self.output_dir / f"summary_{filepath.stem}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.processed_count += 1
            logger.info(f"Successfully processed {filepath.name}")
            logger.info(f"Output saved to: {output_file}")
            logger.info(f"Summary saved to: {summary_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing {filepath}: {str(e)}")
            return False

    def process_all_files(self):
        """Process all files in the input directory"""
        logger.info("Starting batch processing of all files")
        
        files = list(self.input_dir.glob('*.csv'))
        if not files:
            logger.info("No CSV files found in input directory")
            return 0
        
        processed = 0
        for filepath in files:
            if self.process_file(filepath):
                processed += 1
        
        logger.info(f"Batch processing complete: {processed}/{len(files)} files processed")
        return processed

    def get_status(self):
        """Get current processing status"""
        input_files = len(list(self.input_dir.glob('*.csv')))
        output_files = len(list(self.output_dir.glob('*.csv')))
        summary_files = len(list(self.output_dir.glob('*.json')))
        
        uptime = datetime.now() - self.start_time
        
        return {
            'status': 'running',
            'uptime_seconds': int(uptime.total_seconds()),
            'processed_files': self.processed_count,
            'input_files': input_files,
            'output_files': output_files,
            'summary_files': summary_files,
            'directories': {
                'input': str(self.input_dir),
                'output': str(self.output_dir),
                'backup': str(self.backup_dir)
            }
        }

# Initialize processor
processor = DataProcessor()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/status', methods=['GET'])
def get_status():
    """Get processing status"""
    return jsonify(processor.get_status())

@app.route('/generate-data', methods=['POST'])
def generate_data():
    """Generate sample data"""
    try:
        data = request.get_json() or {}
        filename = data.get('filename', 'sample_data.csv')
        rows = data.get('rows', 100)
        
        filepath = processor.generate_sample_data(filename, rows)
        return jsonify({
            'message': 'Sample data generated successfully',
            'file': str(filepath),
            'rows': rows
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process', methods=['POST'])
def process_files():
    """Process files in input directory"""
    try:
        data = request.get_json() or {}
        filename = data.get('filename')
        
        if filename:
            # Process specific file
            filepath = processor.input_dir / filename
            if filepath.exists():
                success = processor.process_file(filepath)
                return jsonify({
                    'message': 'File processed successfully' if success else 'File processing failed',
                    'file': filename,
                    'success': success
                })
            else:
                return jsonify({'error': f'File not found: {filename}'}), 404
        else:
            # Process all files
            processed = processor.process_all_files()
            return jsonify({
                'message': f'Processed {processed} files',
                'processed_count': processed
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/files', methods=['GET'])
def list_files():
    """List files in input and output directories"""
    try:
        input_files = [f.name for f in processor.input_dir.glob('*') if f.is_file()]
        output_files = [f.name for f in processor.output_dir.glob('*') if f.is_file()]
        
        return jsonify({
            'input_files': input_files,
            'output_files': output_files,
            'count': {
                'input': len(input_files),
                'output': len(output_files)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/volumes', methods=['GET'])
def volume_info():
    """Get volume information"""
    try:
        import shutil
        
        def get_dir_size(path):
            total = 0
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total += os.path.getsize(filepath)
            return total
        
        return jsonify({
            'volumes': {
                'input': {
                    'path': str(processor.input_dir),
                    'size_bytes': get_dir_size(processor.input_dir),
                    'files': len(list(processor.input_dir.glob('*')))
                },
                'output': {
                    'path': str(processor.output_dir),
                    'size_bytes': get_dir_size(processor.output_dir),
                    'files': len(list(processor.output_dir.glob('*')))
                },
                'backup': {
                    'path': str(processor.backup_dir),
                    'size_bytes': get_dir_size(processor.backup_dir),
                    'files': len(list(processor.backup_dir.glob('*')))
                }
            },
            'disk_usage': {
                'total': shutil.disk_usage('/app').total,
                'used': shutil.disk_usage('/app').used,
                'free': shutil.disk_usage('/app').free
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Generate initial sample data
    processor.generate_sample_data()
    
    # Start the Flask application
    port = int(os.getenv('PORT', 8080))
    logger.info(f"Starting Data Processor service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
