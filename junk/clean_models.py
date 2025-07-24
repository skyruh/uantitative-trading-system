#!/usr/bin/env python3
"""
Script to clean previously trained models to prepare for GPU training.
"""

import os
import shutil
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ModelCleaner")

def main():
    """Clean previously trained models."""
    models_dir = "models"
    
    # Create backup directory
    backup_dir = os.path.join(models_dir, "backup_cpu_models")
    os.makedirs(backup_dir, exist_ok=True)
    
    # Get list of model directories
    model_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d)) and d != "checkpoints" and d != "backup_cpu_models"]
    
    logger.info(f"Found {len(model_dirs)} model directories to clean")
    
    # Move each model directory to backup
    for model_dir in model_dirs:
        src_path = os.path.join(models_dir, model_dir)
        dst_path = os.path.join(backup_dir, model_dir)
        
        try:
            shutil.move(src_path, dst_path)
            logger.info(f"Moved {model_dir} to backup")
        except Exception as e:
            logger.error(f"Error moving {model_dir}: {str(e)}")
    
    # Reset system state to indicate models need to be trained
    system_state_path = "data/system_state.json"
    if os.path.exists(system_state_path):
        import json
        with open(system_state_path, 'r') as f:
            system_state = json.load(f)
        
        system_state['system_state']['models_trained'] = False
        
        with open(system_state_path, 'w') as f:
            json.dump(system_state, f, indent=2)
        
        logger.info("Updated system state: models_trained = False")
    
    logger.info("Model cleaning completed")

if __name__ == "__main__":
    main()