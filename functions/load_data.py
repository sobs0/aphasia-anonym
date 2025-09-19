"""
Data Loading Functions
"""

import os
import csv
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional


class DataLoader:
    """Handles loading and organizing CHAT and wav files"""
    
    def __init__(self, config: dict, logger: logging.Logger):
        """Initialize DataLoader with configuration."""
        self.config = config
        self.logger = logger
        
        # Extract configuration
        self.input_dir = config['data_input']['base_data_dir']
        self.audio_subdir = config['data_input']['audio_subdir']
        self.exclude_dirs = config['data_input']['exclude_dirs']
        
        # Output configuration
        self.temp_dir = config['output']['temporary_data_dir']
        self.load_data_headers = config['output']['load_data_headers']
        
        # Filtering configuration
        self.filter_mode = config['data_filtering']['filter_mode']
        self.specific_speakers = config['data_filtering']['specific_speakers']
        self.specific_recordings = config['data_filtering']['specific_recordings']
        self.include_dirs = config['data_filtering']['include_dirs']
    
    def normalize_patient_id(self, rel_path: str) -> Tuple[str, str, str]:
        """
        Normalize patient ID and recording ID from relative path
        """
        rel_path = rel_path.strip("/")
        parts = rel_path.split("/")
        filename = parts[-1].replace(".cha", "")
        folder = parts[-2] if len(parts) > 1 else ""

        if "-" in filename:
            patient_raw = filename.split("-")[0]
            patient_id = folder.lower().split("-")[0] + patient_raw
            recording_id = folder.lower().split("-")[0] + filename
        else:
            patient_id = filename[:-1]
            recording_id = filename

        return patient_id.lower(), recording_id.lower(), filename.lower()
    
    def load_existing_recording_ids(self, csv_path: Path) -> Set[str]:
        """
        Load existing recording IDs to avoid duplicates
        """
        existing_ids = set()
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                if 'recording_id' in df.columns:
                    existing_ids = set(df['recording_id'].tolist())
                    self.logger.info(f"Loaded {len(existing_ids)} existing recording IDs")
            except Exception as e:
                self.logger.warning(f"Could not load existing recording IDs: {e}")
        
        return existing_ids
    
    def should_include_directory(self, rel_root: str) -> bool:
        """
        Check if directory should be included based on filtering configuration
        """
        # Always exclude specified directories
        if any(excluded in rel_root for excluded in self.exclude_dirs):
            return False
        
        # Apply filtering based on filter_mode
        if self.filter_mode == "all":
            return True
        elif self.filter_mode == "include_dirs":
            return any(included in rel_root for included in self.include_dirs)
        else:
            # For speaker/recording specific filtering, includes all directories
            # Filtering will be done at the file level
            return True
    
    def should_include_file(self, patient_id: str, recording_id: str) -> bool:
        """
        Check if file should be included based on filtering configuration
        """
        if self.filter_mode == "specific_speakers":
            return patient_id in self.specific_speakers
        elif self.filter_mode == "specific_recordings":
            return recording_id in self.specific_recordings
        else:
            return True
    
    def find_matching_wav_files(self, root_dir: str) -> Dict[str, str]:
        """
        Find wav files in the audio subdirectory
        """
        wav_files = {}
        audio_dir = os.path.join(root_dir, self.audio_subdir)
        
        if os.path.isdir(audio_dir):
            for wav_file in os.listdir(audio_dir):
                if wav_file.endswith(".wav"):
                    wav_name = wav_file.replace(".wav", "").lower()
                    wav_files[wav_name] = os.path.join(audio_dir, wav_file)
        
        return wav_files
    
    def load_all_data(self) -> str:
        """
        Load all .cha and .wav files and create temporary output file
        """
        self.logger.info("=== Step 1: Loading all .cha and .wav files ===")
        
        # Create temporary directory
        temp_dir = Path(self.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Output file path
        csv_path = temp_dir / "output_load_data.csv"
        
        chat_wav_entries = []
        
        # Load existing recording IDs to avoid duplicates
        existing_ids = self.load_existing_recording_ids(csv_path)
        
        files_processed = 0
        files_skipped = 0
        duplicates_found = 0
        
        # Walk through the input directory
        for root, dirs, files in os.walk(self.input_dir):
            rel_root = os.path.relpath(root, self.input_dir)
            
            # Check if directory should be included
            if not self.should_include_directory(rel_root):
                continue
            
            # Find .cha files in current directory
            cha_files = [f for f in files if f.endswith(".cha")]
            if not cha_files:
                continue
            
            # Find corresponding WAV files
            wav_files = self.find_matching_wav_files(root)
            
            # Process each .cha file
            for cha_file in cha_files:
                rel_cha_path = os.path.join(rel_root, cha_file)
                patient_id, recording_id, file_name = self.normalize_patient_id(rel_cha_path)
                
                # Apply file-level filtering
                if not self.should_include_file(patient_id, recording_id):
                    files_skipped += 1
                    continue
                
                # Check for duplicates
                if recording_id in existing_ids:
                    self.logger.warning(f"Duplicate recording_id detected: {recording_id} (file: {file_name})")
                    duplicates_found += 1
                    continue
                
                # Get paths
                cha_path_abs = os.path.join(root, cha_file)
                wav_path_abs = wav_files.get(file_name.lower(), "")
                
                # Warn if wav file not found
                if not wav_path_abs:
                    self.logger.warning(f"No matching WAV file found for: {file_name}")
                
                # Add entry
                chat_wav_entries.append([
                    patient_id,
                    recording_id,
                    file_name,
                    cha_path_abs,
                    wav_path_abs
                ])
                
                files_processed += 1
        
        # Save to CSV
        self.save_temp_csv(chat_wav_entries, csv_path)
        
        # Log statistics
        self.logger.info(f"Data loading completed:")
        self.logger.info(f"  - Files processed: {files_processed}")
        self.logger.info(f"  - Files skipped (filtering): {files_skipped}")
        self.logger.info(f"  - Duplicates found: {duplicates_found}")
        self.logger.info(f"  - Temporary output saved to: {csv_path}")
        
        return str(csv_path)
    
    def save_temp_csv(self, entries: List[List[str]], csv_path: Path):
        """
        Save entries to temporary CSV file
        """
        # Determine if we need to write header
        write_header = not csv_path.exists()
        
        # Create directory if it doesn't exist
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write/append entries to CSV
        mode = "w" if write_header else "a"
        with open(csv_path, mode, newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(self.load_data_headers)
            writer.writerows(entries)
    
    def create_placeholder_csvs(self):
        """
        Create placeholder final CSV files if not existing
        """
        final_dir = Path(self.config['output']['final_data_dir'])
        final_dir.mkdir(parents=True, exist_ok=True)
        
        # Create final index.csv placeholder
        index_path = final_dir / self.config['output']['final_index_filename']
        if not index_path.exists():
            with open(index_path, "w", newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.config['output']['final_index_headers'])
            self.logger.info(f"Created placeholder index file: {index_path}")
        
        # Create final metadata.csv placeholder
        metadata_path = final_dir / self.config['output']['final_metadata_filename']
        if not metadata_path.exists():
            with open(metadata_path, "w", newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.config['output']['final_metadata_headers'])
            self.logger.info(f"Created placeholder metadata file: {metadata_path}")


def load_and_organize_data(config: dict, logger: logging.Logger) -> str:
    """
    Main function to load and organize all data files
    """
    data_loader = DataLoader(config, logger)
    
    # Load all data and create temporary CSV
    csv_path = data_loader.load_all_data()
    
    # Create placeholder files for final output
    data_loader.create_placeholder_csvs()
    
    return csv_path