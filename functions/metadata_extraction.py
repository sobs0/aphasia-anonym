"""
Metadata Extraction Functions
"""

import os
import csv
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Set, Tuple, Any
from collections import defaultdict


class MetadataExtractor:
    """Extracting and processing speaker metadata from CHAT files"""
    
    def __init__(self, config: dict, logger: logging.Logger):
        """Initialize MetadataExtractor with configuration"""
        self.config = config
        self.logger = logger
        
        # Output configuration
        self.temp_dir = config['output']['temporary_data_dir']
        
        # Metadata extraction configuration
        self.target_id_pattern = config['metadata_extraction']['target_id_pattern']
        self.gender_conflict_resolution = config['metadata_extraction']['gender_conflict_resolution']
        self.age_conflict_resolution = config['metadata_extraction']['age_conflict_resolution']
        self.wab_type_conflict_resolution = config['metadata_extraction']['wab_type_conflict_resolution']
        self.wab_score_conflict_resolution = config['metadata_extraction']['wab_score_conflict_resolution']
        self.default_wab_type = config['metadata_extraction']['default_wab_type']
        
        # Setup metadata extraction logging
        self.setup_metadata_logging()
    
    def setup_metadata_logging(self):
        """Setup separate logging for metadata extraction conflicts."""
        self.metadata_logger = logging.getLogger('metadata_extraction')
        self.metadata_logger.setLevel(logging.INFO)
        
        # Create metadata extraction log file
        temp_dir = Path(self.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = temp_dir / 'metadata_extraction_conflicts.log'
        handler = logging.FileHandler(log_file, mode='w') 
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Clear any existing handlers
        self.metadata_logger.handlers.clear()
        self.metadata_logger.addHandler(handler)
        
        self.logger.info(f"Metadata extraction conflicts will be logged to: {log_file}")
    
    def parse_id_line(self, line: str) -> Tuple[str, str, str, str]:
        """
        Parse metadata from @ID line in CHAT file
        """
        try:
            # Remove @ID: prefix and split by |
            fields = line.strip().replace('@ID:', '').strip('|').split('|')
            
            # Extract fields according to CHAT format
            age_field = fields[3] if len(fields) > 3 else ""
            age_years = age_field.split(';')[0] if ';' in age_field else age_field
            age = age_years if age_years.isdigit() else ""
            
            gender = fields[4] if len(fields) > 4 else ""
            wab_type = fields[5] if len(fields) > 5 else ""
            wab_aq = fields[9] if len(fields) > 9 and fields[9] else ""
            
            return age, gender, wab_type, wab_aq
            
        except Exception as e:
            self.logger.warning(f"Failed to parse @ID line: {line.strip()}, Error: {e}")
            return "", "", "", ""
    
    def extract_metadata_from_file(self, cha_file_path: str, speaker_id: str) -> Dict[str, str]:
        """
        Extract metadata from a single .cha file
        """
        metadata = {"age": "", "gender": "", "WAB_type": "", "WAB_score": ""}
        
        if not os.path.isfile(cha_file_path):
            self.logger.warning(f"File not found: {cha_file_path}")
            return metadata
        
        try:
            with open(cha_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    # Look for @ID lines containing PAR
                    if line.startswith("@ID:") and "|PAR|" in line:
                        age, gender, wab_type, wab_aq = self.parse_id_line(line)
                        
                        # Only update if we got a value
                        if age:
                            metadata["age"] = age
                        if gender:
                            metadata["gender"] = gender
                        if wab_type:
                            metadata["WAB_type"] = wab_type
                        if wab_aq:
                            metadata["WAB_score"] = wab_aq
                        
                        break
                        
        except Exception as e:
            self.logger.error(f"Error reading file {cha_file_path}: {e}")
        
        return metadata
    
    def collect_all_metadata_per_speaker(self, input_csv_path: str) -> Dict[str, Dict[str, List[str]]]:
        """
        Collect all metadata from all recordings per speaker
        """
        self.logger.info("Collecting metadata from all recordings per speaker...")
        
        # Load input CSV
        df = pd.read_csv(input_csv_path)
        
        # Step 1: Create list of unique speakers
        unique_speakers = df['patient_id'].unique()
        self.logger.info(f"Found {len(unique_speakers)} unique speakers")
        
        # Step 2: Initialize speaker metadata dictionaries
        speaker_metadata = {}
        for speaker_id in unique_speakers:
            speaker_metadata[speaker_id] = {
                "age": [],
                "gender": [],
                "WAB_type": [],
                "WAB_score": []
            }
        
        # Step 3: Process all recordings per speaker
        processed_files = 0
        for _, row in df.iterrows():
            speaker_id = row['patient_id']
            cha_file_path = row['CHAT_file_path']
            
            # Extract metadata from this recording
            file_metadata = self.extract_metadata_from_file(cha_file_path, speaker_id)
            
            # Add unique values to speaker's metadata lists
            for key, value in file_metadata.items():
                if value and value not in speaker_metadata[speaker_id][key]:
                    speaker_metadata[speaker_id][key].append(value)
            
            processed_files += 1
            
            if processed_files % 100 == 0:
                self.logger.info(f"Processed {processed_files} files...")
        
        self.logger.info(f"Collected metadata from {processed_files} files")
        return speaker_metadata
    
    def resolve_metadata_conflicts(self, speaker_metadata: Dict[str, Dict[str, List[str]]]) -> List[List[str]]:
        """
        Resolve conflicts when multiple values exist per speaker
        """
        self.logger.info("Resolving metadata conflicts...")
        
        resolved_entries = []
        conflicts_logged = 0
        
        for speaker_id, metadata in speaker_metadata.items():
            resolved_entry = [speaker_id] 
            
            # Process each metadata field
            for field in ["gender", "age", "WAB_type", "WAB_score"]:
                values = metadata[field]
                resolved_value = ""
                
                if len(values) == 0:
                    # No values found
                    if field == "WAB_type":
                        resolved_value = self.default_wab_type
                    else:
                        resolved_value = ""
                        
                elif len(values) == 1:
                    # Single value, use it
                    resolved_value = values[0]
                    
                else:
                    # Multiple values - resolve conflict
                    conflicts_logged += 1
                    
                    if field == "gender":
                        resolved_value = self.gender_conflict_resolution
                        self.metadata_logger.info(f"Speaker {speaker_id}: Multiple genders {values} -> {resolved_value}")
                        
                    elif field == "age":
                        if self.age_conflict_resolution == "max":
                            try:
                                resolved_value = str(max(int(v) for v in values if v.isdigit()))
                            except ValueError:
                                resolved_value = values[0] 
                        else:
                            resolved_value = values[0]
                        self.metadata_logger.info(f"Speaker {speaker_id}: Multiple ages {values} -> {resolved_value}")
                        
                    elif field == "WAB_type":
                        resolved_value = self.wab_type_conflict_resolution
                        self.metadata_logger.info(f"Speaker {speaker_id}: Multiple WAB types {values} -> {resolved_value}")
                        
                    elif field == "WAB_score":
                        if self.wab_score_conflict_resolution == "min":
                            try:
                                # Convert to float for comparison, keep as string
                                min_score = min(float(v) for v in values if v and v.replace('.','').isdigit())
                                resolved_value = str(min_score)
                            except ValueError:
                                resolved_value = values[0] 
                        else:
                            resolved_value = values[0] 
                        self.metadata_logger.info(f"Speaker {speaker_id}: Multiple WAB scores {values} -> {resolved_value}")
                
                resolved_entry.append(resolved_value)
            
            resolved_entries.append(resolved_entry)
        
        self.logger.info(f"Resolved metadata for {len(resolved_entries)} speakers")
        self.logger.info(f"Conflicts logged: {conflicts_logged}")
        
        return resolved_entries
    
    def extract_speaker_metadata(self, input_csv_path: str) -> str:
        """
        Main function to extract speaker metadata
        """
        self.logger.info("=== Step 2: Extracting speaker metadata ===")
        
        # Create output path
        temp_dir = Path(self.temp_dir)
        output_path = temp_dir / "output_metadata_extraction.csv"
        
        # Step 1-3: Collect all metadata per speaker
        speaker_metadata = self.collect_all_metadata_per_speaker(input_csv_path)
        
        # Step 4: Resolve conflicts and create final entries
        resolved_entries = self.resolve_metadata_conflicts(speaker_metadata)
        
        # Save to temporary CSV
        headers = ["speaker_id", "gender", "age", "WAB_type", "WAB_score"]
        
        with open(output_path, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(resolved_entries)
        
        self.logger.info(f"Metadata extraction completed")
        self.logger.info(f"Temporary output saved to: {output_path}")
        
        return str(output_path)


def extract_speaker_metadata(config: dict, logger: logging.Logger, input_csv_path: str) -> str:
    """
    Main function to extract speaker metadata
    """
    extractor = MetadataExtractor(config, logger)
    return extractor.extract_speaker_metadata(input_csv_path)