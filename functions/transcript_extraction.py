import re
import pandas as pd
from pathlib import Path


def extract_timestamps_transcripts(config, logger, input_csv_path):
    """
    Extract timestamps and original transcripts from CHAT files
    """
    logger.info("Starting transcript and timestamp extraction from .cha files")
    
    # Load filtered data
    load_data_df = pd.read_csv(input_csv_path)
    logger.info(f"Processing {len(load_data_df)} recordings from filtered data")
    
    # Prepare output - save directly to temporary directory
    temp_dir = Path(config['output']['temporary_data_dir'])
    chunks_filename = config['processing']['chunks_data_filename']
    chunks_csv_path = temp_dir / chunks_filename
    
    # Initialize results
    chunks_data = []
    total_chunks = 0
    processed_recordings = 0
    failed_recordings = 0
    
    for _, row in load_data_df.iterrows():
        recording_id = row['recording_id']
        cha_path = row['CHAT_file_path']
        patient_id = row['patient_id']
        
        logger.info(f"Processing recording: {recording_id}")
        
        try:
            # Read .cha file
            with open(cha_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            logger.info(f"File {recording_id}: Total lines in .cha file: {len(lines)}")
            
            chunk_counter = 1
            par_chunks_found = 0
            star_lines_found = 0
            par_lines_without_timestamps = 0
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                
                # Only process speaker lines (starting with *)
                if not line.startswith("*"):
                    continue
                
                star_lines_found += 1
                
                # Debug: Show first few * lines for inspection
                if star_lines_found <= 5:
                    logger.info(f"Sample * line {star_lines_found}: {line}")
                
                # Extract speaker, text, and timestamps using regex
                match = re.match(r"^\*(\w+):\s*(.*?)\s+(\d+)_(\d+)", line)
                
                # Test specifically on PAR lines
                if line.startswith("*PAR:") and star_lines_found <= 5:
                    logger.info(f"DEBUG PAR line: '{line}'")
                    logger.info(f"DEBUG PAR line length: {len(line)}")
                    logger.info(f"DEBUG PAR line repr: {repr(line)}")
                    if match:
                        logger.info(f"DEBUG PAR MATCHED: {match.groups()}")
                    else:
                        logger.info("DEBUG PAR NOT MATCHED")
                        # Try a simpler pattern
                        simple_match = re.search(r"(\d+)_(\d+)", line)
                        if simple_match:
                            logger.info(f"DEBUG Found timestamp in line: {simple_match.groups()}")
                        else:
                            logger.info("DEBUG No timestamp pattern found at all")
                
                if not match:
                    # Still increment counter for any * line, even if no timestamps
                    chunk_counter += 1
                    continue
                
                speaker, text, start_ts, end_ts = match.groups()
                
                # Create chunk ID
                chunk_id = f"{recording_id}_{chunk_counter:04d}"
                
                # Only keep PAR lines
                if speaker == "PAR":
                    chunks_data.append({
                        'speaker_id': patient_id,
                        'recording_id': recording_id,
                        'chunk_id': chunk_id,
                        'cha_file_path': cha_path,
                        'start_time': int(start_ts),
                        'end_time': int(end_ts),
                        'original_transcript': text.strip()
                    })
                    par_chunks_found += 1
                    total_chunks += 1
                
                chunk_counter += 1
            
            logger.info(f"Recording {recording_id} summary:")
            logger.info(f"  - Total * lines found: {star_lines_found}")
            logger.info(f"  - PAR lines without timestamps: {par_lines_without_timestamps}")
            logger.info(f"  - PAR chunks with timestamps: {par_chunks_found}")
            processed_recordings += 1
            
        except FileNotFoundError:
            logger.error(f"File not found: {cha_path}")
            failed_recordings += 1
            continue
        except Exception as e:
            logger.error(f"Error processing {cha_path}: {str(e)}")
            failed_recordings += 1
            continue
    
    # Create DataFrame and save
    if chunks_data:
        chunks_df = pd.DataFrame(chunks_data)
        chunks_df.to_csv(chunks_csv_path, index=False)
        
        logger.info(f"Transcript extraction completed:")
        logger.info(f"  - Processed recordings: {processed_recordings}")
        logger.info(f"  - Failed recordings: {failed_recordings}")
        logger.info(f"  - Total PAR chunks extracted: {total_chunks}")
        logger.info(f"  - Average chunks per recording: {total_chunks/processed_recordings:.1f}")
        logger.info(f"Chunks data saved to: {chunks_csv_path}")
        
    else:
        logger.warning("No chunks data extracted!")
        # Create empty CSV with headers
        empty_df = pd.DataFrame(columns=[
            'speaker_id', 'recording_id', 'chunk_id', 'cha_file_path', 
            'start_time', 'end_time', 'original_transcript'
        ])
        empty_df.to_csv(chunks_csv_path, index=False)
    
    return str(chunks_csv_path)