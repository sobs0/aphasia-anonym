import os
import pandas as pd
import soundfile as sf
from pathlib import Path


def create_audio_chunks(config, logger, chunks_csv_path, load_data_csv_path):
    """
    Cut original .wav audio & create audio chunks.
    """
    logger.info("Starting audio chunk creation from original .wav files")
    
    # Create audio chunks directory
    temp_dir = Path(config['output']['temporary_data_dir'])
    audio_chunks_dir = temp_dir / 'audio_chunks'
    audio_chunks_dir.mkdir(parents=True, exist_ok=True)
    
    # Output file path
    chunks_with_audio_filename = config['processing']['chunks_with_audio_filename']
    output_path = temp_dir / chunks_with_audio_filename
    
    # Load input chunks
    chunks_df_input = pd.read_csv(chunks_csv_path)
    
    # Check what's already processed
    if output_path.exists():
        logger.info(f"Found existing chunks file: {output_path}")
        existing_df = pd.read_csv(output_path)
        already_processed_chunk_ids = set(existing_df['chunk_id'].unique())
        logger.info(f"Already processed chunks: {len(already_processed_chunk_ids)}")
        
        # Filter input to only new chunks that aren't processed yet
        new_chunks_df = chunks_df_input[~chunks_df_input['chunk_id'].isin(already_processed_chunk_ids)].copy()
        
        if new_chunks_df.empty:
            logger.info("No new chunks to process - all chunks already exist!")
            return str(output_path)
        
        logger.info(f"Found {len(new_chunks_df)} new chunks to process")
        
        # Use existing as base
        chunks_df = existing_df.copy()
        
    else:
        logger.info("No existing chunks file found, processing all chunks")
        new_chunks_df = chunks_df_input.copy()
        already_processed_chunk_ids = set()
        
        # Initialize empty result dataframe
        chunks_df = pd.DataFrame(columns=list(chunks_df_input.columns) + ['chunk_wav_path', 'audio_length'])
    
    # Only process new chunks
    if not new_chunks_df.empty:
        # Initialize new columns for new chunks
        new_chunks_df['chunk_wav_path'] = None
        new_chunks_df['audio_length'] = None
        
        # Load wav file paths mapping
        load_data_df = pd.read_csv(load_data_csv_path)
        wav_path_mapping = dict(zip(load_data_df['recording_id'], load_data_df['wav_file_path']))
        
        logger.info(f"Processing {len(new_chunks_df)} new chunks...")
        
        successful_chunks = 0
        failed_chunks = 0
        
        for idx, row in new_chunks_df.iterrows():
            recording_id = row['recording_id']
            chunk_id = row['chunk_id']
            start_time = row['start_time']
            end_time = row['end_time']
            
            # Get original wav file path
            if recording_id not in wav_path_mapping:
                logger.error(f"WAV file path not found for recording_id: {recording_id}")
                failed_chunks += 1
                continue
                
            original_wav_path = wav_path_mapping[recording_id]
            
            # Create chunk filename and path
            chunk_filename = f"{chunk_id}.wav"
            chunk_wav_path = audio_chunks_dir / chunk_filename
            
            try:
                # Extract audio chunk
                duration = extract_audio_chunk(
                    wav_file=original_wav_path,
                    start_ms=start_time,
                    end_ms=end_time,
                    output_path=str(chunk_wav_path)
                )
                
                # Update dataframe with chunk info
                new_chunks_df.at[idx, 'chunk_wav_path'] = str(chunk_wav_path)
                new_chunks_df.at[idx, 'audio_length'] = duration
                
                successful_chunks += 1
                
                if successful_chunks % 100 == 0:
                    logger.info(f"Processed {successful_chunks} new chunks...")
                    
            except Exception as e:
                logger.error(f"Failed to extract chunk {chunk_id}: {str(e)}")
                failed_chunks += 1
                continue
        
        # Combine existing + new processed chunks
        chunks_df = pd.concat([chunks_df, new_chunks_df], ignore_index=True)
        
        logger.info(f"Audio chunk creation completed:")
        logger.info(f"  - Successfully created: {successful_chunks}")
        logger.info(f"  - Failed: {failed_chunks}")
        
    else:
        successful_chunks = 0
        failed_chunks = 0
        logger.info("No new chunks to process")
    
    # Save updated chunks data
    chunks_df.to_csv(output_path, index=False)
    
    total_chunks = len(chunks_df)
    skipped_chunks = len(already_processed_chunk_ids)
    
    logger.info(f"Final statistics:")
    logger.info(f"  - Total chunks in file: {total_chunks}")
    logger.info(f"  - Previously processed: {skipped_chunks}")
    logger.info(f"  - Newly processed: {successful_chunks}")
    logger.info(f"  - Audio chunks directory: {audio_chunks_dir}")
    logger.info(f"Updated chunks data saved to: {output_path}")
    
    return str(output_path)


def extract_audio_chunk(wav_file, start_ms, end_ms, output_path):
    """
    Extract audio chunk from wav file using timestamps
    """
    # Read original audio file
    signal, sample_rate = sf.read(wav_file)
    
    # Convert milliseconds to sample indices
    start_sample = int(start_ms / 1000 * sample_rate)
    end_sample = int(end_ms / 1000 * sample_rate)
    
    # Extract chunk
    chunk = signal[start_sample:end_sample]
    
    # Save chunk to file
    sf.write(output_path, chunk, sample_rate)
    
    # Return duration in seconds
    duration = len(chunk) / sample_rate
    return duration