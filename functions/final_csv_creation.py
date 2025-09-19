import pandas as pd
from pathlib import Path


def create_final_combined_csv(config, logger, transcript_cleaned_path, filtered_metadata_path):
    """
    Create final combined CSV with all available information
    """
    logger.info("Creating final combined CSV with all available information")
    
    # Load chunks data
    chunks_df = pd.read_csv(transcript_cleaned_path)
    logger.info(f"Loaded {len(chunks_df)} chunks from transcript-cleaned data")
    
    # Load speaker metadata
    metadata_df = pd.read_csv(filtered_metadata_path)
    logger.info(f"Loaded metadata for {len(metadata_df)} speakers")
    
    # Join chunks with speaker metadata
    combined_df = chunks_df.merge(
        metadata_df, 
        on='speaker_id', 
        how='left'
    )
    
    # Check for missing metadata
    missing_metadata = combined_df[combined_df['gender'].isna()]
    if not missing_metadata.empty:
        logger.warning(f"Missing metadata for {len(missing_metadata)} chunks")
        missing_speakers = missing_metadata['speaker_id'].unique()
        logger.warning(f"Speakers without metadata: {list(missing_speakers)}")
    
    # Reorder columns logically
    column_order = [
        # Speaker & Recording Info
        'speaker_id', 'recording_id', 'chunk_id',
        
        # Speaker Metadata
        'gender', 'age', 'WAB_type', 'WAB_score',
        
        # Timing & Audio Info  
        'start_time', 'end_time', 'audio_length',
        
        # Silence Analysis
        'silence_lengths', 'max_pause_ms', 'total_pause_ms',
        
        # Transcripts
        'original_transcript', 'cleaned_transcript',
        
        # Audio File Paths (Original)
        'cha_file_path', 'chunk_wav_path', 'mcadams_anonymized_chunk_wav_path',
        
        # Audio File Paths (Silence-Padded for ASR)
        'silence_padded_chunk_wav_path', 'mcadams_anonymized_silence_padded_chunk_wav_path',
        
        # Processing Info
        'mcadams_anonym_value', 'use'
    ]
    
    # Add any missing columns that might exist
    existing_columns = list(combined_df.columns)
    for col in existing_columns:
        if col not in column_order:
            column_order.append(col)
    
    # Select and reorder columns
    final_columns = [col for col in column_order if col in existing_columns]
    combined_df = combined_df[final_columns]
    
    # Generate final statistics
    total_chunks = len(combined_df)
    usage_counts = combined_df['use'].value_counts()
    
    logger.info(f"Final combined dataset statistics:")
    logger.info(f"  - Total chunks: {total_chunks}")
    logger.info(f"  - Unique speakers: {combined_df['speaker_id'].nunique()}")
    logger.info(f"  - Unique recordings: {combined_df['recording_id'].nunique()}")
    
    logger.info(f"Usage distribution:")
    for usage_type in ['NO', 'ONLY_ASR', 'ONLY_ASV', 'BOTH']:
        count = usage_counts.get(usage_type, 0)
        logger.info(f"  - {usage_type}: {count} ({count/total_chunks*100:.1f}%)")
    
    # Calculate usable chunks for training
    asr_usable = usage_counts.get('ONLY_ASR', 0) + usage_counts.get('BOTH', 0)
    asv_usable = usage_counts.get('ONLY_ASV', 0) + usage_counts.get('BOTH', 0)
    
    logger.info(f"Usable for training:")
    logger.info(f"  - ASR training: {asr_usable} chunks ({asr_usable/total_chunks*100:.1f}%)")
    logger.info(f"  - ASV training: {asv_usable} chunks ({asv_usable/total_chunks*100:.1f}%)")
    
    # Audio statistics
    audio_stats = combined_df['audio_length'].describe()
    logger.info(f"Audio length statistics:")
    logger.info(f"  - Mean: {audio_stats['mean']:.2f}s")
    logger.info(f"  - Median: {audio_stats['50%']:.2f}s") 
    logger.info(f"  - Min: {audio_stats['min']:.2f}s")
    logger.info(f"  - Max: {audio_stats['max']:.2f}s")
    
    # Save final combined CSV
    final_dir = Path(config['output']['final_data_dir'])
    final_dir.mkdir(parents=True, exist_ok=True)
    
    final_combined_filename = config['output']['final_combined_filename']
    output_path = final_dir / final_combined_filename
    
    combined_df.to_csv(output_path, index=False)
    logger.info(f"Final combined CSV saved to: {output_path}")
    logger.info(f"Final CSV contains {len(final_columns)} columns and {len(combined_df)} rows")
    
    return str(output_path)