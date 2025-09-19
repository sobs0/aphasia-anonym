import pandas as pd
from pathlib import Path


def mark_chunk_usage_by_length(config, logger, silence_filtered_path):
    """
    Measure chunk lengths and mark usage based on audio duration
    """
    logger.info("Starting length-based usage marking for chunks")
    
    # Load chunks data
    chunks_df = pd.read_csv(silence_filtered_path)
    logger.info(f"Processing {len(chunks_df)} chunks for usage marking")
    
    # Get length thresholds from config
    min_asr_length = config['processing']['min_asr_length']
    min_asv_length = config['processing']['min_asv_length']
    
    logger.info(f"Length thresholds: ASR >= {min_asr_length}s, ASV >= {min_asv_length}s")
    
    # Initialize usage column
    chunks_df['use'] = None
    
    # Count chunks in each category
    no_chunks = 0
    only_asr_chunks = 0
    both_chunks = 0
    missing_length = 0
    
    # Apply length-based marking logic
    for idx, row in chunks_df.iterrows():
        audio_length = row['audio_length']
        
        # Handle missing audio length
        if pd.isna(audio_length):
            chunks_df.at[idx, 'use'] = 'NO'
            missing_length += 1
            no_chunks += 1
            continue
        
        # Apply usage marking logic
        if audio_length < min_asr_length:
            # Too short for any use
            chunks_df.at[idx, 'use'] = 'NO'
            no_chunks += 1
        elif audio_length < min_asv_length:
            # Long enough for ASR only
            chunks_df.at[idx, 'use'] = 'ONLY_ASR'
            only_asr_chunks += 1
        else:
            # Long enough for both ASR and ASV
            chunks_df.at[idx, 'use'] = 'BOTH'
            both_chunks += 1
    
    # Log statistics
    total_chunks = len(chunks_df)
    logger.info(f"Usage marking completed:")
    logger.info(f"  - NO (< {min_asr_length}s): {no_chunks} ({no_chunks/total_chunks*100:.1f}%)")
    logger.info(f"  - ONLY_ASR ({min_asr_length}s - {min_asv_length}s): {only_asr_chunks} ({only_asr_chunks/total_chunks*100:.1f}%)")
    logger.info(f"  - BOTH (>= {min_asv_length}s): {both_chunks} ({both_chunks/total_chunks*100:.1f}%)")
    
    if missing_length > 0:
        logger.warning(f"  - Missing audio length: {missing_length} chunks")
    
    # Calculate usable chunks for ASR and ASV
    asr_usable = only_asr_chunks + both_chunks
    asv_usable = both_chunks
    
    logger.info(f"Usable chunks:")
    logger.info(f"  - For ASR training: {asr_usable} ({asr_usable/total_chunks*100:.1f}%)")
    logger.info(f"  - For ASV training: {asv_usable} ({asv_usable/total_chunks*100:.1f}%)")
    
    # Save updated chunks data
    temp_dir = Path(config['output']['temporary_data_dir'])
    length_marked_filename = config['processing']['length_marked_filename']
    output_path = temp_dir / length_marked_filename
    
    chunks_df.to_csv(output_path, index=False)
    logger.info(f"Length-marked chunks data saved to: {output_path}")
    
    return str(output_path)