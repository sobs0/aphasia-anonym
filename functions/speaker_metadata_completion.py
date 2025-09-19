import pandas as pd
import numpy as np
from pathlib import Path


def complete_speaker_metadata(config, logger, split_chunks_path, filtered_metadata_path):
    """
    Fill missing speaker metadata
    """
    logger.info("Starting speaker metadata completion")
    
    # Load chunks data (contains all processing results)
    chunks_df = pd.read_csv(split_chunks_path)
    logger.info(f"Processing {len(chunks_df)} chunks for metadata calculation")
    
    # Load existing speaker metadata
    metadata_df = pd.read_csv(filtered_metadata_path)
    logger.info(f"Found metadata for {len(metadata_df)} speakers")
    
    # Calculate per-speaker statistics
    logger.info("Calculating speaker statistics...")
    
    # 1. Count number of recordings per speaker
    recordings_per_speaker = chunks_df.groupby('speaker_id')['recording_id'].nunique().reset_index()
    recordings_per_speaker.columns = ['speaker_id', 'num_recordings']
    
    # 2. Calculate total speech length per speaker (sum of all chunk durations)
    # Only count chunks that are actually usable (not filtered out)
    usable_chunks = chunks_df[chunks_df['use'] != 'NO'].copy()
    
    speech_length_per_speaker = usable_chunks.groupby('speaker_id')['audio_length'].agg([
        ('total_speech_length_seconds', 'sum'),
        ('num_usable_chunks', 'count'),
        ('avg_chunk_length', 'mean'),
        ('min_chunk_length', 'min'),
        ('max_chunk_length', 'max')
    ]).reset_index()
    
    # Convert seconds to minutes for better readability
    speech_length_per_speaker['total_speech_length_minutes'] = (
        speech_length_per_speaker['total_speech_length_seconds'] / 60
    )
    
    # 3. Calculate ASR/ASV specific statistics
    asr_chunks = chunks_df[chunks_df['use'].isin(['ONLY_ASR', 'BOTH'])]
    asv_chunks = chunks_df[chunks_df['use'].isin(['ONLY_ASV', 'BOTH'])]
    
    asr_stats_per_speaker = asr_chunks.groupby('speaker_id').agg({
        'audio_length': ['sum', 'count'],
        'asr_set': lambda x: x.value_counts().to_dict()
    }).reset_index()
    asr_stats_per_speaker.columns = ['speaker_id', 'asr_speech_length_seconds', 'asr_chunk_count', 'asr_set_distribution']
    
    asv_stats_per_speaker = asv_chunks.groupby('speaker_id').agg({
        'audio_length': ['sum', 'count']
    }).reset_index()
    asv_stats_per_speaker.columns = ['speaker_id', 'asv_speech_length_seconds', 'asv_chunk_count']
    
    # 4. Merge all statistics with existing metadata
    logger.info("Merging calculated statistics with existing metadata...")
    
    # Start with existing metadata
    complete_metadata = metadata_df.copy()
    
    # Add recordings count
    complete_metadata = complete_metadata.merge(
        recordings_per_speaker, 
        on='speaker_id', 
        how='left'
    )
    
    # Add speech length statistics
    complete_metadata = complete_metadata.merge(
        speech_length_per_speaker, 
        on='speaker_id', 
        how='left'
    )
    
    # Add ASR statistics
    complete_metadata = complete_metadata.merge(
        asr_stats_per_speaker, 
        on='speaker_id', 
        how='left'
    )
    
    # Add ASV statistics
    complete_metadata = complete_metadata.merge(
        asv_stats_per_speaker, 
        on='speaker_id', 
        how='left'
    )
    
    # Fill NaN values with 0 for counts and statistics
    numeric_columns = [
        'num_recordings', 'total_speech_length_seconds', 'total_speech_length_minutes',
        'num_usable_chunks', 'avg_chunk_length', 'min_chunk_length', 'max_chunk_length',
        'asr_speech_length_seconds', 'asr_chunk_count',
        'asv_speech_length_seconds', 'asv_chunk_count'
    ]
    
    for col in numeric_columns:
        if col in complete_metadata.columns:
            complete_metadata[col] = complete_metadata[col].fillna(0)
    
    # Log detailed statistics
    log_metadata_statistics(complete_metadata, logger)
    
    # 5. Create final metadata file with clean column order
    final_columns = [
        # Basic speaker info
        'speaker_id', 'gender', 'age', 'WAB_type', 'WAB_score',
        
        # Recording counts
        'num_recordings', 'num_usable_chunks',
        
        # Speech length statistics
        'total_speech_length_seconds', 'total_speech_length_minutes',
        'avg_chunk_length', 'min_chunk_length', 'max_chunk_length',
        
        # ASR-specific statistics
        'asr_chunk_count', 'asr_speech_length_seconds',
        
        # ASV-specific statistics  
        'asv_chunk_count', 'asv_speech_length_seconds'
    ]
    
    # Only include columns that exist
    available_columns = [col for col in final_columns if col in complete_metadata.columns]
    final_metadata = complete_metadata[available_columns].copy()
    
    # Round numeric columns for cleaner output
    round_columns = [
        'total_speech_length_seconds', 'total_speech_length_minutes',
        'avg_chunk_length', 'min_chunk_length', 'max_chunk_length',
        'asr_speech_length_seconds', 'asv_speech_length_seconds'
    ]
    
    for col in round_columns:
        if col in final_metadata.columns:
            final_metadata[col] = final_metadata[col].round(2)
    
    # Save completed metadata
    temp_dir = Path(config['output']['temporary_data_dir'])
    completed_metadata_filename = config['processing'].get('completed_metadata_filename', 'completed_metadata.csv')
    output_path = temp_dir / completed_metadata_filename
    
    final_metadata.to_csv(output_path, index=False)
    
    logger.info(f"Speaker metadata completion finished:")
    logger.info(f"  - Total speakers: {len(final_metadata)}")
    logger.info(f"  - Columns in final metadata: {len(available_columns)}")
    logger.info(f"Completed metadata saved to: {output_path}")
    
    return str(output_path)


def log_metadata_statistics(metadata_df, logger):
    """
    Log detailed statistics about the completed metadata
    """
    logger.info("="*50)
    logger.info("SPEAKER METADATA STATISTICS")
    logger.info("="*50)
    
    total_speakers = len(metadata_df)
    logger.info(f"Total speakers: {total_speakers}")
    
    # Recording statistics
    if 'num_recordings' in metadata_df.columns:
        recordings_stats = metadata_df['num_recordings'].describe()
        logger.info(f"Recordings per speaker:")
        logger.info(f"  - Mean: {recordings_stats['mean']:.1f}")
        logger.info(f"  - Median: {recordings_stats['50%']:.0f}")
        logger.info(f"  - Min: {recordings_stats['min']:.0f}")
        logger.info(f"  - Max: {recordings_stats['max']:.0f}")
        logger.info(f"  - Total recordings: {metadata_df['num_recordings'].sum():.0f}")
    
    # Speech length statistics
    if 'total_speech_length_minutes' in metadata_df.columns:
        speech_stats = metadata_df['total_speech_length_minutes'].describe()
        logger.info(f"Speech length per speaker (minutes):")
        logger.info(f"  - Mean: {speech_stats['mean']:.1f}")
        logger.info(f"  - Median: {speech_stats['50%']:.1f}")
        logger.info(f"  - Min: {speech_stats['min']:.1f}")
        logger.info(f"  - Max: {speech_stats['max']:.1f}")
        logger.info(f"  - Total speech: {metadata_df['total_speech_length_minutes'].sum():.1f} minutes")
        logger.info(f"  - Total speech: {metadata_df['total_speech_length_minutes'].sum()/60:.1f} hours")
    
    # Chunk statistics
    if 'num_usable_chunks' in metadata_df.columns:
        chunk_stats = metadata_df['num_usable_chunks'].describe()
        logger.info(f"Usable chunks per speaker:")
        logger.info(f"  - Mean: {chunk_stats['mean']:.1f}")
        logger.info(f"  - Median: {chunk_stats['50%']:.0f}")
        logger.info(f"  - Min: {chunk_stats['min']:.0f}")
        logger.info(f"  - Max: {chunk_stats['max']:.0f}")
        logger.info(f"  - Total usable chunks: {metadata_df['num_usable_chunks'].sum():.0f}")
    
    # ASR statistics
    if 'asr_chunk_count' in metadata_df.columns:
        asr_speakers = (metadata_df['asr_chunk_count'] > 0).sum()
        logger.info(f"ASR training data:")
        logger.info(f"  - Speakers with ASR data: {asr_speakers}")
        logger.info(f"  - Total ASR chunks: {metadata_df['asr_chunk_count'].sum():.0f}")
        if 'asr_speech_length_seconds' in metadata_df.columns:
            logger.info(f"  - Total ASR speech: {metadata_df['asr_speech_length_seconds'].sum()/60:.1f} minutes")
    
    # ASV statistics
    if 'asv_chunk_count' in metadata_df.columns:
        asv_speakers = (metadata_df['asv_chunk_count'] > 0).sum()
        logger.info(f"ASV training data:")
        logger.info(f"  - Speakers with ASV data: {asv_speakers}")
        logger.info(f"  - Total ASV chunks: {metadata_df['asv_chunk_count'].sum():.0f}")
        if 'asv_speech_length_seconds' in metadata_df.columns:
            logger.info(f"  - Total ASV speech: {metadata_df['asv_speech_length_seconds'].sum()/60:.1f} minutes")
    
    # Demographic statistics
    if 'gender' in metadata_df.columns:
        gender_counts = metadata_df['gender'].value_counts()
        logger.info(f"Gender distribution:")
        for gender, count in gender_counts.items():
            logger.info(f"  - {gender}: {count} ({count/total_speakers*100:.1f}%)")
    
    if 'age' in metadata_df.columns:
        # Convert age to numeric, handling potential string values
        age_numeric = pd.to_numeric(metadata_df['age'], errors='coerce')
        age_stats = age_numeric.describe()
        logger.info(f"Age distribution:")
        logger.info(f"  - Mean: {age_stats['mean']:.1f} years")
        logger.info(f"  - Median: {age_stats['50%']:.1f} years")
        logger.info(f"  - Range: {age_stats['min']:.0f} - {age_stats['max']:.0f} years")
    
    if 'WAB_type' in metadata_df.columns:
        wab_type_counts = metadata_df['WAB_type'].value_counts()
        logger.info(f"WAB type distribution:")
        for wab_type, count in wab_type_counts.items():
            logger.info(f"  - {wab_type}: {count} ({count/total_speakers*100:.1f}%)")
    
    logger.info("="*50)