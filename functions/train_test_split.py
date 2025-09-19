import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import defaultdict


def apply_train_test_split(config, logger, transcript_cleaned_path):
    """
    Stratified train/validation/test split for ASR-suitable chunks
    """
    logger.info("Starting hierarchically stratified train/validation/test split")
    
    # Load chunks data
    chunks_df = pd.read_csv(transcript_cleaned_path)
    logger.info(f"Processing {len(chunks_df)} total chunks for train/test split")
    
    # Get split parameters from config
    split_ratios = config['processing']['split_ratios'] 
    random_seed = config['processing']['random_seed']
    
    # Validate split ratios
    if abs(sum(split_ratios) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {sum(split_ratios)}")
    
    train_ratio, val_ratio, test_ratio = split_ratios
    logger.info(f"Split ratios - Train: {train_ratio}, Validation: {val_ratio}, Test: {test_ratio}")
    logger.info(f"Using random seed: {random_seed}")
    
    # Load speaker metadata for stratification
    metadata_path = Path(config['output']['temporary_data_dir']) / config['processing']['filtered_metadata_filename']
    metadata_df = pd.read_csv(metadata_path)
    logger.info(f"Loaded metadata for {len(metadata_df)} speakers")
    
    # Initialize asr_set column for all chunks
    chunks_df['asr_set'] = None
    
    # Filter for ASR-suitable chunks
    asr_suitable_mask = chunks_df['use'].isin(['ONLY_ASR', 'BOTH'])
    asr_suitable_chunks = chunks_df[asr_suitable_mask].copy()
    
    total_asr_chunks = len(asr_suitable_chunks)
    logger.info(f"ASR-suitable chunks found: {total_asr_chunks}")
    
    if total_asr_chunks == 0:
        logger.warning("No ASR-suitable chunks found for splitting!")
        # Save with empty asr_set column
        temp_dir = Path(config['output']['temporary_data_dir'])
        split_filename = config['processing'].get('split_chunks_filename', 'split_chunks.csv')
        output_path = temp_dir / split_filename
        chunks_df.to_csv(output_path, index=False)
        return str(output_path)
    
    # Get unique speakers with chunk count and merge with metadata
    speaker_chunks = asr_suitable_chunks.groupby('speaker_id').size().reset_index(name='chunk_count')
    speaker_info = speaker_chunks.merge(metadata_df, on='speaker_id', how='left')
    
    logger.info(f"Found {len(speaker_info)} unique speakers in ASR-suitable chunks")
    logger.info(f"Total ASR chunks: {speaker_info['chunk_count'].sum()}")
    
    # Perform hierarchical stratified split
    train_speakers, val_speakers, test_speakers = hierarchical_stratified_split(
        speaker_info=speaker_info,
        split_ratios=split_ratios,
        random_seed=random_seed,
        logger=logger
    )
    
    # Assign chunks to sets based on speaker assignment
    def assign_speaker_to_set(speaker_id):
        if speaker_id in train_speakers:
            return 'train'
        elif speaker_id in val_speakers:
            return 'validation'
        elif speaker_id in test_speakers:
            return 'test'
        else:
            return None
    
    # Apply assignments to ASR-suitable chunks
    asr_suitable_chunks['asr_set'] = asr_suitable_chunks['speaker_id'].apply(assign_speaker_to_set)
    
    # Count chunks in each set
    set_counts = asr_suitable_chunks['asr_set'].value_counts()
    train_chunks = set_counts.get('train', 0)
    val_chunks = set_counts.get('validation', 0)
    test_chunks = set_counts.get('test', 0)
    
    logger.info(f"Final chunk distribution:")
    logger.info(f"  - Train chunks: {train_chunks} ({train_chunks/total_asr_chunks*100:.1f}%)")
    logger.info(f"  - Validation chunks: {val_chunks} ({val_chunks/total_asr_chunks*100:.1f}%)")
    logger.info(f"  - Test chunks: {test_chunks} ({test_chunks/total_asr_chunks*100:.1f}%)")
    
    # Verify speaker isolation and demographic balance
    verify_speaker_isolation(asr_suitable_chunks, logger)
    verify_demographic_balance(asr_suitable_chunks, metadata_df, logger)
    
    # Update the main dataframe with split assignments
    for idx, row in asr_suitable_chunks.iterrows():
        chunks_df.at[idx, 'asr_set'] = row['asr_set']
    
    # Count final distribution including non-ASR chunks
    final_set_counts = chunks_df['asr_set'].value_counts()
    logger.info(f"Complete dataset distribution:")
    logger.info(f"  - Train: {final_set_counts.get('train', 0)}")
    logger.info(f"  - Validation: {final_set_counts.get('validation', 0)}")
    logger.info(f"  - Test: {final_set_counts.get('test', 0)}")
    logger.info(f"  - No assignment (non-ASR): {chunks_df['asr_set'].isnull().sum()}")
    
    # Save the split dataset
    temp_dir = Path(config['output']['temporary_data_dir'])
    split_filename = config['processing'].get('split_chunks_filename', 'split_chunks.csv')
    output_path = temp_dir / split_filename
    
    chunks_df.to_csv(output_path, index=False)
    logger.info(f"Hierarchically stratified split completed and saved to: {output_path}")
    
    return str(output_path)


def hierarchical_stratified_split(speaker_info, split_ratios, random_seed, logger):
    """
    Perform hierarchical stratified split based on WAB_type → WAB_score → Gender → Age
    """
    logger.info("Performing hierarchical stratified speaker split")
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Clean and prepare stratification variables
    speaker_info = speaker_info.copy()
    
    # 1. Normalize WAB_type (treat "aphasia" as separate type, handle missing)
    speaker_info['wab_type_clean'] = speaker_info['WAB_type'].fillna('unknown').str.lower()
    
    # 2. Create WAB_score bins (based on config thresholds)
    speaker_info['wab_severity'] = speaker_info.apply(lambda row: categorize_wab_severity(row['WAB_score']), axis=1)
    
    # 3. Normalize gender
    speaker_info['gender_clean'] = speaker_info['gender'].fillna('unknown').str.lower()
    
    # 4. Create age bins
    speaker_info['age_bin'] = speaker_info.apply(lambda row: categorize_age(row['age']), axis=1)
    
    # Log stratification variables distribution
    log_stratification_stats(speaker_info, logger)
    
    # Create hierarchical strata
    speaker_info['stratum'] = (
        speaker_info['wab_type_clean'].astype(str) + '_' +
        speaker_info['wab_severity'].astype(str) + '_' +
        speaker_info['gender_clean'].astype(str) + '_' +
        speaker_info['age_bin'].astype(str)
    )
    
    # Group by strata and perform stratified split
    train_speakers = set()
    val_speakers = set()
    test_speakers = set()
    
    train_ratio, val_ratio, test_ratio = split_ratios
    
    strata_info = []
    
    for stratum, group in speaker_info.groupby('stratum'):
        speakers_in_stratum = list(group['speaker_id'])
        chunks_in_stratum = group['chunk_count'].sum()
        n_speakers = len(speakers_in_stratum)
        
        strata_info.append({
            'stratum': stratum,
            'n_speakers': n_speakers,
            'n_chunks': chunks_in_stratum
        })
        
        if n_speakers == 1:
            # Single speaker - assign randomly to maintain ratios
            choice = np.random.choice(['train', 'val', 'test'], p=split_ratios)
            if choice == 'train':
                train_speakers.add(speakers_in_stratum[0])
            elif choice == 'val':
                val_speakers.add(speakers_in_stratum[0])
            else:
                test_speakers.add(speakers_in_stratum[0])
                
        elif n_speakers == 2:
            # Two speakers - assign to different sets
            np.random.shuffle(speakers_in_stratum)
            train_speakers.add(speakers_in_stratum[0])
            val_speakers.add(speakers_in_stratum[1])
            
        else:
            # Multiple speakers - stratified split
            # Calculate target counts
            n_train = max(1, int(n_speakers * train_ratio))
            n_val = max(1, int(n_speakers * val_ratio))
            n_test = n_speakers - n_train - n_val
            
            # Ensure at least one in test if possible
            if n_test == 0 and n_speakers >= 3:
                n_train -= 1
                n_test = 1
            
            # Shuffle and assign
            np.random.shuffle(speakers_in_stratum)
            train_speakers.update(speakers_in_stratum[:n_train])
            val_speakers.update(speakers_in_stratum[n_train:n_train + n_val])
            test_speakers.update(speakers_in_stratum[n_train + n_val:])
    
    # Log strata information
    logger.info(f"Created {len(strata_info)} strata for stratification")
    for info in sorted(strata_info, key=lambda x: x['n_speakers'], reverse=True)[:10]:
        logger.info(f"  Stratum '{info['stratum']}': {info['n_speakers']} speakers, {info['n_chunks']} chunks")
    
    logger.info(f"Speaker assignment completed:")
    logger.info(f"  - Train speakers: {len(train_speakers)}")
    logger.info(f"  - Validation speakers: {len(val_speakers)}")
    logger.info(f"  - Test speakers: {len(test_speakers)}")
    
    return train_speakers, val_speakers, test_speakers


def categorize_wab_severity(wab_score):
    """Categorize WAB score into severity levels (manual treshholds)"""
    if pd.isna(wab_score):
        return 'unknown'
    
    try:
        score = float(wab_score)
        if score >= 93.8:
            return 'very_mild'
        elif score >= 76:
            return 'mild' 
        elif score >= 51:
            return 'moderate'
        elif score >= 26:
            return 'severe'
        else:
            return 'very_severe'
    except (ValueError, TypeError):
        return 'unknown'


def categorize_age(age):
    """Categorize age into bins"""
    if pd.isna(age):
        return 'unknown'
    
    try:
        age_val = float(age)
        if age_val < 40:
            return 'young'
        elif age_val < 60:
            return 'middle'
        elif age_val < 75:
            return 'senior'
        else:
            return 'elderly'
    except (ValueError, TypeError):
        return 'unknown'


def log_stratification_stats(speaker_info, logger):
    """Log statistics about stratification variables."""
    logger.info("Stratification variables distribution:")
    
    # WAB Type
    wab_type_counts = speaker_info['wab_type_clean'].value_counts()
    logger.info(f"WAB Types: {dict(wab_type_counts)}")
    
    # WAB Severity
    severity_counts = speaker_info['wab_severity'].value_counts()
    logger.info(f"WAB Severity: {dict(severity_counts)}")
    
    # Gender
    gender_counts = speaker_info['gender_clean'].value_counts()
    logger.info(f"Gender: {dict(gender_counts)}")
    
    # Age bins
    age_counts = speaker_info['age_bin'].value_counts()
    logger.info(f"Age bins: {dict(age_counts)}")


def verify_demographic_balance(asr_suitable_chunks, metadata_df, logger):
    """
    Verify that demographic characteristics are balanced across train/val/test sets
    """
    logger.info("Verifying demographic balance across train/val/test sets...")
    
    # Merge chunks with metadata
    chunks_with_meta = asr_suitable_chunks.merge(metadata_df, on='speaker_id', how='left')
    
    # Analyze balance for each demographic variable
    for variable in ['WAB_type', 'gender', 'age']:
        logger.info(f"\n{variable} distribution across sets:")
        
        if variable == 'age':
            # Create age bins for analysis
            chunks_with_meta['age_bin'] = chunks_with_meta['age'].apply(categorize_age)
            variable_col = 'age_bin'
        else:
            variable_col = variable
        
        # Count by set and variable
        balance_table = pd.crosstab(chunks_with_meta['asr_set'], chunks_with_meta[variable_col], margins=True)
        
        # Calculate percentages
        balance_pct = pd.crosstab(chunks_with_meta['asr_set'], chunks_with_meta[variable_col], normalize='index') * 100
        
        # Log results
        for asr_set in ['train', 'validation', 'test']:
            if asr_set in balance_pct.index:
                set_dist = balance_pct.loc[asr_set].to_dict()
                logger.info(f"  {asr_set.capitalize()}: {set_dist}")
    
    logger.info("Demographic balance verification completed")


def verify_speaker_isolation(asr_suitable_chunks, logger):
    """
    Verify that no speaker appears in multiple train/val/test sets
    """
    logger.info("Verifying speaker isolation across train/val/test sets...")
    
    # Group by speaker and check if any speaker has multiple set assignments
    speaker_sets = asr_suitable_chunks.groupby('speaker_id')['asr_set'].nunique()
    violating_speakers = speaker_sets[speaker_sets > 1]
    
    if len(violating_speakers) > 0:
        logger.error(f"VIOLATION: {len(violating_speakers)} speakers appear in multiple sets!")
        for speaker_id, num_sets in violating_speakers.items():
            speaker_data = asr_suitable_chunks[asr_suitable_chunks['speaker_id'] == speaker_id]
            sets = speaker_data['asr_set'].unique()
            logger.error(f"  Speaker {speaker_id} appears in sets: {sets}")
        raise ValueError("Speaker isolation violated - speakers found in multiple sets!")
    else:
        logger.info("✓ Speaker isolation verified - no speaker appears in multiple sets")
    
    # Show some speaker examples
    logger.info("Sample speaker assignments:")
    sample_speakers = asr_suitable_chunks['speaker_id'].unique()[:5]
    for speaker in sample_speakers:
        speaker_data = asr_suitable_chunks[asr_suitable_chunks['speaker_id'] == speaker]
        set_assignment = speaker_data['asr_set'].iloc[0]
        chunk_count = len(speaker_data)
        logger.info(f"  Speaker {speaker}: {chunk_count} chunks → {set_assignment} set")


def get_split_statistics(chunks_df, logger):
    """
    Generate detailed statistics about the train/val/test split
    """
    stats = {}
    
    # Overall statistics
    total_chunks = len(chunks_df)
    asr_chunks = chunks_df[chunks_df['use'].isin(['ONLY_ASR', 'BOTH'])]
    
    stats['total_chunks'] = total_chunks
    stats['asr_suitable_chunks'] = len(asr_chunks)
    
    # Per-set statistics
    for set_name in ['train', 'validation', 'test']:
        set_chunks = chunks_df[chunks_df['asr_set'] == set_name]
        stats[f'{set_name}_chunks'] = len(set_chunks)
        stats[f'{set_name}_speakers'] = set_chunks['speaker_id'].nunique() if len(set_chunks) > 0 else 0
        
        if len(set_chunks) > 0:
            stats[f'{set_name}_avg_duration'] = set_chunks['audio_length'].mean()
            stats[f'{set_name}_total_duration'] = set_chunks['audio_length'].sum()
    
    return stats