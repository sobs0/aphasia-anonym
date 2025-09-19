import pandas as pd
import numpy as np
from pathlib import Path


def create_asv_pairs(config, logger, split_chunks_path):
    """
    Create ASV trial pairs.
    """
    logger.info("Starting ASV pairs creation")
    
    # Load chunks data
    chunks_df = pd.read_csv(split_chunks_path)
    logger.info(f"Processing {len(chunks_df)} total chunks for ASV pairs")
    
    # Filter for ASV-suitable chunks
    asv_suitable_mask = chunks_df['use'].isin(['ONLY_ASV', 'BOTH'])
    asv_chunks = chunks_df[asv_suitable_mask].copy()
    
    total_asv_chunks = len(asv_chunks)
    logger.info(f"ASV-suitable chunks found: {total_asv_chunks}")
    
    if total_asv_chunks == 0:
        logger.warning("No ASV-suitable chunks found for pairs creation!")
        return None
    
    # Initialize asv_role column for all chunks
    chunks_df['asv_role'] = 'unused'
    
    # Get ASV configuration parameters
    asv_config = config['asv_evaluation']
    same_speaker_ratio = asv_config['same_speaker_ratio']
    trials_per_speaker = asv_config['trials_per_speaker']
    min_enrollment_utterances = asv_config['min_enrollment_utterances']
    max_enrollment_ratio = asv_config['max_enrollment_ratio']
    
    logger.info(f"ASV Configuration:")
    logger.info(f"  - Trials per speaker: {trials_per_speaker}")
    logger.info(f"  - Same speaker ratio per speaker: {same_speaker_ratio*100:.1f}%")
    logger.info(f"  - Min enrollment utterances: {min_enrollment_utterances}")
    logger.info(f"  - Max enrollment ratio: {max_enrollment_ratio*100:.1f}%")
    
    # Analyze speaker data and create enrollment/trial splits
    speaker_analysis = analyze_speakers_for_asv(asv_chunks, logger)
    enrollment_assignments, trial_pool = create_enrollment_trial_split(
        asv_chunks, speaker_analysis, max_enrollment_ratio, min_enrollment_utterances, logger
    )
    
    # Update chunks with ASV role assignments
    for speaker_id, utterance_ids in enrollment_assignments.items():
        mask = (chunks_df['speaker_id'] == speaker_id) & (chunks_df['chunk_id'].isin(utterance_ids))
        chunks_df.loc[mask, 'asv_role'] = 'enrollment'
    
    # Mark trial utterances
    for _, row in trial_pool.iterrows():
        mask = chunks_df['chunk_id'] == row['chunk_id']
        chunks_df.loc[mask, 'asv_role'] = 'trial'
    
    # Generate trial pairs
    trial_pairs = generate_trial_pairs_per_speaker(
        enrollment_assignments, trial_pool, 
        same_speaker_ratio, trials_per_speaker, logger
    )
    
    asv_pairs_df = create_asv_pairs_dataframe(trial_pairs, asv_chunks, enrollment_assignments, logger)
    
    # Save ASV pairs CSV
    final_dir = Path(config['output']['final_data_dir'])
    asv_pairs_path = final_dir / config['output']['asv_pairs_filename']
    asv_pairs_df.to_csv(asv_pairs_path, index=False)
    
    # Update and save chunks with asv_role
    temp_dir = Path(config['output']['temporary_data_dir'])
    updated_chunks_filename = config['processing']['chunks_with_asv_roles_filename']
    updated_chunks_path = temp_dir / updated_chunks_filename
    chunks_df.to_csv(updated_chunks_path, index=False)
    
    # Log final statistics
    log_asv_statistics(asv_pairs_df, chunks_df, logger)
    
    logger.info(f"ASV pairs creation completed:")
    logger.info(f"  - ASV pairs saved to: {asv_pairs_path}")
    logger.info(f"  - Updated chunks saved to: {updated_chunks_path}")
    
    return str(asv_pairs_path)


def analyze_speakers_for_asv(asv_chunks, logger):
    """
    Analyze speakers to determine enrollment/trial strategy
    """
    logger.info("Analyzing speakers for ASV enrollment/trial assignment")
    
    speaker_analysis = {}
    
    for speaker_id in asv_chunks['speaker_id'].unique():
        speaker_chunks = asv_chunks[asv_chunks['speaker_id'] == speaker_id]
        recordings = speaker_chunks['recording_id'].unique()
        
        analysis = {
            'speaker_id': speaker_id,
            'total_utterances': len(speaker_chunks),
            'num_recordings': len(recordings),
            'recordings': list(recordings),
            'utterances_per_recording': {}
        }
        
        # Count utterances per recording
        for recording_id in recordings:
            recording_chunks = speaker_chunks[speaker_chunks['recording_id'] == recording_id]
            analysis['utterances_per_recording'][recording_id] = len(recording_chunks)
        
        speaker_analysis[speaker_id] = analysis
    
    # Log speaker statistics
    total_speakers = len(speaker_analysis)
    multi_recording_speakers = sum(1 for s in speaker_analysis.values() if s['num_recordings'] > 1)
    single_recording_speakers = total_speakers - multi_recording_speakers
    
    logger.info(f"Speaker analysis:")
    logger.info(f"  - Total speakers: {total_speakers}")
    logger.info(f"  - Single recording speakers: {single_recording_speakers}")
    logger.info(f"  - Multi-recording speakers: {multi_recording_speakers}")
    
    # Show examples
    sorted_speakers = sorted(speaker_analysis.values(), key=lambda x: x['total_utterances'], reverse=True)
    logger.info("Top speakers by utterance count:")
    for i, speaker in enumerate(sorted_speakers[:5]):
        logger.info(f"  {i+1}. {speaker['speaker_id']}: {speaker['total_utterances']} utterances, {speaker['num_recordings']} recordings")
    
    return speaker_analysis


def create_enrollment_trial_split(asv_chunks, speaker_analysis, max_enrollment_ratio, min_enrollment_utterances, logger):
    """
    Create enrollment/trial split for each speaker
    """
    logger.info("Creating enrollment/trial splits per speaker")
    
    enrollment_assignments = {}
    trial_utterances = []
    excluded_speakers = []
    
    for speaker_id, analysis in speaker_analysis.items():
        speaker_chunks = asv_chunks[asv_chunks['speaker_id'] == speaker_id]
        total_utterances = analysis['total_utterances']
        num_recordings = analysis['num_recordings']
        
        # Calculate max enrollment utterances
        max_enrollment_count = max(
            min_enrollment_utterances,
            int(total_utterances * max_enrollment_ratio)
        )
        
        if total_utterances < min_enrollment_utterances:
            logger.warning(f"Speaker {speaker_id} has only {total_utterances} utterances (< {min_enrollment_utterances}), excluding from ASV")
            excluded_speakers.append(speaker_id)
            continue
        
        if num_recordings > 1:
            # Multi-recording speaker: use one recording for enrollment, others for trial
            # Choose recording with most utterances for enrollment (up to max_enrollment_count)
            recordings_by_count = sorted(
                analysis['utterances_per_recording'].items(),
                key=lambda x: x[1], reverse=True
            )
            
            enrollment_recording = recordings_by_count[0][0]
            enrollment_utterances_in_recording = recordings_by_count[0][1]
            
            # Get enrollment utterances from chosen recording
            enrollment_chunks = speaker_chunks[speaker_chunks['recording_id'] == enrollment_recording]
            
            # If recording has more utterances than max allowed, sample randomly
            if enrollment_utterances_in_recording > max_enrollment_count:
                enrollment_chunks = enrollment_chunks.sample(n=max_enrollment_count, random_state=42)
            
            enrollment_utterance_ids = enrollment_chunks['chunk_id'].tolist()
            
            # Get trial utterances from other recordings
            trial_chunks = speaker_chunks[speaker_chunks['recording_id'] != enrollment_recording]
            
        else:
            # Single recording speaker: split utterances within recording
            # Take first max_enrollment_count utterances for enrollment, rest for trial
            speaker_chunks_shuffled = speaker_chunks.sample(frac=1, random_state=42).reset_index(drop=True)
            
            enrollment_chunks = speaker_chunks_shuffled.head(max_enrollment_count)
            trial_chunks = speaker_chunks_shuffled.tail(len(speaker_chunks_shuffled) - max_enrollment_count)
            
            enrollment_utterance_ids = enrollment_chunks['chunk_id'].tolist()
        
        # Store enrollment assignment
        enrollment_assignments[speaker_id] = enrollment_utterance_ids
        
        # Add trial utterances to pool
        for _, trial_chunk in trial_chunks.iterrows():
            trial_utterances.append(trial_chunk)
    
    trial_pool = pd.DataFrame(trial_utterances) if trial_utterances else pd.DataFrame()
    
    logger.info(f"Enrollment/trial split results:")
    logger.info(f"  - Speakers with enrollment data: {len(enrollment_assignments)}")
    logger.info(f"  - Excluded speakers (insufficient utterances): {len(excluded_speakers)}")
    logger.info(f"  - Total trial utterances: {len(trial_pool)}")
    
    # Log enrollment statistics
    enrollment_counts = [len(utterances) for utterances in enrollment_assignments.values()]
    if enrollment_counts:
        logger.info(f"Enrollment utterances per speaker:")
        logger.info(f"  - Mean: {np.mean(enrollment_counts):.1f}")
        logger.info(f"  - Min: {min(enrollment_counts)}")
        logger.info(f"  - Max: {max(enrollment_counts)}")
        logger.info(f"  - Median: {np.median(enrollment_counts):.1f}")
    
    return enrollment_assignments, trial_pool


def generate_trial_pairs_per_speaker(enrollment_assignments, trial_pool, same_speaker_ratio, trials_per_speaker, logger):
    """
    Generate trial pairs for ASV evaluation
    """
    logger.info("Generating ASV trial pairs")
    
    enrollment_speakers = list(enrollment_assignments.keys())
    all_trial_speakers = trial_pool['speaker_id'].unique()
    
    # Find speakers that have both enrollment and trial data (for same-speaker pairs)
    speakers_with_both = set(enrollment_speakers) & set(all_trial_speakers)
    
    logger.info(f"Enrollment speakers: {len(enrollment_speakers)}")
    logger.info(f"Speakers with trial data: {len(all_trial_speakers)}")
    logger.info(f"Speakers available for same-speaker pairs: {len(speakers_with_both)}")
    
    trial_pairs = []
    successful_speakers = 0
    skipped_speakers = 0
    
    # Calculate target counts per speaker
    same_pairs_per_speaker = int(trials_per_speaker * same_speaker_ratio)
    diff_pairs_per_speaker = trials_per_speaker - same_pairs_per_speaker
    
    logger.info(f"Per speaker: {same_pairs_per_speaker} same-speaker, {diff_pairs_per_speaker} different-speaker pairs")
    
    # Generate pairs for each enrollment speaker
    for enrollment_speaker in enrollment_speakers:
        speaker_pairs = []
        
        # 1. Generate same-speaker pairs (if possible)
        same_pairs_generated = 0
        if enrollment_speaker in speakers_with_both:
            # Get trial utterances for this speaker
            speaker_trial_utterances = trial_pool[
                trial_pool['speaker_id'] == enrollment_speaker
            ]['chunk_id'].tolist()
            
            # Generate same-speaker pairs (up to available or needed)
            available_same_pairs = min(len(speaker_trial_utterances), same_pairs_per_speaker)
            
            # Randomly sample trial utterances to avoid always using the same ones
            np.random.seed(42 + hash(enrollment_speaker) % 1000)
            sampled_trial_utterances = np.random.choice(
                speaker_trial_utterances, 
                size=available_same_pairs, 
                replace=False
            ).tolist()
            
            for trial_utterance_id in sampled_trial_utterances:
                pair = {
                    'trial_id': f"{enrollment_speaker}+{enrollment_speaker}",
                    'same_speaker_label': 1,
                    'enrollment_speaker_id': enrollment_speaker,
                    'trial_speaker_id': enrollment_speaker,
                    'trial_utterance_id': trial_utterance_id
                }
                speaker_pairs.append(pair)
                same_pairs_generated += 1
        
        # 2. Generate different-speaker pairs
        diff_pairs_generated = 0
        other_speakers = [s for s in all_trial_speakers if s != enrollment_speaker]
        
        if not other_speakers:
            logger.warning(f"No different speakers available for {enrollment_speaker}")
            skipped_speakers += 1
            continue
        
        # Generate different-speaker pairs
        np.random.seed(42 + hash(enrollment_speaker) % 1000)
        
        while diff_pairs_generated < diff_pairs_per_speaker:
            # Randomly select a different speaker
            trial_speaker = np.random.choice(other_speakers)
            
            # Get trial utterances from this speaker
            speaker_trial_utterances = trial_pool[
                trial_pool['speaker_id'] == trial_speaker
            ]['chunk_id'].tolist()
            
            if not speaker_trial_utterances:
                continue
            
            # Select random trial utterance
            trial_utterance_id = np.random.choice(speaker_trial_utterances)
            
            pair = {
                'trial_id': f"{enrollment_speaker}+{trial_speaker}",
                'same_speaker_label': 0,
                'enrollment_speaker_id': enrollment_speaker,
                'trial_speaker_id': trial_speaker,
                'trial_utterance_id': trial_utterance_id
            }
            speaker_pairs.append(pair)
            diff_pairs_generated += 1
        
        # Add speaker's pairs to total
        trial_pairs.extend(speaker_pairs)
        successful_speakers += 1
        
        # Log progress
        if successful_speakers % 50 == 0:
            logger.info(f"Generated pairs for {successful_speakers} speakers...")
    
    # Final statistics
    total_pairs = len(trial_pairs)
    total_same_pairs = sum(1 for p in trial_pairs if p['same_speaker_label'] == 1)
    total_diff_pairs = total_pairs - total_same_pairs
    
    logger.info(f"Trial pairs generation completed:")
    logger.info(f"  - Enrollment speakers processed: {successful_speakers}")
    logger.info(f"  - Speakers skipped (no different speakers): {skipped_speakers}")
    logger.info(f"  - Total trial pairs: {total_pairs}")
    logger.info(f"  - Same-speaker pairs: {total_same_pairs} ({total_same_pairs/total_pairs*100:.1f}%)")
    logger.info(f"  - Different-speaker pairs: {total_diff_pairs} ({total_diff_pairs/total_pairs*100:.1f}%)")
    logger.info(f"  - Average pairs per speaker: {total_pairs/successful_speakers:.1f}")
    
    return trial_pairs


def create_asv_pairs_dataframe(trial_pairs, asv_chunks, enrollment_assignments, logger):
    """
    Create final ASV pairs df with all required info
    """
    logger.info("Creating ASV pairs DataFrame")
    
    # Convert trial pairs to DataFrame
    pairs_df = pd.DataFrame(trial_pairs)
    
    # Add additional columns
    pairs_df['num_recordings_enrollment_speaker'] = None
    pairs_df['enrollment_length_seconds'] = None
    pairs_df['enrollment_utterance_ids'] = None
    
    # Add empty score/prediction columns for later filling
    attack_levels = ['oo', 'oa', 'aa']
    for level in attack_levels:
        pairs_df[f'{level}_score'] = None
        pairs_df[f'{level}_prediction'] = None
    
    # Get speaker metadata for filling additional info
    speaker_info = {}
    for speaker_id in pairs_df['enrollment_speaker_id'].unique():
        speaker_chunks = asv_chunks[asv_chunks['speaker_id'] == speaker_id]
        num_recordings = speaker_chunks['recording_id'].nunique()
        
        # Get enrollment utterances and calculate total length
        enrollment_utterance_ids = enrollment_assignments.get(speaker_id, [])
        enrollment_chunks = asv_chunks[asv_chunks['chunk_id'].isin(enrollment_utterance_ids)]
        enrollment_length = enrollment_chunks['audio_length'].sum()
        
        speaker_info[speaker_id] = {
            'num_recordings': num_recordings,
            'enrollment_utterance_ids': enrollment_utterance_ids,
            'enrollment_length_seconds': round(enrollment_length, 2)
        }
    
    # Fill in metadata for each pair
    for idx, row in pairs_df.iterrows():
        enrollment_speaker = row['enrollment_speaker_id']
        
        # Number of recordings
        pairs_df.at[idx, 'num_recordings_enrollment_speaker'] = speaker_info[enrollment_speaker]['num_recordings']
        
        # Enrollment length in seconds
        pairs_df.at[idx, 'enrollment_length_seconds'] = speaker_info[enrollment_speaker]['enrollment_length_seconds']
        
        # Enrollment utterance IDs as comma-separated string
        enrollment_ids = speaker_info[enrollment_speaker]['enrollment_utterance_ids']
        pairs_df.at[idx, 'enrollment_utterance_ids'] = ','.join(enrollment_ids)
    
    # Reorder columns to match specification
    column_order = [
        'trial_id', 'same_speaker_label', 'enrollment_speaker_id', 'trial_speaker_id',
        'num_recordings_enrollment_speaker', 'enrollment_length_seconds', 
        'enrollment_utterance_ids', 'trial_utterance_id',
        'oo_score', 'oo_prediction', 'oa_score', 'oa_prediction', 'aa_score', 'aa_prediction'
    ]
    
    pairs_df = pairs_df[column_order]
    
    logger.info(f"ASV pairs DataFrame created with {len(pairs_df)} pairs and {len(column_order)} columns")
    
    return pairs_df


def log_asv_statistics(asv_pairs_df, chunks_df, logger):
    """
    Log comprehensive ASV statistics
    """
    logger.info("="*60)
    logger.info("ASV PAIRS STATISTICS (PER-SPEAKER APPROACH)")
    logger.info("="*60)
    
    # Pair statistics
    total_pairs = len(asv_pairs_df)
    same_pairs = asv_pairs_df['same_speaker_label'].sum()
    diff_pairs = total_pairs - same_pairs
    
    logger.info(f"Trial pairs:")
    logger.info(f"  - Total pairs: {total_pairs}")
    logger.info(f"  - Same speaker pairs: {same_pairs} ({same_pairs/total_pairs*100:.1f}%)")
    logger.info(f"  - Different speaker pairs: {diff_pairs} ({diff_pairs/total_pairs*100:.1f}%)")
    
    # Per-speaker statistics
    enrollment_speakers = asv_pairs_df['enrollment_speaker_id'].nunique()
    trial_speakers = asv_pairs_df['trial_speaker_id'].nunique()
    
    # Calculate pairs per enrollment speaker
    pairs_per_speaker = asv_pairs_df.groupby('enrollment_speaker_id').size()
    same_pairs_per_speaker = asv_pairs_df[asv_pairs_df['same_speaker_label'] == 1].groupby('enrollment_speaker_id').size()
    
    logger.info(f"Speaker statistics:")
    logger.info(f"  - Unique enrollment speakers: {enrollment_speakers}")
    logger.info(f"  - Unique trial speakers: {trial_speakers}")
    logger.info(f"  - Pairs per enrollment speaker:")
    logger.info(f"    - Mean: {pairs_per_speaker.mean():.1f}")
    logger.info(f"    - Min: {pairs_per_speaker.min()}")
    logger.info(f"    - Max: {pairs_per_speaker.max()}")
    
    if len(same_pairs_per_speaker) > 0:
        logger.info(f"  - Same-speaker pairs per enrollment speaker:")
        logger.info(f"    - Mean: {same_pairs_per_speaker.mean():.1f}")
        logger.info(f"    - Min: {same_pairs_per_speaker.min()}")
        logger.info(f"    - Max: {same_pairs_per_speaker.max()}")
    
    # ASV role distribution in chunks
    asv_role_counts = chunks_df['asv_role'].value_counts()
    logger.info(f"ASV role distribution in chunks:")
    for role, count in asv_role_counts.items():
        total_chunks = len(chunks_df)
        logger.info(f"  - {role}: {count} ({count/total_chunks*100:.1f}%)")
    
    # Enrollment vs Trial utterances
    enrollment_utterances = asv_role_counts.get('enrollment', 0)
    trial_utterances = asv_role_counts.get('trial', 0)
    
    if enrollment_utterances > 0 and trial_utterances > 0:
        logger.info(f"Enrollment vs Trial balance:")
        logger.info(f"  - Enrollment utterances: {enrollment_utterances}")
        logger.info(f"  - Trial utterances: {trial_utterances}")
        logger.info(f"  - Enrollment/Trial ratio: 1:{trial_utterances/enrollment_utterances:.1f}")
    
    logger.info("="*60)