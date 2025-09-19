import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import scipy.signal
from pathlib import Path
from tqdm import tqdm


def prepare_audio(file_path):
    # Load original audio
    y, sr = librosa.load(file_path, sr=None, mono=False)
    # 1. Convert to Mono audio if necessary
    if y.ndim == 2:
        y = librosa.to_mono(y)
    
    # 2. Resample to 16kHz if necessary
    target_sr=16000
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    # 3. Convert to float32 if necesssary
    if y.dtype != np.float32:
        y = y.astype(np.float32)
    # Return prepared audio
    return y, sr


def anonymize_audio_chunks(config, logger, silence_filtered_path):
    """
    Run anonymization on audio chunks using McAdams coefficient
    """
    logger.info("Starting McAdams anonymization of audio chunks")
    
    # Create anonymized audio directory
    temp_dir = Path(config['output']['temporary_data_dir'])
    anonymized_dir = temp_dir / 'audio_chunks_mcadams'
    anonymized_dir.mkdir(parents=True, exist_ok=True)
    
    # Output file path
    anonymized_chunks_filename = config['processing']['anonymized_chunks_filename']
    output_path = temp_dir / anonymized_chunks_filename
    
    # Load input chunks
    chunks_df_input = pd.read_csv(silence_filtered_path)
    logger.info(f"Total chunks to process: {len(chunks_df_input)}")
    
    # Check already processed chunks
    existing_chunks_df = None
    already_processed_chunk_ids = set()
    
    if output_path.exists():
        logger.info(f"Found existing anonymized chunks file: {output_path}")
        existing_chunks_df = pd.read_csv(output_path)
        
        # Check which chunks have both CSV entry and actual file
        verified_processed = []
        for _, row in existing_chunks_df.iterrows():
            chunk_id = row['chunk_id']
            anon_path = row.get('mcadams_anonymized_chunk_wav_path', '')
            
            if pd.notna(anon_path) and Path(anon_path).exists():
                verified_processed.append(chunk_id)
            else:
                logger.warning(f"CSV entry exists but file missing for chunk {chunk_id}: {anon_path}")
        
        already_processed_chunk_ids = set(verified_processed)
        logger.info(f"Verified processed chunks (both CSV + file exist): {len(already_processed_chunk_ids)}")
        
        # Filter existing chunks to only keep those with verified files
        existing_chunks_df = existing_chunks_df[
            existing_chunks_df['chunk_id'].isin(already_processed_chunk_ids)
        ].copy()
        
    else:
        logger.info("No existing anonymized chunks file found")
    
    # Determine which chunks need processing
    new_chunks_df = chunks_df_input[
        ~chunks_df_input['chunk_id'].isin(already_processed_chunk_ids)
    ].copy()
    
    if new_chunks_df.empty:
        logger.info("No new chunks to anonymize - all chunks already exist with verified files!")
        return str(output_path)
    
    logger.info(f"Found {len(new_chunks_df)} new chunks to anonymize")
    
    # Initialize result dataframe
    if existing_chunks_df is not None:
        # Start with existing verified data
        chunks_df = existing_chunks_df.copy()
    else:
        # Start with empty dataframe
        chunks_df = pd.DataFrame(columns=list(chunks_df_input.columns) + 
                                         ['mcadams_anonymized_chunk_wav_path', 'mcadams_anonym_value'])
    
    # Get McAdams parameters from config
    mcadams_range = config['anonymization']['mcadams_coefficient_range']
    
    # Initialize new columns for new chunks
    new_chunks_df['mcadams_anonymized_chunk_wav_path'] = None
    new_chunks_df['mcadams_anonym_value'] = None
    
    logger.info(f"Anonymizing {len(new_chunks_df)} new chunks...")
    logger.info(f"McAdams coefficient range: {mcadams_range}")
    
    successful_anonymizations = 0
    failed_anonymizations = 0
    skipped_existing = 0
    
    # Process each new chunk
    for idx, row in tqdm(new_chunks_df.iterrows(), total=len(new_chunks_df), desc="Anonymizing chunks"):
        chunk_id = row['chunk_id']
        original_chunk_path = row['chunk_wav_path']
        
        # Create anonymized filename and path
        anonymized_filename = f"{chunk_id}_mca.wav"
        anonymized_path = anonymized_dir / anonymized_filename
        
        # Check if anonymized file already exists on disk
        if anonymized_path.exists():
            logger.info(f"Anonymized file already exists for {chunk_id}, skipping processing")
            
            new_chunks_df.at[idx, 'mcadams_anonymized_chunk_wav_path'] = str(anonymized_path)
            new_chunks_df.at[idx, 'mcadams_anonym_value'] = 0.7
            
            skipped_existing += 1
            continue
        
        # Skip if original chunk doesn't exist
        if pd.isna(original_chunk_path) or not Path(original_chunk_path).exists():
            logger.warning(f"Original chunk not found for {chunk_id}: {original_chunk_path}")
            failed_anonymizations += 1
            continue
        
        try:
            # Load and prepare audio
            y, sr = prepare_audio(original_chunk_path)
            
            # Generate random McAdams coefficient for this chunk
            mcadams_coeff = round(np.random.uniform(mcadams_range[0], mcadams_range[1]), 3)
            
            # Apply McAdams anonymization
            anonymized_audio = anonym_v2(sr, y, mcadams=mcadams_coeff)
            
            # Save anonymized audio
            sf.write(str(anonymized_path), anonymized_audio, sr)
            
            # Verify file was created successfully
            if not anonymized_path.exists():
                raise FileNotFoundError(f"Failed to create anonymized file: {anonymized_path}")
            
            # Update dataframe
            new_chunks_df.at[idx, 'mcadams_anonymized_chunk_wav_path'] = str(anonymized_path)
            new_chunks_df.at[idx, 'mcadams_anonym_value'] = mcadams_coeff
            
            successful_anonymizations += 1
            
            if successful_anonymizations % 50 == 0:
                logger.info(f"Anonymized {successful_anonymizations} chunks...")
                
        except Exception as e:
            logger.error(f"Failed to anonymize chunk {chunk_id}: {str(e)}")
            failed_anonymizations += 1
            continue
    
    # Combine existing + new processed chunks
    if not new_chunks_df.empty:
        chunks_df = pd.concat([chunks_df, new_chunks_df], ignore_index=True)
    
    # Final verification: ensure all entries have existing files
    logger.info("Performing final verification of anonymized files...")
    verified_chunks = []
    missing_files = 0
    
    for _, row in chunks_df.iterrows():
        anon_path = row.get('mcadams_anonymized_chunk_wav_path', '')
        if pd.notna(anon_path) and Path(anon_path).exists():
            verified_chunks.append(row)
        else:
            missing_files += 1
            logger.warning(f"Missing anonymized file for chunk {row['chunk_id']}: {anon_path}")
    
    if missing_files > 0:
        logger.warning(f"Found {missing_files} chunks with missing anonymized files")
        chunks_df = pd.DataFrame(verified_chunks)
    
    # Save updated chunks data
    chunks_df.to_csv(output_path, index=False)
    
    # Final statistics
    total_chunks = len(chunks_df)
    previously_processed = len(already_processed_chunk_ids)
    
    logger.info(f"McAdams anonymization completed:")
    logger.info(f"  - Total chunks in final file: {total_chunks}")
    logger.info(f"  - Previously processed (verified): {previously_processed}")
    logger.info(f"  - Skipped (file existed): {skipped_existing}")
    logger.info(f"  - Newly processed: {successful_anonymizations}")
    logger.info(f"  - Failed: {failed_anonymizations}")
    logger.info(f"  - Missing files removed: {missing_files}")
    logger.info(f"  - Anonymized audio directory: {anonymized_dir}")
    logger.info(f"Updated chunks data saved to: {output_path}")
    
    # Verify directory contents
    actual_files = len(list(anonymized_dir.glob("*.wav")))
    logger.info(f"Verification: {actual_files} anonymized audio files in directory")
    
    return str(output_path)

# Exact same anonymization function from VPC 2022 baseline B2
def anonym_v2(freq, samples, winLengthinms=20, shiftLengthinms=10, lp_order=20, mcadams=0.95):
    eps = np.finfo(np.float32).eps
    samples = samples + eps

    winlen = int(np.floor(winLengthinms * 0.001 * freq))
    shift = int(np.floor(shiftLengthinms * 0.001 * freq))
    length_sig = len(samples)

    wPR = np.hanning(winlen)
    K = np.sum(wPR) / shift
    win = np.sqrt(wPR / K)

    frames = librosa.util.frame(samples, frame_length=winlen, hop_length=shift).T
    windowed_frames = frames * win
    nframe = windowed_frames.shape[0]

    lpc_coefs = np.stack([librosa.core.lpc(f + eps, order=lp_order) for f in windowed_frames])
    ar_poles = np.array([scipy.signal.tf2zpk(np.array([1]), x)[1] for x in lpc_coefs])

    def _mcadam_angle(poles, mcadams):
        old_angles = np.angle(poles)
        new_angles = np.zeros_like(old_angles) + old_angles
        real_idx = ~np.isreal(poles)
        neg_idx = np.bitwise_and(real_idx, old_angles < 0.0)
        pos_idx = np.bitwise_and(real_idx, old_angles > 0.0)
        new_angles[neg_idx] = -((-np.angle(poles[neg_idx])) ** mcadams)
        new_angles[pos_idx] = np.angle(poles[pos_idx]) ** mcadams
        return new_angles

    def _new_poles(old_poles, new_angles):
        return np.abs(old_poles) * np.exp(1j * new_angles)

    def _lpc_ana_syn(old_lpc_coef, new_lpc_coef, data):
        res = scipy.signal.lfilter(old_lpc_coef, np.array(1), data)
        return scipy.signal.lfilter(np.array([1]), new_lpc_coef, res)

    pole_new_angles = np.array([_mcadam_angle(ar_poles[x], mcadams) for x in range(nframe)])
    poles_new = np.array([_new_poles(ar_poles[x], pole_new_angles[x]) for x in range(nframe)])

    recon_frames = [_lpc_ana_syn(lpc_coefs[x], np.real(np.poly(poles_new[x])), windowed_frames[x]) for x in range(nframe)]
    recon_frames = np.stack(recon_frames, axis=0) * win

    # Overlap-Add
    anonymized_data = np.zeros(length_sig)
    overlap_sum = np.zeros(length_sig)

    for i in range(nframe):
        start = i * shift
        end = start + winlen
        anonymized_data[start:end] += recon_frames[i][:min(winlen, length_sig - start)]
        overlap_sum[start:end] += win

    anon_normalized = anonymized_data / (overlap_sum + 1e-8)
    anon_normalized = anon_normalized / np.max(np.abs(anon_normalized)) * 0.99

    return anon_normalized