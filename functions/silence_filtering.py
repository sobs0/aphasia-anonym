import pandas as pd
import numpy as np
import webrtcvad
from pydub import AudioSegment
from pathlib import Path
import sys


def filter_silence_outliers(config, logger, chunks_with_audio_path):
    """
    Filter chunks with outlier silence
    """
    logger.info("Starting silence outlier filtering")
    
    # Load chunks data
    chunks_df = pd.read_csv(chunks_with_audio_path)
    logger.info(f"Processing {len(chunks_df)} chunks for silence analysis")
    
    # Get silence parameters from config
    silence_threshold_ms = config['processing'].get('silence_threshold_ms', 2000)
    outlier_factor = config['processing']['silence_outlier_factor']
    
    # Initialize silence columns
    chunks_df['silence_lengths'] = None
    chunks_df['max_pause_ms'] = None
    chunks_df['total_pause_ms'] = None
    
    # Process each chunk for silence detection
    logger.info("Analyzing silence patterns in audio chunks...")
    
    # Collect all individual pauses > threshold
    all_significant_pauses = [] 
    chunk_max_pauses = []
    
    processed_chunks = 0
    failed_chunks = 0
    
    for idx, row in chunks_df.iterrows():
        chunk_wav_path = row['chunk_wav_path']
        chunk_id = row['chunk_id']
        
        if pd.isna(chunk_wav_path) or not Path(chunk_wav_path).exists():
            logger.warning(f"Audio file not found for chunk {chunk_id}: {chunk_wav_path}")
            failed_chunks += 1
            continue
        
        try:
            # Detect pauses in audio chunk (only those > threshold)
            pause_lengths = detect_pauses_over_threshold(
                wav_path=chunk_wav_path,
                threshold_ms=silence_threshold_ms
            )
            
            # Debug: Log for first few chunks
            if processed_chunks < 5:
                logger.info(f"DEBUG chunk {chunk_id}: found {len(pause_lengths)} pauses > {silence_threshold_ms}ms: {pause_lengths}")
            
            # Calculate silence statistics for this chunk
            if pause_lengths:
                max_pause = max(pause_lengths)
                total_pause = sum(pause_lengths)
                silence_lengths_str = ",".join(map(str, pause_lengths))
                
                # Add all significant pauses to list
                all_significant_pauses.extend(pause_lengths)
            else:
                max_pause = 0
                total_pause = 0
                silence_lengths_str = ""
            
            # Store in dataframe
            chunks_df.at[idx, 'silence_lengths'] = silence_lengths_str
            chunks_df.at[idx, 'max_pause_ms'] = max_pause
            chunks_df.at[idx, 'total_pause_ms'] = total_pause
            
            # Collect max pauses for per-chunk analysis
            chunk_max_pauses.append(max_pause)
            
            processed_chunks += 1
            
            if processed_chunks % 1000 == 0:
                logger.info(f"Analyzed silence in {processed_chunks} chunks...")
                
        except Exception as e:
            logger.error(f"Failed to analyze silence for chunk {chunk_id}: {str(e)}")
            failed_chunks += 1
            continue
    
    logger.info(f"Silence analysis completed: {processed_chunks} processed, {failed_chunks} failed")
    logger.info(f"Found {len(all_significant_pauses)} total pauses > {silence_threshold_ms}ms across all chunks")
    
    # Calculate outlier threshold using only significant pauses
    if all_significant_pauses:
        outlier_stats = calculate_outlier_thresholds(all_significant_pauses, method='iqr', factor=outlier_factor)
        outlier_threshold = outlier_stats['outlier_threshold']
        
        # Human-in-the-loop inspection
        logger.info("="*60)
        logger.info("SILENCE ANALYSIS COMPLETE - HUMAN REVIEW REQUIRED")
        logger.info("="*60)
        
        print("\n" + "="*60)
        print("SILENCE ANALYSIS STATISTICS")
        print("="*60)
        print(f"Total chunks analyzed: {len(chunks_df)}")
        print(f"Chunks with significant pauses (> {silence_threshold_ms}ms): {len([p for p in chunk_max_pauses if p > 0])}")
        print(f"Total significant pauses found: {len(all_significant_pauses)}")
        print(f"Silence threshold used: {silence_threshold_ms} ms")
        print(f"IQR outlier factor: {outlier_factor}")
        print()
        print("PAUSE LENGTH STATISTICS (SIGNIFICANT PAUSES ONLY):")
        print(f"  Mean pause length: {outlier_stats['mean']:.0f} ms")
        print(f"  Median pause length: {outlier_stats['median']:.0f} ms")
        print(f"  Standard deviation: {outlier_stats['std']:.0f} ms")
        print(f"  Minimum pause length: {outlier_stats['min']:.0f} ms")
        print(f"  Maximum pause length: {outlier_stats['max']:.0f} ms")
        print()
        print("IQR ANALYSIS (SIGNIFICANT PAUSES ONLY):")
        print(f"  Q1 (25th percentile): {outlier_stats['q1']:.0f} ms")
        print(f"  Q3 (75th percentile): {outlier_stats['q3']:.0f} ms")
        print(f"  IQR: {outlier_stats['iqr']:.0f} ms")
        print(f"  Outlier threshold (Q3 + {outlier_factor}*IQR): {outlier_threshold:.0f} ms")
        print()
        
        # Calculate filtering impact: chunks with max_pause > outlier_threshold
        initial_count = len(chunks_df)
        chunks_to_exclude = len([p for p in chunk_max_pauses if p > outlier_threshold])
        remaining_count = initial_count - chunks_to_exclude
        exclusion_rate = chunks_to_exclude / initial_count * 100
        
        print("FILTERING IMPACT PREVIEW:")
        print(f"  Chunks before filtering: {initial_count}")
        print(f"  Chunks that would be excluded (max pause > {outlier_threshold:.0f}ms): {chunks_to_exclude}")
        print(f"  Chunks that would remain: {remaining_count}")
        print(f"  Exclusion rate: {exclusion_rate:.1f}%")
        print()
        
        # Show distribution of significant pause lengths
        percentiles = [50, 75, 90, 95, 99]
        print("SIGNIFICANT PAUSE DISTRIBUTION:")
        for p in percentiles:
            value = np.percentile(all_significant_pauses, p)
            print(f"  {p}th percentile: {value:.0f} ms")
        print()
        
        # Show examples of outlier pauses
        outlier_pauses = [p for p in all_significant_pauses if p > outlier_threshold]
        if outlier_pauses:
            print("EXAMPLES OF OUTLIER PAUSE LENGTHS (first 10):")
            for i, pause in enumerate(outlier_pauses[:10], 1):
                print(f"  {i}. {pause:.0f} ms")
            if len(outlier_pauses) > 10:
                print(f"  ... and {len(outlier_pauses) - 10} more outlier pauses")
            print()
            print(f"OUTLIER STATISTICS:")
            print(f"  Outlier pauses (> {outlier_threshold:.0f} ms): {len(outlier_pauses)} ({len(outlier_pauses)/len(all_significant_pauses)*100:.1f}% of significant pauses)")
            print(f"  Normal pauses (≤ {outlier_threshold:.0f} ms): {len(all_significant_pauses) - len(outlier_pauses)} ({(len(all_significant_pauses) - len(outlier_pauses))/len(all_significant_pauses)*100:.1f}% of significant pauses)")
        print()
        
        print("="*60)
        print("PLEASE REVIEW THE STATISTICS ABOVE")
        print("="*60)
        print("The filtering will remove chunks with max pause length > {:.0f} ms".format(outlier_threshold))
        print("This represents {:.1f}% of your total chunks.".format(exclusion_rate))
        print("Among chunks with significant pauses, {:.1f}% would be removed.".format(
            len(outlier_pauses)/len([p for p in chunk_max_pauses if p > 0])*100 if len([p for p in chunk_max_pauses if p > 0]) > 0 else 0
        ))
        print()
        
        # HUMAN CONFIRMATION LOOP
        while True:
            response = input("Do you want to proceed with this filtering? (yes/no/adjust): ").strip().lower()
            
            if response in ['yes', 'y']:
                logger.info("User confirmed filtering with current parameters")
                break
            elif response in ['no', 'n']:
                logger.info("User rejected filtering - stopping pipeline")
                print("Filtering cancelled. Pipeline stopped.")
                sys.exit(0)
            elif response in ['adjust', 'a']:
                print(f"\nCurrent outlier factor: {outlier_factor:.1f}")
                try:
                    new_factor = float(input("Enter new outlier factor (e.g., 1.5, 2.0, 2.5): "))
                    if new_factor > 0:
                        outlier_factor = new_factor
                        # Recalculate with new factor
                        outlier_stats = calculate_outlier_thresholds(all_significant_pauses, method='iqr', factor=outlier_factor)
                        outlier_threshold = outlier_stats['outlier_threshold']
                        
                        # Recalculate impact
                        chunks_to_exclude = len([p for p in chunk_max_pauses if p > outlier_threshold])
                        remaining_count = initial_count - chunks_to_exclude
                        exclusion_rate = chunks_to_exclude / initial_count * 100
                        
                        print(f"\nWith factor {outlier_factor}:")
                        print(f"  New threshold: {outlier_threshold:.0f} ms")
                        print(f"  Chunks to exclude: {chunks_to_exclude}")
                        print(f"  Exclusion rate: {exclusion_rate:.1f}%")
                        print()
                    else:
                        print("Invalid factor. Please enter a positive number.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
            else:
                print("Please enter 'yes', 'no', or 'adjust'")
        
        # Apply filtering with confirmed parameters
        logger.info(f"Applying filtering with outlier factor {outlier_factor}, threshold {outlier_threshold:.0f} ms")
        
        filtered_chunks_df = chunks_df[chunks_df['max_pause_ms'] <= outlier_threshold].copy()
        filtered_count = len(filtered_chunks_df)
        excluded_count = initial_count - filtered_count
        
        logger.info(f"Filtering results:")
        logger.info(f"  - Chunks before filtering: {initial_count}")
        logger.info(f"  - Chunks after filtering: {filtered_count}")
        logger.info(f"  - Chunks excluded (outlier silence): {excluded_count}")
        logger.info(f"  - Exclusion rate: {excluded_count/initial_count*100:.1f}%")
        logger.info(f"  - Outlier threshold used: {outlier_threshold:.0f} ms")
        
    else:
        logger.warning("No significant pauses found for outlier calculation")
        filtered_chunks_df = chunks_df.copy()
    
    # Save filtered chunks
    temp_dir = Path(config['output']['temporary_data_dir'])
    filtered_chunks_filename = config['processing']['silence_filtered_filename']
    output_path = temp_dir / filtered_chunks_filename
    
    filtered_chunks_df.to_csv(output_path, index=False)
    logger.info(f"Silence-filtered chunks saved to: {output_path}")
    
    return str(output_path)


def detect_pauses_over_threshold(wav_path, threshold_ms=2000, aggressiveness=2, frame_duration=30):
    """
    Detects pauses > threshold_ms in a WAV file using VAD
    """
    try:
        # Prepare audio: 16kHz Mono PCM
        audio = AudioSegment.from_wav(wav_path).set_channels(1).set_frame_rate(16000)
        samples = np.array(audio.get_array_of_samples())
        pcm_audio = samples.tobytes()
        
        audio_duration_ms = len(audio)
        if audio_duration_ms < 500:
            return []
        
        vad = webrtcvad.Vad(aggressiveness)
        sample_rate = 16000
        frame_len = int(sample_rate * frame_duration / 1000)
        n_frames = len(pcm_audio) // (2 * frame_len)
        
        if n_frames < 2:
            return []
        
        speech_segments = []
        
        for i in range(n_frames):
            start_byte = i * frame_len * 2
            end_byte = start_byte + frame_len * 2
            frame = pcm_audio[start_byte:end_byte]
            timestamp_ms = i * frame_duration
            
            if len(frame) < frame_len * 2:
                break
                
            try:
                is_speech = vad.is_speech(frame, sample_rate)
                if is_speech:
                    speech_segments.append(timestamp_ms)
            except Exception as vad_e:
                # VAD failed for this frame, skip
                continue
        
        if len(speech_segments) < 2:
            return []
        
        # Calculate pauses (only above threshold)
        pauses = []
        for i in range(1, len(speech_segments)):
            gap = speech_segments[i] - speech_segments[i - 1] - frame_duration
            if gap > threshold_ms:
                pauses.append(gap)
        
        return pauses
        
    except Exception as e:
        return []


def calculate_outlier_thresholds(pause_lengths, method='iqr', factor=1.5):
    """
    Calculates outlier thresholds for pause lengths
    """
    if not pause_lengths:
        return {"error": "No pause lengths found"}
    
    pauses = np.array(pause_lengths)
    
    stats = {
        'count': len(pauses),
        'mean': np.mean(pauses),
        'median': np.median(pauses),
        'std': np.std(pauses),
        'min': np.min(pauses),
        'max': np.max(pauses)
    }
    
    if method == 'iqr':
        q1 = np.percentile(pauses, 25)
        q3 = np.percentile(pauses, 75)
        iqr = q3 - q1
        
        # IQR method: Everything outside of factor×IQR are outliers
        outlier_threshold = q3 + factor * iqr
        
        stats.update({
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'outlier_threshold': outlier_threshold,
            'method': f'IQR ({factor}×IQR)',
            'factor': factor
        })
        
    elif method == 'zscore':
        mean = np.mean(pauses)
        std = np.std(pauses)
        
        # Z-Score > 2.5 as outliers
        outlier_threshold = mean + 2.5 * std
        
        stats.update({
            'outlier_threshold': outlier_threshold,
            'method': 'Z-Score (2.5σ)'
        })
        
    elif method == 'percentile':
        # 95th percentile as outlier threshold
        outlier_threshold = np.percentile(pauses, 95)
        
        stats.update({
            'outlier_threshold': outlier_threshold,
            'method': 'Percentile (95%)'
        })
    
    return stats