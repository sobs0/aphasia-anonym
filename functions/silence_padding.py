import pandas as pd
import webrtcvad
from pydub import AudioSegment
from pathlib import Path
from tqdm import tqdm


def create_silence_padded_chunks(config, logger, anonymized_chunks_path):
    """
    Create silence-padded versions of original and anonymized audio chunks for ASR
    """
    logger.info("Starting silence padding creation for ASR training")
    
    # Load chunks data (original and anonymized paths)
    chunks_df = pd.read_csv(anonymized_chunks_path)
    logger.info(f"Processing {len(chunks_df)} chunks for silence padding")
    
    # Create silence-padded directories
    temp_dir = Path(config['output']['temporary_data_dir'])
    silence_padded_orig_dir = temp_dir / 'audio_chunks_silpad'
    silence_padded_anon_dir = temp_dir / 'audio_chunks_mcadams_silpad'
    
    silence_padded_orig_dir.mkdir(parents=True, exist_ok=True)
    silence_padded_anon_dir.mkdir(parents=True, exist_ok=True)
    
    # Get silence padding parameters from config
    silence_config = config['silence_padding']
    target_start_sil_ms = silence_config['target_start_silence_ms']
    target_end_sil_ms = silence_config['target_end_silence_ms']
    vad_mode = silence_config['vad_mode']
    frame_ms = silence_config['frame_ms']
    
    logger.info(f"Silence padding parameters:")
    logger.info(f"  - Start silence: {target_start_sil_ms}ms")
    logger.info(f"  - End silence: {target_end_sil_ms}ms")
    logger.info(f"  - VAD mode: {vad_mode}")
    logger.info(f"  - Frame size: {frame_ms}ms")
    
    # Initialize new columns
    chunks_df['silence_padded_chunk_wav_path'] = None
    chunks_df['mcadams_anonymized_silence_padded_chunk_wav_path'] = None
    
    successful_padding = 0
    failed_padding = 0
    skipped_padding = 0
    
    # Process each chunk
    for idx, row in tqdm(chunks_df.iterrows(), total=len(chunks_df), desc="Creating silence-padded chunks"):
        chunk_id = row['chunk_id']
        original_chunk_path = row['chunk_wav_path']
        anonymized_chunk_path = row['mcadams_anonymized_chunk_wav_path']
        
        # Skip if paths don't exist
        if pd.isna(original_chunk_path) or not Path(original_chunk_path).exists():
            logger.warning(f"Original chunk not found for {chunk_id}: {original_chunk_path}")
            failed_padding += 1
            continue
            
        if pd.isna(anonymized_chunk_path) or not Path(anonymized_chunk_path).exists():
            logger.warning(f"Anonymized chunk not found for {chunk_id}: {anonymized_chunk_path}")
            failed_padding += 1
            continue
        
        # Create filenames
        silence_padded_orig_filename = f"{chunk_id}_silpad.wav"
        silence_padded_anon_filename = f"{chunk_id}_silpad_mca.wav"
        
        silence_padded_orig_path = silence_padded_orig_dir / silence_padded_orig_filename
        silence_padded_anon_path = silence_padded_anon_dir / silence_padded_anon_filename
        
        # Skip if already processed
        if silence_padded_orig_path.exists() and silence_padded_anon_path.exists():
            chunks_df.at[idx, 'silence_padded_chunk_wav_path'] = str(silence_padded_orig_path)
            chunks_df.at[idx, 'mcadams_anonymized_silence_padded_chunk_wav_path'] = str(silence_padded_anon_path)
            skipped_padding += 1
            continue
        
        try:
            # Step 1: Create silence-padded version of original audio chunk
            if not silence_padded_orig_path.exists():
                padded_orig_audio = create_silence_padded_audio(
                    input_path=original_chunk_path,
                    target_start_sil_ms=target_start_sil_ms,
                    target_end_sil_ms=target_end_sil_ms,
                    vad_mode=vad_mode,
                    frame_ms=frame_ms
                )
                padded_orig_audio.export(str(silence_padded_orig_path), format="wav")
            
            # Step 2: Create silence-padded version of anonymized audio chunk
            if not silence_padded_anon_path.exists():
                padded_anon_audio = create_silence_padded_audio(
                    input_path=anonymized_chunk_path,
                    target_start_sil_ms=target_start_sil_ms,
                    target_end_sil_ms=target_end_sil_ms,
                    vad_mode=vad_mode,
                    frame_ms=frame_ms
                )
                padded_anon_audio.export(str(silence_padded_anon_path), format="wav")
            
            # Update dataframe
            chunks_df.at[idx, 'silence_padded_chunk_wav_path'] = str(silence_padded_orig_path)
            chunks_df.at[idx, 'mcadams_anonymized_silence_padded_chunk_wav_path'] = str(silence_padded_anon_path)
            
            successful_padding += 1
            
            if successful_padding % 100 == 0:
                logger.info(f"Created silence-padded chunks for {successful_padding} chunks...")
                
        except Exception as e:
            logger.error(f"Failed to create silence-padded chunks for {chunk_id}: {str(e)}")
            failed_padding += 1
            continue
    
    # Save updated chunks data
    silence_padded_filename = config['processing']['silence_padded_filename']
    output_path = temp_dir / silence_padded_filename
    chunks_df.to_csv(output_path, index=False)
    
    logger.info(f"Silence padding creation completed:")
    logger.info(f"  - Successfully created: {successful_padding}")
    logger.info(f"  - Skipped (already existed): {skipped_padding}")
    logger.info(f"  - Failed: {failed_padding}")
    logger.info(f"Audio directories created:")
    logger.info(f"  - Original silence-padded: {silence_padded_orig_dir}")
    logger.info(f"  - Anonymized silence-padded: {silence_padded_anon_dir}")
    logger.info(f"Silence-padded chunks data saved to: {output_path}")
    
    return str(output_path)


def create_silence_padded_audio(input_path, target_start_sil_ms, target_end_sil_ms, vad_mode=3, frame_ms=30):
    """
    Create silence-padded version of audio chunk
    """
    # Load and prepare audio
    sample_rate = 16000 
    audio = AudioSegment.from_wav(input_path).set_frame_rate(sample_rate).set_channels(1).set_sample_width(2)
    
    # Get speech range using VAD
    start_ms, end_ms = get_speech_range(audio, sample_rate, vad_mode, frame_ms)
    
    if start_ms is None:
        # No speech detected, return original audio with padding
        silence_start = AudioSegment.silent(duration=target_start_sil_ms)
        silence_end = AudioSegment.silent(duration=target_end_sil_ms)
        return silence_start + audio + silence_end
    
    # Calculate required padding
    missing_start_padding = max(0, target_start_sil_ms - start_ms)
    if missing_start_padding > 0:
        silence = AudioSegment.silent(duration=missing_start_padding)
        audio = silence + audio
    
    # Recalculate end position after start padding
    new_end_ms = end_ms + missing_start_padding
    remaining_ms = len(audio) - new_end_ms
    missing_end_padding = max(0, target_end_sil_ms - remaining_ms)
    
    if missing_end_padding > 0:
        silence = AudioSegment.silent(duration=missing_end_padding)
        audio = audio + silence
    
    return audio


def get_speech_range(audio: AudioSegment, sample_rate: int, vad_mode: int, frame_ms: int):
    """
    Detect speech range in audio using WebRTC VAD
    """
    vad = webrtcvad.Vad(vad_mode)
    speech_times = []
    
    for ms in range(0, len(audio) - frame_ms + 1, frame_ms):
        frame = audio[ms:ms + frame_ms]
        if len(frame.raw_data) != int(sample_rate * frame_ms / 1000) * 2:
            continue
        if vad.is_speech(frame.raw_data, sample_rate):
            speech_times.append(ms)
    
    if not speech_times:
        return None, None
    
    return min(speech_times), max(speech_times) + frame_ms