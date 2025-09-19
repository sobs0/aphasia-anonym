"""
ASR Training Module for Wav2Vec2 Fine-tuning
"""

import os
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
import soundfile as sf
import librosa
from datasets import Dataset  # Removed Audio import
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer
)
from transformers.trainer_utils import IntervalStrategy
from evaluate import load
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator for CTC training with dynamic padding
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they need different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features if len(feature.get("labels", [])) > 0]

        # Process input features (audio)
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Process label features (text) if any exist
        if label_features:
            try:
                labels_batch = self.processor.tokenizer.pad(
                    label_features,
                    padding=self.padding,
                    return_tensors="pt",
                )

                # Replace padding with -100 to ignore in loss calculation
                if "attention_mask" in labels_batch:
                    labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
                else:
                    labels = labels_batch["input_ids"]

                batch["labels"] = labels

            except Exception as e:
                # If text processing fails, create empty labels
                batch_size = batch["input_values"].shape[0]
                batch["labels"] = torch.full((batch_size, 1), -100, dtype=torch.long)
        else:
            # No labels to process
            batch_size = batch["input_values"].shape[0]
            batch["labels"] = torch.full((batch_size, 1), -100, dtype=torch.long)

        return batch


class ASRTrainer:
    """
    ASR Training class for Wav2Vec2 fine-tuning
    """

    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger
        self.model_config = config['model_training']

        # Setup directories
        self.setup_directories()

        # Initialize components
        self.processor = None
        self.model = None
        self.data_collator = None

        # Metrics
        self.wer_metric = load("wer")

        # Get training configuration
        self.audio_type = self.model_config['audio_type']  # 'original' or 'anonymized'
        self.model_suffix = self.model_config['model_suffix']  # same as audio_type
        self.data_percentage = self.model_config['data_percentage']  # 1-100%

        self.logger.info(f"ASR Training Configuration:")
        self.logger.info(f"  - Audio Type: {self.audio_type}")
        self.logger.info(f"  - Model Suffix: {self.model_suffix}")
        self.logger.info(f"  - Data Percentage: {self.data_percentage}%")

        # Validate GPU setup
        self.setup_gpu()

    def setup_directories(self):
        """Setup output directories for training"""
        self.temp_dir = Path(self.config['output']['temporary_data_dir'])
        self.models_dir = self.temp_dir / 'models'
        self.plots_dir = self.temp_dir / 'training_plots'

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Models directory: {self.models_dir}")
        self.logger.info(f"Plots directory: {self.plots_dir}")

    def setup_gpu(self):
        """Setup and validate GPU configuration"""
        self.logger.info("=== GPU Setup for Training ===")

        # Check CUDA availability
        if not torch.cuda.is_available():
            self.logger.error("CUDA not available! Training will be extremely slow on CPU")
            self.logger.error("Please check CUDA installation and GPU drivers")
            raise RuntimeError("CUDA not available for training")

        # Check visible devices
        torch.cuda.set_device(0)
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
        self.logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_devices}")

        # Get device info
        num_gpus = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)

        self.logger.info(f"Available GPUs: {num_gpus}")
        self.logger.info(f"Current device: {current_device} ({device_name})")

        # Test GPU memory allocation
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            memory_allocated = torch.cuda.memory_allocated() / 1024**3 
            del test_tensor
            torch.cuda.empty_cache()
            self.logger.info(f"GPU memory test passed - allocated {memory_allocated:.2f} GB")
        except Exception as e:
            self.logger.error(f"GPU memory test failed: {e}")
            raise RuntimeError(f"GPU memory allocation failed: {e}")

        # Set device for training
        self.device = torch.device(f"cuda:{current_device}")
        self.logger.info(f"Training device: {self.device}")

    def load_stratified_data(self, split_chunks_path: str) -> Dict[str, Dataset]:
        """
        Load data using hierarchical stratified split
        """
        self.logger.info(f"Loading stratified training data for audio type: {self.audio_type}")

        chunks_df = pd.read_csv(split_chunks_path)

        self.logger.info("Fixing data types...")

        numeric_columns = ['audio_length', 'start_time', 'end_time', 'max_pause_ms', 'total_pause_ms', 'mcadams_anonym_value']
        for col in numeric_columns:
            if col in chunks_df.columns:
                chunks_df[col] = pd.to_numeric(chunks_df[col], errors='coerce')
                chunks_df[col] = chunks_df[col].fillna(0)

        string_columns = ['cleaned_transcript', 'use', 'asr_set', 'speaker_id', 'chunk_id', 'recording_id']
        for col in string_columns:
            if col in chunks_df.columns:
                chunks_df[col] = chunks_df[col].fillna('').astype(str)
                chunks_df[col] = chunks_df[col].replace('nan', '')

        # Ensure audio path columns exist
        required_audio_columns = ['silence_padded_chunk_wav_path', 'mcadams_anonymized_silence_padded_chunk_wav_path']
        for col in required_audio_columns:
            if col not in chunks_df.columns:
                self.logger.error(f"Required audio column missing: {col}")
                raise ValueError(f"Missing required column: {col}")

        # Select audio column based on audio_type
        if self.audio_type == 'original':
            audio_column = 'silence_padded_chunk_wav_path'
        elif self.audio_type == 'anonymized':
            audio_column = 'mcadams_anonymized_silence_padded_chunk_wav_path'
        else:
            raise ValueError(f"Invalid audio_type: {self.audio_type}. Must be 'original' or 'anonymized'")

        self.logger.info(f"Using audio column: {audio_column}")

        # Filter valid transcripts
        self.logger.info("Filtering for ASR-suitable chunks with valid transcripts...")

        use_mask = chunks_df['use'].isin(['ONLY_ASR', 'BOTH'])
        transcript_not_na_mask = chunks_df['cleaned_transcript'].notna()
        transcript_not_empty_mask = chunks_df['cleaned_transcript'] != ''
        transcript_not_nan_str_mask = chunks_df['cleaned_transcript'] != 'nan'
        
        # Check transcript length
        transcript_length_mask = chunks_df['cleaned_transcript'].str.len() >= 2
        
        audio_path_not_na_mask = chunks_df[audio_column].notna()
        audio_path_not_empty_mask = chunks_df[audio_column] != ''

        # Combine all filters
        final_mask = (use_mask & transcript_not_na_mask & transcript_not_empty_mask &
                      transcript_not_nan_str_mask & transcript_length_mask &
                      audio_path_not_na_mask & audio_path_not_empty_mask)

        asr_chunks = chunks_df[final_mask].copy()

        self.logger.info(f"Filtering results:")
        self.logger.info(f"  - Total chunks: {len(chunks_df)}")
        self.logger.info(f"  - Use filter (ONLY_ASR/BOTH): {use_mask.sum()}")
        self.logger.info(f"  - Has transcript: {transcript_not_na_mask.sum()}")
        self.logger.info(f"  - Non-empty transcript: {transcript_not_empty_mask.sum()}")
        self.logger.info(f"  - Min transcript length (≥2 chars): {transcript_length_mask.sum()}")
        self.logger.info(f"  - Has audio path: {audio_path_not_na_mask.sum()}")
        self.logger.info(f"  - Final ASR-suitable chunks: {len(asr_chunks)}")

        if len(asr_chunks) == 0:
            raise ValueError("No ASR-suitable chunks found after filtering!")

        # Create datasets for each split
        datasets = {}
        for split in ['train', 'validation', 'test']:
            split_data = asr_chunks[asr_chunks['asr_set'] == split].copy()

            if len(split_data) == 0:
                self.logger.warning(f"No data found for {split} set!")
                continue

            # Apply data percentage reduction for test mode
            if self.data_percentage < 100:
                original_size = len(split_data)
                new_size = max(1, int(len(split_data) * self.data_percentage / 100))
                split_data = split_data.sample(n=new_size, random_state=42).copy()
                self.logger.info(f"{split} set reduced from {original_size} to {new_size} samples ({self.data_percentage}%)")

            # Prepare data for HuggingFace Dataset
            dataset_dict = {
                'audio': split_data[audio_column].tolist(),
                'transcript': split_data['cleaned_transcript'].tolist(),
                'speaker_id': split_data['speaker_id'].tolist(),
                'chunk_id': split_data['chunk_id'].tolist()
            }

            valid_audio_paths = []
            valid_transcripts = []
            valid_speaker_ids = []
            valid_chunk_ids = []

            for i, (audio_path, transcript, speaker_id, chunk_id) in enumerate(zip(
                dataset_dict['audio'], dataset_dict['transcript'],
                dataset_dict['speaker_id'], dataset_dict['chunk_id']
            )):
                # Check audio file exists
                audio_exists = Path(audio_path).exists()
                
                # Check transcript is valid (not empty, has content)
                transcript_clean = str(transcript).strip().upper()
                transcript_valid = len(transcript_clean) >= 2  # At least 2 characters
                
                if audio_exists and transcript_valid:
                    valid_audio_paths.append(audio_path)
                    valid_transcripts.append(transcript_clean)  # Use cleaned version
                    valid_speaker_ids.append(speaker_id)
                    valid_chunk_ids.append(chunk_id)
                else:
                    self.logger.warning(f"Skipping invalid sample {chunk_id}: audio_exists={audio_exists}, transcript_valid={transcript_valid} ('{transcript[:20]}...')")

            if not valid_audio_paths:
                self.logger.error(f"No valid samples found for {split} set!")
                continue

            # Create final dataset dict
            final_dataset_dict = {
                'audio': valid_audio_paths,
                'transcript': valid_transcripts,
                'speaker_id': valid_speaker_ids,
                'chunk_id': valid_chunk_ids
            }

            
            dataset = Dataset.from_dict(final_dataset_dict)

            datasets[split] = dataset
            self.logger.info(f"{split.capitalize()} set: {len(dataset)} valid samples")

            # Log speaker distribution
            speakers_in_split = len(set(valid_speaker_ids))
            self.logger.info(f"{split.capitalize()} speakers: {speakers_in_split}")

        if not datasets:
            raise ValueError("No valid datasets created! Check your data and paths.")

        return datasets

    def setup_processor_pretrained(self):
        """
        Setup processor using pre-trained wav2vec2 vocabulary
        """
        self.logger.info("Setting up processor with pre-trained vocabulary")

        model_name = self.model_config['huggingface_model_name']
        self.logger.info(f"Loading processor from: {model_name}")

        try:
            # Load pre-trained processor directly
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)

            # Validate the processor loaded correctly
            tokenizer_vocab_size = len(self.processor.tokenizer)
            self.logger.info(f"Processor loaded successfully:")
            self.logger.info(f"  - Tokenizer vocabulary size: {tokenizer_vocab_size}")
            self.logger.info(f"  - Feature extractor sampling rate: {self.processor.feature_extractor.sampling_rate}")

            if tokenizer_vocab_size < 20:
                raise ValueError(f"Tokenizer vocabulary too small ({tokenizer_vocab_size}). Expected >20 characters.")

            # Show vocabulary preview
            vocab_items = list(self.processor.tokenizer.get_vocab().items())
            vocab_preview = [f"'{char}':{idx}" for char, idx in sorted(vocab_items, key=lambda x: x[1])[:10]]
            self.logger.info(f"Vocabulary preview: {vocab_preview}")

            # Test tokenizer
            try:
                test_text = "HELLO WORLD"
                test_tokenized = self.processor.tokenizer(test_text, add_special_tokens=False)
                self.logger.info(f"  - Test tokenization '{test_text}' -> {test_tokenized.input_ids}")
            except Exception as test_error:
                self.logger.error(f"  - Test tokenization failed: {test_error}")
                raise

            # Create data collator
            self.data_collator = DataCollatorCTCWithPadding(
                processor=self.processor,
                padding=True
            )

            self.logger.info("Processor and data collator setup completed successfully")

        except Exception as e:
            self.logger.error(f"Failed to setup processor: {str(e)}")
            raise

    def load_audio_file(self, audio_path: str) -> tuple:
        """
        Load audio file with multiple fallbacks
        """
        if not audio_path or not Path(audio_path).exists():
            self.logger.error(f"Audio file not found: {audio_path}")
            return None, None
        
        # Try soundfile
        try:
            audio_array, sampling_rate = sf.read(audio_path)
            
            # Ensure mono
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            
            # Ensure correct sampling rate (16kHz)
            if sampling_rate != 16000:
                audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
                sampling_rate = 16000
            
            # Ensure float32
            audio_array = audio_array.astype(np.float32)
            
            if len(audio_array) > 0:
                return audio_array, sampling_rate
                
        except Exception as e:
            
            self.logger.warning(f"soundfile failed for {audio_path}: {e}")
        
        # Try librosa
        try:
            # Ensure mono and correct sampling rate (16kHz)
            audio_array, sampling_rate = librosa.load(audio_path, sr=16000, mono=True)

            # Ensure float32
            audio_array = audio_array.astype(np.float32)
            
            if len(audio_array) > 0:
                return audio_array, 16000
                
        except Exception as e:
            self.logger.warning(f"librosa failed for {audio_path}: {e}")
        
        # Try torchaudio
        try:
            import torchaudio
            waveform, sample_rate = torchaudio.load(audio_path)
            
             # Ensure mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
             # Ensure correct sampling rate (16kHz)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Ensure float32
            audio_array = waveform.squeeze().numpy().astype(np.float32)
            
            if len(audio_array) > 0:
                return audio_array, 16000
                
        except ImportError:
            self.logger.warning("torchaudio not available")
        except Exception as e:
            self.logger.warning(f"torchaudio failed for {audio_path}: {e}")
        
        self.logger.error(f"All audio loading methods failed for: {audio_path}")
        return None, None

    def prepare_dataset(self, batch):
        """
        Prepare dataset batch for training
        """
        try:
            # Load audio 
            audio_path = batch["audio"] 
            
            audio_array, sampling_rate = self.load_audio_file(audio_path)
            
            if audio_array is None:
                self.logger.error(f"Failed to load audio: {audio_path}")
                return None

            # Process audio with feature extractor
            try:
                processed_audio = self.processor.feature_extractor(
                    audio_array,
                    sampling_rate=sampling_rate,
                    return_tensors="pt",
                    padding=False,
                    max_length=None,
                    truncation=False
                )

                # Extract tensor and remove batch dimension 
                if hasattr(processed_audio, 'input_values'):
                    input_values = processed_audio.input_values
                elif isinstance(processed_audio, dict) and 'input_values' in processed_audio:
                    input_values = processed_audio['input_values']
                else:
                    input_values = processed_audio

                # Ensure correct shape
                if input_values.dim() > 1:
                    input_values = input_values.squeeze(0)

                batch["input_values"] = input_values
                batch["input_length"] = len(input_values)

            except Exception as audio_error:
                self.logger.error(f"Audio processing error for {audio_path}: {audio_error}")
                return None

            # Process transcript with guaranteed valid output
            transcript = batch.get("transcript", "")

            # Convert to string and clean
            transcript_str = str(transcript).strip().upper()
            
            if len(transcript_str) < 2:
                self.logger.error(f"Invalid transcript passed to prepare_dataset: '{transcript_str}'")
                return None

            # Process transcript with tokenizer
            try:
                tokenized = self.processor.tokenizer(
                    transcript_str,
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                    add_special_tokens=False
                )

                # Extract input_ids and remove batch dimension
                if hasattr(tokenized, 'input_ids'):
                    input_ids = tokenized.input_ids.squeeze(0)
                elif isinstance(tokenized, dict) and 'input_ids' in tokenized:
                    input_ids = tokenized['input_ids'].squeeze(0)
                else:
                    input_ids = tokenized

                # Convert to list if tensor
                if torch.is_tensor(input_ids):
                    batch_labels = input_ids.tolist()
                else:
                    batch_labels = input_ids

                # Ensure valid labels
                if not batch_labels or len(batch_labels) == 0:
                    self.logger.warning(f"Empty labels after tokenization for transcript: '{transcript_str}'")
                    return None

                batch["labels"] = batch_labels
                batch["input_ids"] = batch_labels

            except Exception as transcript_error:
                self.logger.error(f"Error processing transcript '{transcript_str}': {transcript_error}")
                return None

            return batch

        except Exception as e:
            self.logger.error(f"Error in prepare_dataset: {str(e)}")
            return None

    def filter_and_prepare_datasets(self, datasets: Dict[str, Dataset]) -> Dict[str, Dataset]:
        """
        Filter datasets and training preparation
        """
        self.logger.info("Preparing datasets for training")

        prepared_datasets = {}

        for split_name, dataset in datasets.items():
            self.logger.info(f"Preparing {split_name} dataset...")

            try:
                prepared_dataset = dataset.map(
                    self.prepare_dataset,
                    remove_columns=dataset.column_names,
                    num_proc=1
                )
                
                original_length = len(prepared_dataset)
                prepared_dataset = prepared_dataset.filter(lambda x: x is not None)
                filtered_length = len(prepared_dataset)
                
                if filtered_length < original_length:
                    self.logger.warning(f"{split_name}: Filtered out {original_length - filtered_length} invalid samples")

                if filtered_length == 0:
                    self.logger.error(f"No valid samples remaining in {split_name} dataset!")
                    continue

                prepared_datasets[split_name] = prepared_dataset
                self.logger.info(f"{split_name} dataset prepared: {filtered_length} valid samples")

            except Exception as e:
                self.logger.error(f"Failed to prepare {split_name} dataset: {str(e)}")
                continue

        if not prepared_datasets:
            raise ValueError("No datasets could be prepared successfully!")

        return prepared_datasets

    def debug_tokenizer_behavior(self):
        """
        Debug function to understand tokenizer behavior
        """
        self.logger.info("=== DEBUGGING TOKENIZER BEHAVIOR ===")
        
        # Test common words
        test_words = ["HELLO", "WORLD", "THE", "AND", "I", "YOU", "A", "TO"]
        
        for word in test_words:
            try:
                # Tokenize
                tokenized = self.processor.tokenizer(word, add_special_tokens=False)
                token_ids = tokenized.input_ids
                
                # Decode back
                decoded = self.processor.tokenizer.decode(token_ids)
                
                self.logger.info(f"Word: '{word}' -> Tokens: {token_ids} -> Decoded: '{decoded}'")
                
                # Check if round-trip works
                if decoded.strip() == word:
                    self.logger.info(f"Round-trip successful")
                else:
                    self.logger.warning(f"Round-trip failed: '{word}' != '{decoded.strip()}'")
                    
            except Exception as e:
                self.logger.error(f"Error tokenizing '{word}': {e}")
        
        # Test pad token behavior
        pad_token = self.processor.tokenizer.pad_token
        pad_token_id = self.processor.tokenizer.pad_token_id
        
        self.logger.info(f"Pad token: '{pad_token}' (ID: {pad_token_id})")
        
        # Test decoding with pad tokens
        test_ids_with_pad = [5, 10, pad_token_id, 15, pad_token_id, pad_token_id]
        decoded_with_pad = self.processor.tokenizer.decode(test_ids_with_pad)
        self.logger.info(f"Test decode with pads: {test_ids_with_pad} -> '{decoded_with_pad}'")
        
        self.logger.info("=== END TOKENIZER DEBUGGING ===")

    def debug_dataset_samples_detailed(self, datasets: Dict[str, Dataset]):
        """
        Enhanced debugging of actual dataset samples
        """
        self.logger.info("=== DETAILED DATASET SAMPLE DEBUGGING ===")
        
        for split_name, dataset in datasets.items():
            if split_name != 'validation':
                continue
                
            self.logger.info(f"Debugging {split_name} dataset samples:")
            
            # Check first 5 samples
            for i in range(min(5, len(dataset))):
                sample = dataset[i]
                
                self.logger.info(f"Sample {i+1}:")
                self.logger.info(f"  Chunk ID: {sample.get('chunk_id', 'N/A')}")
                self.logger.info(f"  Speaker ID: {sample.get('speaker_id', 'N/A')}")
                
                # Check labels
                labels = sample.get('labels', [])
                if labels:
                    self.logger.info(f"  Labels: {labels[:20]}...")
                    
                    # Try to decode labels
                    try:
                        decoded_label = self.processor.tokenizer.decode(labels)
                        self.logger.info(f"Decoded Label: '{decoded_label}'")
                        
                        # Check if there are any issues with the decoded text
                        if not decoded_label.strip():
                            self.logger.warning(f"Decoded label is empty!")
                        elif len(decoded_label.strip()) < 2:
                            self.logger.warning(f"Decoded label very short: '{decoded_label.strip()}'")
                        else:
                            self.logger.info(f"Decoded label looks good")
                            
                    except Exception as e:
                        self.logger.error(f"Failed to decode labels: {e}")
                else:
                    self.logger.warning(f"No labels found!")
                
                # Check input values
                input_values = sample.get('input_values')
                if input_values is not None:
                    if hasattr(input_values, '__len__'):
                        self.logger.info(f"  Input length: {len(input_values)}")
                    else:
                        self.logger.info(f"  Input type: {type(input_values)}")
                else:
                    self.logger.warning(f"  ❌ No input_values found!")
                    
                self.logger.info("")
        
        self.logger.info("=== END DETAILED DATASET SAMPLE DEBUGGING ===")

    def debug_dataset_labels(self, datasets: Dict[str, Dataset]):
        """
        Debug function to check dataset labels for training issues
        """
        self.logger.info("=== DEBUGGING DATASET LABELS ===")

        for split_name, dataset in datasets.items():
            self.logger.info(f"Debugging {split_name} dataset:")

            # Sample first few examples
            sample_size = min(5, len(dataset))
            if sample_size == 0:
                self.logger.warning(f"  {split_name} dataset is empty!")
                continue

            samples = [dataset[i] for i in range(sample_size)]

            empty_labels = 0
            valid_labels = 0

            for i, sample in enumerate(samples):
                labels = sample.get('labels', [])
                input_length = sample.get('input_length', 0)

                if not labels or len(labels) == 0:
                    empty_labels += 1
                    self.logger.warning(f"  Sample {i}: EMPTY LABELS, input_length={input_length}")
                else:
                    valid_labels += 1
                    self.logger.info(f"  Sample {i}: {len(labels)} label tokens, input_length={input_length}")
                    self.logger.info(f"    Labels: {labels[:10]}...")

                    # Decode labels for inspection
                    try:
                        decoded = self.processor.tokenizer.decode(labels)
                        self.logger.info(f"    Decoded: '{decoded[:50]}...'")
                    except:
                        self.logger.warning(f"    Could not decode labels")

            total_samples = len(dataset)
            samples_to_check = min(100, total_samples)

            total_empty = 0
            total_valid = 0

            for i in range(samples_to_check):
                labels = dataset[i].get('labels', [])
                if not labels or len(labels) == 0:
                    total_empty += 1
                else:
                    total_valid += 1

            self.logger.info(f"  {split_name} dataset check ({samples_to_check}/{total_samples} samples):")
            self.logger.info(f"    Valid labels: {total_valid} ({total_valid/samples_to_check*100:.1f}%)")
            self.logger.info(f"Empty labels: {total_empty} ({total_empty/samples_to_check*100:.1f}%)")

            if total_empty > total_valid:
                self.logger.error(f"MORE EMPTY THAN VALID LABELS in {split_name}! This will cause training issues!")
            elif total_empty > 0:
                self.logger.warning(f"Some empty labels in {split_name}, but should be okay")
            else:
                self.logger.info(f"All samples have valid labels in {split_name}")

        self.logger.info("=== END DATASET LABEL DEBUGGING ===")

        # Add detailed sample debugging
        self.debug_dataset_samples_detailed(datasets)

    def setup_model_pretrained(self):
        """
        Setup Wav2Vec2 model using pre-trained vocabulary size
        """
        self.logger.info("Setting up Wav2Vec2 model with pre-trained vocabulary")

        model_name = self.model_config['huggingface_model_name']

        # Load model
        self.model = Wav2Vec2ForCTC.from_pretrained(
            model_name,
            attention_dropout=float(self.model_config['attention_dropout']),
            activation_dropout=float(self.model_config['activation_dropout']),
            hidden_dropout=float(self.model_config['hidden_dropout']),
            feat_proj_dropout=float(self.model_config['feat_proj_dropout']),
            mask_time_prob=float(self.model_config['mask_time_prob']),
            layerdrop=float(self.model_config['layerdrop']),
            ctc_loss_reduction="mean",
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )

        # Freeze feature encoder
        self.model.freeze_feature_encoder()

        # Move model to GPU
        self.model = self.model.to(self.device)

        self.logger.info(f"Model loaded from {model_name}")
        self.logger.info(f"Model vocabulary size: {self.model.config.vocab_size}")
        self.logger.info(f"Model moved to device: {self.device}")
        self.logger.info("Feature encoder frozen, CTC head will be fine-tuned")

    def compute_metrics(self, pred):
        """
        Compute WER
        """
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        self.logger.info("=== CTC PREDICTION ANALYSIS ===")
        
        unique_predicted_tokens = np.unique(pred_ids.flatten())
        token_counts = {token_id: np.sum(pred_ids == token_id) for token_id in unique_predicted_tokens}
        
        self.logger.info(f"Unique predicted token IDs: {unique_predicted_tokens}")
        self.logger.info(f"Token counts: {token_counts}")
        
        # Check if model is only predicting blank tokens
        blank_token_id = self.processor.tokenizer.pad_token_id
        if hasattr(self.model.config, 'pad_token_id'):
            blank_token_id = self.model.config.pad_token_id
        
        self.logger.info(f"Blank/Pad token ID: {blank_token_id}")
        
        total_predictions = pred_ids.size
        blank_predictions = np.sum(pred_ids == blank_token_id)
        blank_percentage = (blank_predictions / total_predictions) * 100
        
        self.logger.info(f"Blank token predictions: {blank_predictions}/{total_predictions} ({blank_percentage:.1f}%)")
        
        # Check logits statistics
        logit_stats = {
            'mean': np.mean(pred_logits),
            'std': np.std(pred_logits),
            'min': np.min(pred_logits),
            'max': np.max(pred_logits)
        }
        self.logger.info(f"Logits statistics: {logit_stats}")
        
        # Check if predictions are too uniform
        logits_entropy = -np.sum(np.exp(pred_logits) * pred_logits, axis=-1).mean()
        self.logger.info(f"Average prediction entropy: {logits_entropy:.4f}")

        # Replace -100 with pad token id
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

        # Decode predictions and labels
        pred_str = self.processor.batch_decode(pred_ids)
        label_str = self.processor.batch_decode(label_ids, group_tokens=False)

        # Log first 10 raw predictions and labels
        self.logger.info("=== EVALUATION DEBUGGING: First 5 Raw Predictions vs Labels ===")
        for i in range(min(5, len(pred_str))):
            sample_pred_ids = pred_ids[i][:50] 
            sample_label_ids = label_ids[i][:50] 
            
            self.logger.info(f"Sample {i+1}:")
            self.logger.info(f"  Pred token IDs: {sample_pred_ids}")
            self.logger.info(f"  Label token IDs: {sample_label_ids}")
            self.logger.info(f"  Raw Prediction: '{pred_str[i]}'")
            self.logger.info(f"  Raw Label:      '{label_str[i]}'")
            self.logger.info(f"  Pred Length:    {len(pred_str[i])}")
            self.logger.info(f"  Label Length:   {len(label_str[i])}")

        # Filter empty strings
        valid_pairs = []
        for i, (p, l) in enumerate(zip(pred_str, label_str)):
            pred_clean = p.strip()
            label_clean = l.strip()

            if len(label_clean) == 0 or label_clean.replace(self.processor.tokenizer.pad_token or '', '').strip() == '':
                if i < 5: 
                    self.logger.warning(f"SKIPPING Sample {i+1}: Empty/invalid label - '{l}'")
                continue
                
            valid_pairs.append((pred_clean, label_clean))

        if not valid_pairs:
            self.logger.warning(f"No valid prediction-label pairs found for WER calculation out of {len(pred_str)} total samples")
            self.logger.warning("This suggests most evaluation samples have empty labels")
            return {"wer": 1.0}

        # Log first 10 valid pairs
        self.logger.info("=== EVALUATION DEBUGGING: First 5 Valid Pairs ===")
        for i in range(min(5, len(valid_pairs))):
            pred_clean, label_clean = valid_pairs[i]
            self.logger.info(f"Valid Pair {i+1}:")
            self.logger.info(f"  Prediction: '{pred_clean}'")
            self.logger.info(f"  Ground Truth: '{label_clean}'")
            self.logger.info(f"  Match: {pred_clean.lower() == label_clean.lower()}")

        # Extract valid predictions and labels
        valid_pred_str = [pair[0] for pair in valid_pairs]
        valid_label_str = [pair[1] for pair in valid_pairs]

        self.logger.info(f"WER calculated on {len(valid_pairs)} valid pairs out of {len(pred_str)} total samples")

        # Calculate WER on valid pairs only
        wer_score = self.wer_metric.compute(predictions=valid_pred_str, references=valid_label_str)
        
        self.logger.info("=== WER ANALYSIS ===")
        self.logger.info(f"Total valid pairs: {len(valid_pairs)}")
        self.logger.info(f"Exact matches: {exact_matches}")
        self.logger.info(f"Empty predictions: {empty_predictions}")
        self.logger.info(f"Non-empty wrong predictions: {len(valid_pairs) - exact_matches - empty_predictions}")
        
        if empty_predictions == len(valid_pairs):
            self.logger.error("ALL PREDICTIONS ARE EMPTY! Model is only predicting blank tokens.")
            self.logger.error("This suggests a training issue - model hasn't learned to output characters yet.")
        elif empty_predictions > len(valid_pairs) * 0.9:
            self.logger.warning(f"{empty_predictions/len(valid_pairs)*100:.1f}% of predictions are empty")
        
        self.logger.info(f"=== FINAL WER SCORE: {wer_score:.4f} ===")

        return {"wer": wer_score}

    def create_training_arguments(self, output_dir: str, train_dataset_size: int) -> TrainingArguments:
        """
        Create training arguments
        """
        # Convert config values to proper types
        batch_size = int(self.model_config['per_device_train_batch_size'])
        epochs = int(self.model_config['epochs'])
        learning_rate = float(self.model_config['learning_rate'])

        self.logger.info(f"Training config:")
        self.logger.info(f"  - batch_size = {batch_size}")
        self.logger.info(f"  - epochs = {epochs}")
        self.logger.info(f"  - learning_rate = {learning_rate}")
        self.logger.info(f"  - train_dataset_size = {train_dataset_size}")

        if learning_rate <= 0:
            raise ValueError(f"Learning rate must be > 0, got: {learning_rate}")

        # Calculate total training steps
        steps_per_epoch = max(1, train_dataset_size // batch_size)
        max_steps = steps_per_epoch * epochs

        warmup_steps = int(self.model_config['warmup_steps'])

        eval_steps = max(50, steps_per_epoch // 4)  # Eval 4 times per epoch
        logging_steps = max(10, steps_per_epoch // 20)  # Log 20 times per epoch
        save_steps = eval_steps

        self.logger.info(f"  - steps_per_epoch = {steps_per_epoch}")
        self.logger.info(f"  - max_steps = {max_steps}")
        self.logger.info(f"  - warmup_steps = {warmup_steps}")
        self.logger.info(f"  - eval_steps = {eval_steps} (adjusted for better monitoring)")
        self.logger.info(f"  - logging_steps = {logging_steps}")

        training_args = TrainingArguments(
            output_dir=output_dir,
            max_steps=max_steps,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=int(self.model_config['gradient_accumulation_steps']),
            learning_rate=learning_rate,
            weight_decay=float(self.model_config['weight_decay']),
            warmup_steps=warmup_steps,
            eval_strategy=IntervalStrategy.STEPS,
            eval_steps=eval_steps,
            logging_strategy=IntervalStrategy.STEPS,
            logging_steps=logging_steps,
            save_strategy=IntervalStrategy.STEPS,
            save_steps=save_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            fp16=False,
            gradient_checkpointing=False,
            group_by_length=True,
            dataloader_pin_memory=True,
            dataloader_num_workers=0,
            skip_memory_metrics=True,
            push_to_hub=False,
            remove_unused_columns=False,
            report_to=[],
        )

        self.logger.info(f"Final TrainingArguments:")
        self.logger.info(f"  - learning_rate: {training_args.learning_rate}")
        self.logger.info(f"  - max_steps: {training_args.max_steps}")
        self.logger.info(f"  - eval_steps: {training_args.eval_steps}")

        return training_args

    def train_model(self, datasets: Dict[str, Dataset]) -> str:
        """
        Train the ASR model
        """
        self.logger.info("Starting ASR model training")

        # Create output directory with audio type suffix and timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_output_dir = self.models_dir / f"wav2vec2-aphasia-{self.model_suffix}-{self.model_config['epochs']}ep-{timestamp}"
        model_output_dir.mkdir(exist_ok=True)

        self.logger.info(f"Model will be saved to: {model_output_dir}")

        # Setup processor and model
        self.setup_processor_pretrained()
        self.setup_model_pretrained()
        self.debug_tokenizer_behavior()

        # Prepare datasets for training
        prepared_datasets = self.filter_and_prepare_datasets(datasets)

        # Debug dataset labels
        self.debug_dataset_labels(prepared_datasets)

        # Create training arguments
        training_args = self.create_training_arguments(
            output_dir=str(model_output_dir),
            train_dataset_size=len(prepared_datasets['train'])
        )

        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            data_collator=self.data_collator,
            args=training_args,
            compute_metrics=self.compute_metrics,
            train_dataset=prepared_datasets['train'],
            eval_dataset=prepared_datasets['validation'],
            tokenizer=self.processor.tokenizer
        )

        # Test a small batch before training
        self.logger.info("Testing trainer with small batch...")
        try:
            # Take first 2 samples and test
            test_batch = [prepared_datasets['train'][i] for i in range(min(2, len(prepared_datasets['train'])))]
            collated = self.data_collator(test_batch)
            self.logger.info(f"Batch test successful - shapes: {collated['input_values'].shape}, {collated['labels'].shape}")
            
            self.logger.info("Testing evaluation on small validation batch...")
            small_val_batch = [prepared_datasets['validation'][i] for i in range(min(4, len(prepared_datasets['validation'])))]
            collated_val = self.data_collator(small_val_batch)
            
            # Move to device
            collated_val = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in collated_val.items()}
            
            with torch.no_grad():
                outputs = self.model(**collated_val)
                logits = outputs.logits
            
            # Create fake pred object
            class FakePred:
                def __init__(self, predictions, label_ids):
                    self.predictions = predictions.cpu().numpy()
                    self.label_ids = label_ids.cpu().numpy()
            
            fake_pred = FakePred(logits, collated_val['labels'])
            metrics = self.compute_metrics(fake_pred)
            
            self.logger.info(f"Small batch evaluation test: {metrics}")
            
        except Exception as e:
            self.logger.error(f"Batch test failed: {e}")
            raise

        # Start training
        self.logger.info("Starting actual training...")
        train_result = trainer.train()
        self.logger.info("Training completed.")

        # Save model and processor
        trainer.save_model(model_output_dir)
        self.processor.save_pretrained(model_output_dir)

        # Log final training metrics
        metrics = train_result.metrics
        self.logger.info(f"Final training metrics: {metrics}")

        # Save training history
        training_history_path = model_output_dir / "training_history.json"
        with open(training_history_path, "w") as f:
            json.dump(trainer.state.log_history, f, indent=4)
        self.logger.info(f"Training history saved to {training_history_path}")

        # Plotting training progress
        self.plot_training_progress(trainer.state.log_history, model_output_dir)

        self.logger.info("================================================================================")
        self.logger.info(f"TRAINING COMPLETED - Best WER: {trainer.state.best_metric:.4f}")
        self.logger.info(f"Model saved to: {model_output_dir}")
        self.logger.info("================================================================================")

        return str(model_output_dir)

    def plot_training_progress(self, log_history: List[Dict[str, Any]], output_dir: Path):
        """
        Plots training and evaluation loss/WER over steps
        """
        self.logger.info("Creating training visualization plots")
        
        # Filter for train and eval logs
        train_logs = [log for log in log_history if 'loss' in log and 'eval_loss' not in log]
        eval_logs = [log for log in log_history if 'eval_loss' in log]

        if not train_logs and not eval_logs:
            self.logger.warning("No training or evaluation logs found for plotting.")
            return

        # Extract data
        steps = [log['step'] for log in train_logs]
        losses = [log['loss'] for log in train_logs]
        learning_rates = [log['learning_rate'] for log in train_logs]

        eval_steps = [log['step'] for log in eval_logs]
        eval_losses = [log['eval_loss'] for log in eval_logs]
        eval_wers = [log['eval_wer'] for log in eval_logs]

        # Create plots
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(3, 1, figsize=(12, 18))
        fig.suptitle('ASR Training Progress', fontsize=16)

        # Plot 1: Training Loss
        axes[0].plot(steps, losses, label='Training Loss', color='skyblue', linewidth=2)
        if eval_losses:
            axes[0].plot(eval_steps, eval_losses, label='Validation Loss', color='salmon', linewidth=2, linestyle='--')
        axes[0].set_title('Loss over Steps')
        axes[0].set_xlabel('Steps')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.6)

        # Plot 2: Learning Rate
        axes[1].plot(steps, learning_rates, label='Learning Rate', color='lightgreen', linewidth=2)
        axes[1].set_title('Learning Rate Schedule')
        axes[1].set_xlabel('Steps')
        axes[1].set_ylabel('Learning Rate')
        axes[1].legend()
        axes[1].grid(True, linestyle='--', alpha=0.6)

        # Plot 3: WER
        if eval_wers:
            axes[2].plot(eval_steps, eval_wers, label='Validation WER', color='gold', linewidth=2)
            axes[2].set_title('Word Error Rate (WER) over Steps')
            axes[2].set_xlabel('Steps')
            axes[2].set_ylabel('WER')
            axes[2].legend()
            axes[2].grid(True, linestyle='--', alpha=0.6)
        else:
            axes[2].set_visible(False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plot_path = output_dir / f"training_progress_{self.model_suffix}.png"
        plt.savefig(plot_path)
        plt.close(fig)
        self.logger.info(f"Training plots saved to {plot_path}")


    def run_training_pipeline(self, split_chunks_path: str) -> str:
        """
        Main method to run the ASR training pipeline
        """
        try:
            self.logger.info("=== Starting ASR Training Pipeline ===")
            self.logger.info(f"Using split chunks data from: {split_chunks_path}")


            # 1. Load Data
            datasets = self.load_stratified_data(split_chunks_path)

            # 2. Train Model
            model_path = self.train_model(datasets)

            self.logger.info("Training completed successfully!")
            self.logger.info(f"Model saved to: {model_path}")

            return model_path

        except Exception as e:
            self.logger.error("================================================================================")
            self.logger.error(f"ASR TRAINING ERROR - FULL TRACEBACK:")
            self.logger.error("================================================================================")
            self.logger.exception(e)
            self.logger.error("================================================================================")
            raise RuntimeError(f"Pipeline failed with error: {e}")


def train_asr_model(config: dict, logger, split_chunks_path: str) -> str:
    """
    Main function for ASR model training
    """
    asr_trainer = ASRTrainer(config, logger)
    model_path = asr_trainer.run_training_pipeline(split_chunks_path)
    return model_path