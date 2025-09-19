"""
ASR Evaluation Module
"""

import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import soundfile as sf
import librosa
from datasets import Dataset
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    Trainer,
    TrainingArguments
)
from evaluate import load
from tqdm.auto import tqdm
import warnings
from dataclasses import dataclass
warnings.filterwarnings("ignore")

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator for CTC evaluation with dynamic padding
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they need different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features if len(feature.get("labels", [])) > 0]

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
                batch_size = batch["input_values"].shape[0]
                batch["labels"] = torch.full((batch_size, 1), -100, dtype=torch.long)
        else:
            batch_size = batch["input_values"].shape[0]
            batch["labels"] = torch.full((batch_size, 1), -100, dtype=torch.long)

        return batch

class ASREvaluator:
    """
    ASR Model Evaluator class
    """
    
    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger
        self.eval_config = config['asr_evaluation']
        
        # Setup directories
        self.setup_directories()
        
        # Initialize components
        self.model = None
        self.processor = None
        self.data_collator = None
        
        # Metrics
        self.wer_metric = load("wer")
        
        # Results storage
        self.evaluation_results = []
        
        # Setup GPU
        self.setup_gpu()
    
    def setup_directories(self):
        """Setup output directories for evaluation"""
        self.temp_dir = Path(self.config['output']['temporary_data_dir'])
        self.final_dir = Path(self.config['output']['final_data_dir'])
        self.results_dir = self.temp_dir / 'asr_evaluation_results'
        
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Evaluation results directory: {self.results_dir}")
    
    def setup_gpu(self):
        """Setup and validate GPU configuration for evaluation"""
        self.logger.info("=== GPU Setup for ASR Evaluation ===")
        
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            self.device = torch.device(f"cuda:{current_device}")
            
            self.logger.info(f"Using GPU: {current_device} ({device_name})")
            
            # Test GPU memory
            try:
                test_tensor = torch.randn(1000, 1000).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                self.logger.info("GPU memory test passed for evaluation")
            except Exception as e:
                self.logger.warning(f"GPU memory test failed: {e}")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
            self.logger.warning("CUDA not available - evaluation will run on CPU")
        
        self.logger.info(f"Evaluation device: {self.device}")
    
    def load_trained_model(self, model_path: str):
        """
        Load specified ASR model and processor
        """
        self.logger.info(f"Loading model for evaluation: {model_path}")
        
        # Check if it's a local path or HuggingFace model name
        if Path(model_path).exists():
            # Local trained model
            self.logger.info(f"Loading local model from: {model_path}")
            model_dir = Path(model_path)
            
            # Load processor
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            
            # Load model
            self.model = Wav2Vec2ForCTC.from_pretrained(model_dir)
            
            # Get model metadata if available
            history_path = model_dir / 'training_history.json'
            if history_path.exists():
                with open(history_path, 'r') as f:
                    history = json.load(f)
                    metadata = history.get('metadata', {})
                    self.logger.info(f"Model metadata:")
                    self.logger.info(f"  - Audio type: {metadata.get('audio_type', 'unknown')}")
                    self.logger.info(f"  - Data percentage: {metadata.get('data_percentage', 'unknown')}%")
            
        else:
            # HuggingFace model name
            self.logger.info(f"Loading HuggingFace model: {model_path}")
            
            # Load processor
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            
            # Load model
            self.model = Wav2Vec2ForCTC.from_pretrained(model_path)
        
        # Set to evaluation mode
        self.model.eval()
        
        # Move to device
        self.model = self.model.to(self.device)

        # Create data collator
        self.data_collator = DataCollatorCTCWithPadding(
            processor=self.processor,
            padding=True
        )
        
        self.logger.info("Model and processor loaded successfully")
        self.logger.info(f"Model moved to device: {self.device}")
    
    def load_test_dataset(self, split_chunks_path: str) -> Dataset:
        """
        Load only the test dataset for evaluation
        """
        self.logger.info("Loading test dataset for evaluation")
        
        # Load chunks with stratified splits
        chunks_df = pd.read_csv(split_chunks_path)
        
        # Filter for ASR-suitable test chunks
        test_chunks = chunks_df[
            (chunks_df['asr_set'] == 'test') &
            chunks_df['use'].isin(['ONLY_ASR', 'BOTH']) & 
            chunks_df['cleaned_transcript'].notna() &
            (chunks_df['cleaned_transcript'] != '') &
            (chunks_df['cleaned_transcript'] != 'nan')
        ].copy()
        
        self.logger.info(f"Test chunks found: {len(test_chunks)}")
        
        if len(test_chunks) == 0:
            raise ValueError("No test chunks found for evaluation!")
        
        # Determine which audio column to use based on the model being evaluated
        model_to_evaluate = self.eval_config.get('model_to_evaluate', 'pretrained')
        
        if 'anonymized' in model_to_evaluate.lower():
            # Use anonymized audio for anonymized models
            audio_column = 'mcadams_anonymized_chunk_wav_path'
            self.logger.info("Using anonymized audio for evaluation (model appears to be trained on anonymized data)")
        else:
            # Use original audio for original/pretrained models
            audio_column = 'chunk_wav_path'
            self.logger.info("Using original audio for evaluation")
        
        # Check if the audio column exists
        if audio_column not in test_chunks.columns:
            self.logger.error(f"Required audio column not found: {audio_column}")
            raise ValueError(f"Missing audio column: {audio_column}")
        
        # Filter for valid audio paths
        valid_audio_mask = (test_chunks[audio_column].notna() & 
                           (test_chunks[audio_column] != '') & 
                           (test_chunks[audio_column] != 'nan'))
        
        test_chunks = test_chunks[valid_audio_mask].copy()
        
        self.logger.info(f"Test chunks after audio path filtering: {len(test_chunks)}")
        
        # Prepare data for HuggingFace Dataset - validate paths exist
        valid_samples = []
        
        for _, row in test_chunks.iterrows():
            audio_path = row[audio_column]
            transcript = row['cleaned_transcript']
            
            if Path(audio_path).exists() and transcript.strip():
                valid_samples.append({
                    'audio': audio_path,
                    'transcript': transcript.strip(),
                    'speaker_id': row['speaker_id'],
                    'chunk_id': row['chunk_id']
                })
            else:
                self.logger.warning(f"Skipping invalid sample: {row['chunk_id']} "
                                  f"(audio_exists: {Path(audio_path).exists()}, "
                                  f"transcript_valid: {bool(transcript.strip())})")
        
        if not valid_samples:
            raise ValueError("No valid samples found for test dataset!")
        
        # Create dataset dictionary
        dataset_dict = {
            'audio': [s['audio'] for s in valid_samples],
            'transcript': [s['transcript'] for s in valid_samples],
            'speaker_id': [s['speaker_id'] for s in valid_samples],
            'chunk_id': [s['chunk_id'] for s in valid_samples]
        }
        
        # Create HuggingFace Dataset
        dataset = Dataset.from_dict(dataset_dict)
        
        # Log test set statistics
        speakers_in_test = len(set(dataset_dict['speaker_id']))
        self.logger.info(f"Test set statistics:")
        self.logger.info(f"  - Speakers: {speakers_in_test}")
        self.logger.info(f"  - Samples: {len(dataset)}")
        self.logger.info(f"  - Audio column used: {audio_column}")
        
        return dataset
        
    def load_audio_file(self, audio_path: str) -> tuple:
        """
        Load audio file with fallbacks
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
        Prepare dataset batch
        """
        try:
            audio_path = batch["audio"]
            
            # Load audio
            audio_array, sampling_rate = self.load_audio_file(audio_path)
            
            if audio_array is None:
                self.logger.error(f"Failed to load audio: {audio_path}")
                return None
            
            # Process audio
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
                self.logger.error(f"Audio processing error: {audio_error}")
                raise
            
            # Process transcript 
            transcript = batch.get("transcript", "")
            
            # Handle different transcript types robustly
            if transcript is None or transcript == "" or str(transcript).lower() in ['nan', 'none']:
                batch["labels"] = []
            else:
                # Convert to string and clean
                transcript_str = str(transcript).strip().upper()
                if len(transcript_str) == 0:
                    batch["labels"] = []
                else:
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
                        
                        if torch.is_tensor(input_ids):
                            batch["labels"] = input_ids.tolist()
                        else:
                            batch["labels"] = input_ids
                            
                    except Exception as transcript_error:
                        self.logger.warning(f"Error processing transcript '{transcript_str}': {transcript_error}")
                        batch["labels"] = []
            
            return batch
            
        except Exception as e:
            self.logger.error(f"Error in prepare_dataset: {str(e)}")
            return {
                "input_values": torch.zeros(32000, dtype=torch.float32),
                "input_length": 32000,
                "labels": []
            }
    
    def prepare_test_dataset(self, test_dataset: Dataset) -> Dataset:
        """
        Prepare test dataset for ASR evaluation
        """
        self.logger.info("Preparing test dataset for evaluation")
        
        # Filter for minimum length
        def is_long_enough(batch):
            audio_path = batch["audio"] 
            try:
                audio_array, sampling_rate = self.load_audio_file(audio_path)
                if audio_array is None:
                    return False
                return len(audio_array) > int(1.0 * 16000)
            except:
                return False
        
        filtered_dataset = test_dataset.filter(is_long_enough)
        self.logger.info(f"Test set: {len(test_dataset)} -> {len(filtered_dataset)} after length filtering")
        
        prepared_dataset = filtered_dataset.map(
            self.prepare_dataset, 
            remove_columns=filtered_dataset.column_names,
            num_proc=1
        )
        
        self.logger.info(f"Test dataset prepared: {len(prepared_dataset)} samples")
        return prepared_dataset

    

    def prepare_bootstrap_dataset(self, bootstrap_dataset: Dataset) -> Dataset:
        """
        Prepare bootstrap dataset for ASR Evaluation
        """
        
        def is_long_enough(batch):
            audio_path = batch["audio"]
            try:
                audio_array, sampling_rate = self.load_audio_file(audio_path)
                if audio_array is None:
                    return False
                return len(audio_array) > int(1.0 * 16000)
            except:
                return False
        
        filtered_dataset = bootstrap_dataset.filter(is_long_enough)
        self.logger.debug(f"Bootstrap dataset: {len(bootstrap_dataset)} -> {len(filtered_dataset)} after length filtering")
        
        return filtered_dataset
    

    def bootstrap_wer_analysis(self, test_dataset: Dataset, n_bootstrap: int = 1000) -> Dict[str, Any]:
        """
        Bootstrap resampling (speaker based) for WER comparison
        """
        self.logger.info(f"Starting bootstrap WER analysis with {n_bootstrap} iterations")
        
        # Extract speakers from test dataset
        try:
            first_sample = test_dataset[0]
            if 'speaker_id' not in first_sample:
                raise KeyError("speaker_id not found in dataset")
            
            speakers_in_test = list(set([sample['speaker_id'] for sample in test_dataset]))
            self.logger.info(f"Found {len(speakers_in_test)} unique speakers in test set")
            
        except Exception as e:
            self.logger.error(f"Cannot access speaker_id from dataset: {e}")
            self.logger.error(f"Available keys in first sample: {list(first_sample.keys()) if 'first_sample' in locals() else 'Cannot access sample'}")
            raise
        
        # Group samples by speaker
        speaker_samples = {}
        for i, sample in enumerate(test_dataset):
            speaker_id = sample['speaker_id']
            if speaker_id not in speaker_samples:
                speaker_samples[speaker_id] = []
            speaker_samples[speaker_id].append(i)
        
        self.logger.info(f"Speaker sample distribution: {[(s, len(indices)) for s, indices in list(speaker_samples.items())[:5]]}...")
        
        bootstrap_results = []
        
        with tqdm(range(n_bootstrap), desc="Bootstrap WER Analysis", unit="iteration") as progress_bar:
            for bootstrap_iter in progress_bar:
                try:
                    # 1. Resample speakers with replacement
                    resampled_speakers = np.random.choice(
                        speakers_in_test, 
                        size=len(speakers_in_test), 
                        replace=True
                    )
                    
                    if bootstrap_iter < 3:
                        from collections import Counter
                        speaker_counts = Counter(resampled_speakers)
                        missing_speakers = set(speakers_in_test) - set(resampled_speakers)
                        overrep_speakers = {s: count for s, count in speaker_counts.items() if count > 1}
                        
                        self.logger.info(f"Bootstrap {bootstrap_iter + 1}:")
                        self.logger.info(f"  Missing speakers ({len(missing_speakers)}): {list(missing_speakers)[:3]}...")
                        self.logger.info(f"  Overrepresented speakers: {dict(list(overrep_speakers.items())[:3])}")
                        self.logger.info(f"  Unique speakers: {len(set(resampled_speakers))}/{len(speakers_in_test)}")
                    
                    # 2. Collect all sample indices from resampled speakers
                    bootstrap_indices = []
                    for speaker in resampled_speakers:
                        bootstrap_indices.extend(speaker_samples[speaker])
                
                    bootstrap_dataset = test_dataset.select(bootstrap_indices)
                    
                    # 3. Prepare the bootstrap dataset
                    prepared_bootstrap_dataset = self.prepare_bootstrap_dataset(bootstrap_dataset)
                    
                    # 4. Evaluate WER on this bootstrap sample
                    wer_score = self.evaluate_bootstrap_sample(prepared_bootstrap_dataset, bootstrap_iter + 1)
                    
                    if wer_score is not None:
                        bootstrap_results.append({
                            'bootstrap_iteration': bootstrap_iter + 1,
                            'wer': wer_score,
                            'num_samples': len(prepared_bootstrap_dataset),
                            'num_original_samples': len(bootstrap_dataset),
                            'num_speakers': len(set(resampled_speakers)),
                            'num_unique_speakers': len(set(resampled_speakers))
                        })
                        
                        current_mean = np.mean([r['wer'] for r in bootstrap_results])
                        progress_bar.set_postfix({
                            'Current WER': f"{wer_score:.4f}",
                            'Mean WER': f"{current_mean:.4f}"
                        })
                    
                except Exception as e:
                    self.logger.warning(f"Bootstrap iteration {bootstrap_iter + 1} failed: {str(e)}")
                    import traceback
                    self.logger.warning(f"Traceback: {traceback.format_exc()}")
                    continue
        
        # Calculate bootstrap statistics
        if not bootstrap_results:
            raise RuntimeError("All bootstrap iterations failed!")
        
        wer_scores = [r['wer'] for r in bootstrap_results]
        
        bootstrap_statistics = {
            'method': 'bootstrap_speaker_level',
            'model_evaluated': self.eval_config.get('model_to_evaluate', 'unknown'),
            'n_bootstrap_iterations': len(bootstrap_results),
            'n_bootstrap_attempted': n_bootstrap,
            'success_rate': len(bootstrap_results) / n_bootstrap,
            'original_speakers': len(speakers_in_test),
            'wer_mean': np.mean(wer_scores),
            'wer_std': np.std(wer_scores),
            'wer_min': np.min(wer_scores),
            'wer_max': np.max(wer_scores),
            'wer_median': np.median(wer_scores),
            'wer_25th_percentile': np.percentile(wer_scores, 25),
            'wer_75th_percentile': np.percentile(wer_scores, 75),
            'confidence_interval_95': np.percentile(wer_scores, [2.5, 97.5]),
            'evaluation_date': datetime.now().isoformat()
        }
        
        return {
            'individual_bootstrap_results': bootstrap_results,
            'bootstrap_statistics': bootstrap_statistics
        }


    def evaluate_bootstrap_sample(self, bootstrap_dataset: Dataset, iteration_num: int) -> float:
        """
        Evaluate WER on single bootstrap sample
        """
        try:
            prepared_dataset = bootstrap_dataset.map(
                self.prepare_dataset, 
                remove_columns=bootstrap_dataset.column_names,
                num_proc=1
            )
            
            # Use small batch size for bootstrap
            batch_size = 1 if self.device.type == 'cpu' else 2
            
            training_args = TrainingArguments(
                output_dir=str(self.results_dir / f"temp_bootstrap_{iteration_num}"),
                per_device_eval_batch_size=batch_size,
                remove_unused_columns=False,
                dataloader_num_workers=0,
                dataloader_pin_memory=False,
                fp16=False,
                report_to=[],
                logging_steps=9999,
            )
            
            trainer = Trainer(
                model=self.model,
                data_collator=self.data_collator,
                args=training_args,
                processing_class=self.processor.feature_extractor,
                compute_metrics=self.compute_metrics
            )
            
            # Quick evaluation
            eval_results = trainer.evaluate(eval_dataset=prepared_dataset)
            wer = None
            for wer_key in ['eval_wer', 'wer', 'test_wer']:
                if wer_key in eval_results:
                    wer = eval_results[wer_key]
                    break
            
            if wer is None:
                self.logger.warning(f"No WER found in bootstrap iteration {iteration_num}")
                return None
            
            # Cleanup
            del trainer
            if torch.cuda.is_available() and self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            import shutil
            temp_dir = self.results_dir / f"temp_bootstrap_{iteration_num}"
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            
            return float(wer)
            
        except Exception as e:
            self.logger.error(f"Bootstrap sample evaluation failed: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None


    
    
    def evaluate_single_run(self, test_dataset: Dataset, run_number: int) -> Dict[str, Any]:
        """
        Perform single ASR evaluation run
        """
        # Clear memory
        if torch.cuda.is_available() and self.device.type == 'cuda':
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        dataset_size = len(test_dataset) 
        self.last_individual_predictions = []
        
        try:
            with torch.no_grad():
                if self.device.type == 'cpu':
                    batch_size = 8  
                    num_workers = 2  
                else:
                    batch_size = 2  
                    num_workers = 0  
                
                training_args = TrainingArguments(
                    output_dir=str(self.results_dir / f"temp_run_{run_number}"),
                    per_device_eval_batch_size=batch_size,
                    remove_unused_columns=False,
                    dataloader_num_workers=num_workers,
                    dataloader_pin_memory=False,
                    fp16=False,
                    report_to=[],
                    logging_steps=9999,
                )
                
                trainer = Trainer(
                    model=self.model,
                    data_collator=self.data_collator,
                    args=training_args,
                    processing_class=self.processor.feature_extractor,
                    compute_metrics=self.compute_metrics
                )
                
                self.logger.info(f"Run {run_number}: Starting evaluation of {dataset_size} samples")
                
                # Run evaluation
                eval_results = trainer.evaluate(eval_dataset=test_dataset)
                
                self.logger.info(f"Run {run_number}: Evaluation completed")
                self.logger.debug(f"Run {run_number}: Available keys: {list(eval_results.keys())}")
                
                wer = None
                loss = None
                
                for wer_key in ['eval_wer', 'wer', 'test_wer']:
                    if wer_key in eval_results:
                        wer = eval_results[wer_key]
                        break
                
                for loss_key in ['eval_loss', 'loss', 'test_loss']:
                    if loss_key in eval_results:
                        loss = eval_results[loss_key]
                        break
                
                # Validate extracted values
                if wer is None:
                    self.logger.error(f"Run {run_number}: No WER found in results! Keys: {list(eval_results.keys())}")
                    wer = 1.0 
                
                if loss is None:
                    self.logger.error(f"Run {run_number}: No loss found in results! Keys: {list(eval_results.keys())}")
                    loss = float('inf') 

                individual_results = []
                if hasattr(self, 'last_individual_predictions') and self.last_individual_predictions:
                    for i, pred_data in enumerate(self.last_individual_predictions):
                        # Get corresponding sample info from dataset
                        if i < len(test_dataset):
                            sample = test_dataset[i]
                            individual_results.append({
                                'run_number': run_number,
                                'sample_index': i,
                                'chunk_id': sample.get('chunk_id', f'unknown_{i}'),
                                'speaker_id': sample.get('speaker_id', 'unknown'),
                                'prediction': pred_data['prediction'],
                                'reference': pred_data['reference'],
                                'individual_wer': pred_data['individual_wer'],
                                'timestamp': datetime.now().isoformat()
                            })
                
                results = {
                    'run_number': run_number,
                    'wer': float(wer),
                    'loss': float(loss),
                    'samples': dataset_size,
                    'steps': eval_results.get('eval_steps_per_second', 0.0),
                    'device': str(self.device),
                    'timestamp': datetime.now().isoformat(),
                    'successful': True,  
                    'eval_keys': list(eval_results.keys()), 
                    'individual_predictions': individual_results 
                }
                
                if results['wer'] > 2.0:
                    self.logger.warning(f"Run {run_number}: Suspicious WER: {results['wer']:.4f}")
                
                self.logger.info(f"Run {run_number}: SUCCESS - WER={results['wer']:.4f}, Loss={results['loss']:.4f}, Samples={results['samples']}")
                
                # Cleanup
                del trainer
                if torch.cuda.is_available() and self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                import shutil
                temp_dir = self.results_dir / f"temp_run_{run_number}"
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                
                return results
                
        except Exception as e:
            self.logger.error(f"Run {run_number}: FAILED with error: {str(e)}")
            
            return {
                'run_number': run_number,
                'wer': 1.0,
                'loss': float('inf'),
                'samples': dataset_size,
                'steps': 0.0,
                'device': str(self.device),
                'timestamp': datetime.now().isoformat(),
                'successful': False, 
                'error': str(e)
            }
        finally:
            # Always cleanup
            if torch.cuda.is_available() and self.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
    
    def compute_metrics(self, pred):
        """
        Compute WER
        """
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        
        # Replace -100 with pad token id
        pred.label_ids[pred.label_ids == -100] = self.processor.tokenizer.pad_token_id
        
        # Decode predictions and labels
        pred_str = self.processor.batch_decode(pred_ids)
        label_str = self.processor.batch_decode(pred.label_ids, group_tokens=False)
        
        individual_predictions = []
        
        # Filter empty strings
        valid_pairs = []
        for i, (p, l) in enumerate(zip(pred_str, label_str)):
            pred_clean = p.strip()
            label_clean = l.strip()
            
            # Calculate individual WER
            if pred_clean and label_clean:
                individual_wer = self.wer_metric.compute(predictions=[pred_clean], references=[label_clean])
                valid_pairs.append((pred_clean, label_clean))
            else:
                individual_wer = 1.0 
            
            individual_predictions.append({
                'sample_index': i,
                'prediction': pred_clean,
                'reference': label_clean,
                'individual_wer': individual_wer
            })
        
        self.last_individual_predictions = individual_predictions
        
        if not valid_pairs:
            self.logger.warning("No valid prediction-label pairs found for WER calculation")
            return {"wer": 1.0}
        
        # Extract valid predictions and labels
        valid_pred_str = [pair[0] for pair in valid_pairs]
        valid_label_str = [pair[1] for pair in valid_pairs]
        
        wer_score = self.wer_metric.compute(predictions=valid_pred_str, references=valid_label_str)
        
        return {"wer": wer_score}
    
    def run_multiple_evaluations(self, test_dataset: Dataset) -> Dict[str, Any]:
        """
        Run multiple ASR evaluation runs
        """
        num_runs = self.eval_config['num_evaluation_runs']
        model_name = self.eval_config.get('model_to_evaluate', 'unknown')
        
        self.logger.info(f"Starting {num_runs} evaluation runs on test set")
        self.logger.info(f"Model being evaluated: {model_name}")
        
        all_results = []
        
        with tqdm(range(num_runs), desc=f"Evaluating {model_name}", unit="run") as progress_bar:
            for run_num in progress_bar:
                torch.manual_seed(42 + run_num)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(42 + run_num)
                
                # Run evaluation
                result = self.evaluate_single_run(test_dataset, run_num + 1)
                all_results.append(result)
                
                if result['successful']:
                    progress_bar.set_postfix({
                        'WER': f"{result['wer']:.4f}",
                        'Status': 'OK'
                    })
                else:
                    progress_bar.set_postfix({
                        'Status': 'FAILED'
                    })
        
        successful_results = [r for r in all_results if r.get('successful', False)]
        failed_results = [r for r in all_results if not r.get('successful', False)]
        
        self.logger.info(f"Evaluation completed: {len(successful_results)} successful, {len(failed_results)} failed")
        
        if not successful_results:
            self.logger.error("All evaluation runs failed!")
            for i, failed_result in enumerate(failed_results):
                self.logger.error(f"Run {i+1} error: {failed_result.get('error', 'Unknown error')}")
            raise RuntimeError("All evaluation runs failed - cannot compute statistics")
        
        if len(successful_results) < num_runs:
            self.logger.warning(f"Only {len(successful_results)}/{num_runs} runs succeeded")
        
        wer_scores = [r['wer'] for r in successful_results]
        
        # Validate WER score
        wer_scores = [w for w in wer_scores if not np.isnan(w) and not np.isinf(w)]
        
        if not wer_scores:
            raise RuntimeError("No valid WER scores found")
        
        statistics = {
            'model_evaluated': model_name,
            'num_runs': len(successful_results),
            'num_attempted_runs': num_runs,
            'num_failed_runs': len(failed_results),
            'success_rate': len(successful_results) / num_runs,
            'num_samples': successful_results[0]['samples'],
            'wer_mean': np.mean(wer_scores),
            'wer_std': np.std(wer_scores) if len(wer_scores) > 1 else 0.0,
            'wer_min': np.min(wer_scores),
            'wer_max': np.max(wer_scores),
            'wer_median': np.median(wer_scores),
            'wer_25th_percentile': np.percentile(wer_scores, 25) if len(wer_scores) > 1 else wer_scores[0],
            'wer_75th_percentile': np.percentile(wer_scores, 75) if len(wer_scores) > 1 else wer_scores[0],
            'evaluation_date': datetime.now().isoformat()
        }
        
        self.logger.info(f"Final statistics: WER = {statistics['wer_mean']:.4f} ± {statistics['wer_std']:.4f}")
        
        return {
            'individual_results': successful_results,
            'failed_results': failed_results,
            'statistics': statistics
        }
    
    def save_results_csv(self, results: Dict[str, Any]):
        """
        Save detailed results to CSV file
        """
        model_name = results['statistics'].get('model_evaluated', 'unknown')
        model_safe_name = model_name.replace('/', '_').replace(':', '_')
        evaluation_method = results.get('evaluation_method', 'unknown')
        
        if evaluation_method == 'bootstrap_speaker_level':
            # Save bootstrap results
            bootstrap_results = results['bootstrap_results']['individual_bootstrap_results']
            bootstrap_df = pd.DataFrame(bootstrap_results)
            bootstrap_df['model_evaluated'] = model_name
            bootstrap_df['evaluation_method'] = evaluation_method
            
            bootstrap_path = self.final_dir / f'asr_bootstrap_results_{model_safe_name}.csv'
            bootstrap_df.to_csv(bootstrap_path, index=False)
            
            self.logger.info(f"Bootstrap results saved to: {bootstrap_path}")
            self.logger.info(f"Total bootstrap iterations: {len(bootstrap_results)}")
            
            # Save standard results if available
            if 'standard_results' in results:
                standard_results = results['standard_results']['individual_results']
                standard_df = pd.DataFrame(standard_results)
                standard_df['model_evaluated'] = model_name
                standard_df['evaluation_method'] = 'standard_comparison'
                
                standard_path = self.final_dir / f'asr_standard_results_{model_safe_name}.csv'
                standard_df.to_csv(standard_path, index=False)
                
                self.logger.info(f"Standard comparison results saved to: {standard_path}")
        
        else:
            individual_results = results['individual_results']
            results_df = pd.DataFrame(individual_results)
            results_df['model_evaluated'] = model_name
            results_df['evaluation_method'] = evaluation_method
            
            csv_path = self.final_dir / f'asr_results_{model_safe_name}.csv'
            results_df.to_csv(csv_path, index=False)
            
            self.logger.info(f"Standard results saved to: {csv_path}")
        
        # Save summary statistics
        stats = results['statistics']
        summary_data = [{
            'model_evaluated': model_name,
            'evaluation_method': evaluation_method,
            'evaluation_date': stats['evaluation_date'],
            'wer_mean': stats['wer_mean'],
            'wer_std': stats['wer_std'],
            'wer_min': stats['wer_min'],
            'wer_max': stats['wer_max'],
            'wer_median': stats['wer_median']
        }]
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = self.final_dir / 'asr_evaluation_summary.csv'
        
        # Append to existing summary
        if summary_path.exists():
            existing_df = pd.read_csv(summary_path)
            summary_df = pd.concat([existing_df, summary_df], ignore_index=True)
        
        summary_df.to_csv(summary_path, index=False)
        self.logger.info(f"Summary statistics saved to: {summary_path}")
    
    
    def create_evaluation_plots(self, results: Dict[str, Any]):
        """
        Create visualization plots for evaluation results
        """
        self.logger.info("Creating evaluation visualization plots")
        
        individual_results = results['individual_results']
        stats = results['statistics']
        model_name = stats['model_evaluated']
        model_safe_name = model_name.replace('/', '_').replace(':', '_')
        
        # Extract WER scores
        wer_scores = [r['wer'] for r in individual_results]
        run_numbers = [r['run_number'] for r in individual_results]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'ASR Evaluation Results: {model_name} ({len(wer_scores)} runs)', fontsize=14)
        
        # 1. WER over runs
        axes[0, 0].plot(run_numbers, wer_scores, 'b-', alpha=0.7, linewidth=1)
        axes[0, 0].axhline(y=stats['wer_mean'], color='r', linestyle='--', linewidth=2, 
                          label=f'Mean: {stats["wer_mean"]:.4f}')
        axes[0, 0].fill_between(run_numbers, 
                               stats['wer_mean'] - stats['wer_std'], 
                               stats['wer_mean'] + stats['wer_std'], 
                               alpha=0.2, color='r', label=f'±1 Std: {stats["wer_std"]:.4f}')
        axes[0, 0].set_title('WER Across Evaluation Runs')
        axes[0, 0].set_xlabel('Run Number')
        axes[0, 0].set_ylabel('WER')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. WER histogram
        axes[0, 1].hist(wer_scores, bins=min(20, len(wer_scores)//2), edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=stats['wer_mean'], color='r', linestyle='--', linewidth=2, 
                          label=f'Mean: {stats["wer_mean"]:.4f}')
        axes[0, 1].axvline(x=stats['wer_median'], color='g', linestyle='--', linewidth=2, 
                          label=f'Median: {stats["wer_median"]:.4f}')
        axes[0, 1].set_title('WER Distribution')
        axes[0, 1].set_xlabel('WER')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. Box plot
        axes[1, 0].boxplot(wer_scores, vert=True)
        axes[1, 0].set_title('WER Box Plot')
        axes[1, 0].set_ylabel('WER')
        axes[1, 0].set_xticklabels([f'{model_safe_name[:20]}...'])
        axes[1, 0].grid(True)
        
        # 4. Statistics summary (text)
        axes[1, 1].axis('off')
        stats_text = f"""
        Model: {model_name}
        
        Evaluation Statistics:
        
        Number of Runs: {stats['num_runs']}
        Test Samples: {stats['num_samples']}
        
        WER Results:
        Mean: {stats['wer_mean']:.4f}
        Std Dev: {stats['wer_std']:.4f}
        Min: {stats['wer_min']:.4f}
        Max: {stats['wer_max']:.4f}
        Median: {stats['wer_median']:.4f}
        
        Confidence Interval (±1 std):
        [{stats['wer_mean'] - stats['wer_std']:.4f}, {stats['wer_mean'] + stats['wer_std']:.4f}]
        
        Percentiles:
        25th: {stats['wer_25th_percentile']:.4f}
        75th: {stats['wer_75th_percentile']:.4f}
        """
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / f'evaluation_results_{model_safe_name}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Evaluation plots saved to {plot_path}")
    
    def run_evaluation_pipeline(self, model_path: str, split_chunks_path: str) -> Dict[str, Any]:
        """
        Run complete ASR evaluation pipeline
        """
        self.logger.info("=== Starting ASR Evaluation Pipeline (Enhanced with Bootstrap) ===")
        
        try:
            # 1. Load specified model
            self.load_trained_model(model_path)
            
            # 2. Load test dataset
            test_dataset = self.load_test_dataset(split_chunks_path)
            
            # 3. Check if bootstrap evaluation is requested
            use_bootstrap = self.eval_config.get('use_bootstrap_analysis', False)
            
            if use_bootstrap:
                self.logger.info("Using BOOTSTRAP speaker-level analysis")
                
                n_bootstrap = self.eval_config.get('n_bootstrap_iterations', 1000)
                bootstrap_results = self.bootstrap_wer_analysis(test_dataset, n_bootstrap)
                
                prepared_test_dataset = self.prepare_test_dataset(test_dataset)
                standard_results = self.run_multiple_evaluations(prepared_test_dataset)
                
                results = {
                    'evaluation_method': 'bootstrap_speaker_level',
                    'bootstrap_results': bootstrap_results,
                    'standard_results': standard_results,
                    'statistics': bootstrap_results['bootstrap_statistics']
                }
                
            else:
                self.logger.info("Using STANDARD multiple-runs analysis")
                
                # 4. Prepare test dataset
                prepared_test_dataset = self.prepare_test_dataset(test_dataset)
                
                # 5. Run standard multiple evaluations
                results = self.run_multiple_evaluations(prepared_test_dataset)
                results['evaluation_method'] = 'standard_multiple_runs'
            
            # 6. Save results to CSV
            self.save_results_csv(results)
            
            # 7. Create visualization plots
            self.create_evaluation_plots(results)
            
            # 8. Log final statistics
            stats = results['statistics']
            self.logger.info("=== ASR Evaluation Results ===")
            self.logger.info(f"Evaluation method: {results['evaluation_method']}")
            self.logger.info(f"Model evaluated: {stats.get('model_evaluated', 'unknown')}")
            
            if use_bootstrap:
                self.logger.info(f"Bootstrap iterations: {stats['n_bootstrap_iterations']}")
                self.logger.info(f"Bootstrap success rate: {stats['success_rate']:.2f}")
                self.logger.info(f"WER Mean: {stats['wer_mean']:.4f} ± {stats['wer_std']:.4f}")
                self.logger.info(f"WER 95% CI: [{stats['confidence_interval_95'][0]:.4f}, {stats['confidence_interval_95'][1]:.4f}]")
            else:
                self.logger.info(f"Number of evaluation runs: {stats['num_runs']}")
                self.logger.info(f"WER Mean: {stats['wer_mean']:.4f} ± {stats['wer_std']:.4f}")
            
            self.logger.info("=== ASR Evaluation Pipeline Completed ===")
            return results
            
        except Exception as e:
            self.logger.error(f"ASR evaluation failed: {str(e)}")
            raise


def evaluate_asr_model(config: dict, logger, model_path: str, split_chunks_path: str) -> Dict[str, Any]:
    """
    Main function for running ASR evaluation
    """
    evaluator = ASREvaluator(config, logger)
    return evaluator.run_evaluation_pipeline(model_path, split_chunks_path)