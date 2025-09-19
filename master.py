#!/usr/bin/env python3
"""
This master file orchestrates all individual experiment pipelines:
1. Initial data processing (Data Processing + Metadata Extraction)
2. ASR training (HuggingFace model training on original or anonymized data)
3. ASR grid search (Grid search over batch_size, epochs, learning_rate)
4. ASR evaluation (Multiple statistical evaluation runs on specified model)
5. ASV evaluation (Speaker verification across privacy attack levels)

Usage: python master.py
All configuration is loaded from ./config.yaml
"""

# Basic imports and environment setup
import os
# If problems occur with Multi-GPU memory allocation, set specific GPU here:
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import sys
import yaml
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
import pandas as pd
import subprocess
import torch

# Imports for individual pipeline functions
from functions.load_data import load_and_organize_data
from functions.metadata_extraction import extract_speaker_metadata
from functions.transcript_extraction import extract_timestamps_transcripts
from functions.audio_processing import create_audio_chunks
from functions.silence_filtering import filter_silence_outliers
from functions.mcadams_anonymization import anonymize_audio_chunks
from functions.silence_padding import create_silence_padded_chunks
from functions.length_marking import mark_chunk_usage_by_length
from functions.transcript_cleaning import clean_transcripts_and_update_usage
from functions.train_test_split import apply_train_test_split
from functions.speaker_metadata_completion import complete_speaker_metadata
from functions.asv_pairs_creation import create_asv_pairs
from functions.final_csv_creation import create_final_combined_csv
from functions.asr_training import train_asr_model
from functions.asr_evaluation import evaluate_asr_model
from functions.asv_evaluation import evaluate_asv_model


class ExperimentPipeline:
    """Main pipeline orchestrator class"""

    def __init__(self):
        """Initialize pipeline with configuration from ./config.yaml"""
        config_path = "./config.yaml"
        self.config = self.load_config(config_path)
        self.logger = self.setup_logging()
        self.validate_config()

        # Initialize output directories
        self.setup_output_directories()

        # Validate GPU setup
        self.check_gpu_setup()

    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            print(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            print(f"Error: Configuration file {config_path} not found!")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML configuration: {e}")
            sys.exit(1)

    def setup_logging(self) -> logging.Logger:
        """Setup logging with timestamps and different levels."""
        # Create main logger
        main_logger = logging.getLogger('experiment_pipeline')
        main_logger.setLevel(getattr(logging, self.config['logging']['level']))

        # Clear any existing handlers
        main_logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Console handler
        if self.config['logging']['log_to_console']:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            main_logger.addHandler(console_handler)

        # File handler (will be set up after output directories are created)
        return main_logger

    def validate_config(self):
        """Validate configuration parameters."""
        if 'model_training' in self.config:
            audio_type = self.config['model_training'].get('audio_type', 'original')
            if audio_type not in ['original', 'anonymized']:
                self.logger.error(f"Invalid audio_type: {audio_type}. Must be 'original' or 'anonymized'")
                sys.exit(1)
            
            # Automatically set model_suffix based on audio_type
            self.config['model_training']['model_suffix'] = audio_type
            
            # Validate data_percentage
            data_percentage = self.config['model_training'].get('data_percentage', 100)
            if not (1 <= data_percentage <= 100):
                self.logger.error(f"Invalid data_percentage: {data_percentage}. Must be between 1 and 100")
                sys.exit(1)

    def setup_output_directories(self):
        """Create output directories."""
        self.temp_dir = Path(self.config['output']['temporary_data_dir'])
        self.final_dir = Path(self.config['output']['final_data_dir'])

        # Create directories
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)

        # Setup file logging in temp directory
        if self.config['logging']['log_to_file']:
            log_file = self.temp_dir / 'main_pipeline.log'
            file_handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        self.logger.info(f"Temporary directory: {self.temp_dir}")
        self.logger.info(f"Final output directory: {self.final_dir}")

    def check_gpu_setup(self):
        """GPU setup validation."""
        self.logger.info("=== GPU Setup Validation ===")
        
        # Check CUDA_VISIBLE_DEVICES
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
        self.logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_devices}")
        
        # Check nvidia-smi
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
            self.logger.info("NVIDIA-SMI successful - GPU drivers are working")
            lines = result.stdout.split('\n')
            for line in lines:
                if 'GPU' in line and ('MiB' in line or 'C' in line):
                    self.logger.info(f"GPU Status: {line.strip()}")
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            self.logger.warning(f"nvidia-smi check failed: {e}")
        
        # Check PyTorch CUDA
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            
            self.logger.info(f"PyTorch CUDA available: YES")
            self.logger.info(f"Visible GPU count: {num_gpus}")
            self.logger.info(f"Current device: {current_device} ({device_name})")
            
            # Test GPU memory
            try:
                x = torch.randn(1000, 1000).cuda()
                del x
                torch.cuda.empty_cache()
                self.logger.info("GPU memory test: PASSED")
            except Exception as e:
                self.logger.error(f"GPU memory test FAILED: {e}")
        else:
            self.logger.error("PyTorch CUDA NOT available - only slow CPU training!")
            self.logger.error("Check CUDA installation and GPU drivers")

    def run_initial_data_processing(self):
        """
        Pipeline 1: Initial Data Processing + Metadata Extraction
        """
        self.logger.info("=== Starting Initial Data Processing Pipeline ===")

        try:
            # Step 1: Load all data (.cha and .wav files)
            self.logger.info("Step 1: Loading .cha and .wav files")
            temp_csv_path = load_and_organize_data(
                config=self.config,
                logger=self.logger
            )
            self.logger.info(f"Temporary data file created: {temp_csv_path}")

            # Step 2: Extract initial metadata per speaker
            self.logger.info("Step 2: Extracting initial metadata per speaker")
            metadata_csv_path = extract_speaker_metadata(
                config=self.config,
                logger=self.logger,
                input_csv_path=temp_csv_path
            )
            self.logger.info(f"Speaker metadata file created: {metadata_csv_path}")

            # Step 3: Filter for diagnosed aphasia patients only
            self.logger.info("Step 3: Filtering for aphasia patients")

            # Load the metadata CSV
            metadata_df = pd.read_csv(metadata_csv_path)
            initial_speaker_count = len(metadata_df)
            self.logger.info(f"Initial number of speakers: {initial_speaker_count}")

            # Get filtering parameters from config
            exclude_wab_type = self.config['data_filtering']['exclude_wab_type'].lower()
            exclude_wab_score = self.config['data_filtering']['exclude_wab_score']

            # Apply filtering criteria
            metadata_df['WAB_type_lower'] = metadata_df['WAB_type'].astype(str).str.lower()

            # Filter 1: Exclude specific WAB_type (case-insensitive)
            type_filter = metadata_df['WAB_type_lower'] != exclude_wab_type

            # Filter 2: Exclude specific WAB_score
            # Handle potential NaN values in WAB_score
            score_filter = (metadata_df['WAB_score'].isna()) | (metadata_df['WAB_score'] != exclude_wab_score)

            # Combine both filters
            combined_filter = type_filter & score_filter
            filtered_metadata_df = metadata_df[combined_filter].copy()

            # Drop the temporary column
            filtered_metadata_df = filtered_metadata_df.drop('WAB_type_lower', axis=1)

            # Log filtering statistics
            excluded_by_type = metadata_df[~type_filter]
            excluded_by_score = metadata_df[type_filter & ~score_filter]  # Only those that passed type filter

            self.logger.info(f"Speakers excluded by WAB_type '{exclude_wab_type}': {len(excluded_by_type)}")
            if len(excluded_by_type) > 0:
                excluded_type_ids = excluded_by_type['speaker_id'].tolist()
                self.logger.info(f"Excluded by type - Speaker IDs: {excluded_type_ids}")

            self.logger.info(f"Speakers excluded by WAB_score = {exclude_wab_score}: {len(excluded_by_score)}")
            if len(excluded_by_score) > 0:
                excluded_score_ids = excluded_by_score['speaker_id'].tolist()
                self.logger.info(f"Excluded by score - Speaker IDs: {excluded_score_ids}")

            # Get list of valid speaker IDs for further processing
            valid_speaker_ids = filtered_metadata_df['speaker_id'].tolist()
            filtered_speaker_count = len(valid_speaker_ids)
            total_excluded = initial_speaker_count - filtered_speaker_count

            self.logger.info(f"Speakers after all filtering: {filtered_speaker_count}")
            self.logger.info(f"Total speakers excluded: {total_excluded}")

            # Save filtered metadata with configurable filename
            filtered_metadata_filename = self.config['processing']['filtered_metadata_filename']
            filtered_metadata_path = self.temp_dir / filtered_metadata_filename
            filtered_metadata_df.to_csv(filtered_metadata_path, index=False)
            self.logger.info(f"Filtered metadata saved: {filtered_metadata_path}")

            # Also filter the index/load_data CSV to only include valid speakers
            load_data_df = pd.read_csv(temp_csv_path)
            initial_recordings_count = len(load_data_df)
            filtered_load_data_df = load_data_df[load_data_df['patient_id'].isin(valid_speaker_ids)].copy()
            filtered_recordings_count = len(filtered_load_data_df)

            self.logger.info(f"Recordings before filtering: {initial_recordings_count}")
            self.logger.info(f"Recordings after filtering: {filtered_recordings_count}")
            self.logger.info(f"Recordings excluded: {initial_recordings_count - filtered_recordings_count}")

            # Save filtered load data with configurable filename
            filtered_load_data_filename = self.config['processing']['filtered_load_data_filename']
            filtered_load_data_path = self.temp_dir / filtered_load_data_filename
            filtered_load_data_df.to_csv(filtered_load_data_path, index=False)
            self.logger.info(f"Filtered load data saved: {filtered_load_data_path}")

            # Step 4: Extract timestamps and original transcripts (chunk level)
            self.logger.info("Step 4: Extracting timestamps and transcripts")
            chunks_csv_path = extract_timestamps_transcripts(
                config=self.config,
                logger=self.logger,
                input_csv_path=filtered_load_data_path
            )
            self.logger.info(f"Chunks data file created: {chunks_csv_path}")

            # Step 5: Cut original .wav audio & create audio chunks
            self.logger.info("Step 5: Creating audio chunks")
            chunks_with_audio_path = create_audio_chunks(
                config=self.config,
                logger=self.logger,
                chunks_csv_path=chunks_csv_path,
                load_data_csv_path=filtered_load_data_path
            )
            self.logger.info(f"Audio chunks created, updated file: {chunks_with_audio_path}")

            # Step 6: Filter chunks with outlier silence (1.5 IQR method)
            self.logger.info("Step 6: Filtering outlier silence chunks")
            silence_filtered_path = filter_silence_outliers(
                config=self.config,
                logger=self.logger,
                chunks_with_audio_path=chunks_with_audio_path
            )
            self.logger.info(f"Silence-filtered chunks file created: {silence_filtered_path}")

            # Step 7: Run anonymization on audio chunks
            self.logger.info("Step 7: Anonymizing audio chunks with McAdams")
            anonymized_chunks_path = anonymize_audio_chunks(
                config=self.config,
                logger=self.logger,
                silence_filtered_path=silence_filtered_path
            )
            self.logger.info(f"Anonymized chunks file created: {anonymized_chunks_path}")

            # Step 7b: Create silence-padded chunks for ASR training
            self.logger.info("Step 7b: Creating silence-padded chunks for ASR")
            silence_padded_path = create_silence_padded_chunks(
                config=self.config,
                logger=self.logger,
                anonymized_chunks_path=anonymized_chunks_path
            )
            self.logger.info(f"Silence-padded chunks file created: {silence_padded_path}")

            # Step 8: Measure chunk lengths and mark usage
            # Chunks < 1s -> NO, >= 1s -> ONLY_ASR, >= 1.8s -> BOTH
            self.logger.info("Step 8: Marking chunk usage based on length")
            length_marked_path = mark_chunk_usage_by_length(
                config=self.config,
                logger=self.logger,
                silence_filtered_path=silence_padded_path
            )
            self.logger.info(f"Length-marked chunks file created: {length_marked_path}")

            # Step 9: Clean transcripts and update usage based on ground truth availability
            self.logger.info("Step 9: Cleaning transcripts and updating usage")
            transcript_cleaned_path = clean_transcripts_and_update_usage(
                config=self.config,
                logger=self.logger,
                length_marked_path=length_marked_path
            )
            self.logger.info(f"Transcript-cleaned chunks file created: {transcript_cleaned_path}")

            # Step 10: Apply random test set split (70/15/15) for ASR-suitable chunks
            self.logger.info("Step 10: Applying train/val/test split")
            split_chunks_path = apply_train_test_split(
                config=self.config,
                logger=self.logger,
                transcript_cleaned_path=transcript_cleaned_path
            )
            self.logger.info(f"Train/test split completed, updated file: {split_chunks_path}")

            # Step 11: Fill missing speaker metadata (speech_length, num_recordings)
            self.logger.info("Step 11: Completing speaker metadata")
            completed_metadata_path = complete_speaker_metadata(
                config=self.config,
                logger=self.logger,
                split_chunks_path=split_chunks_path,
                filtered_metadata_path=filtered_metadata_path
            )
            self.logger.info(f"Speaker metadata completed: {completed_metadata_path}")

            # Step 12: Create ASV pairs for speaker verification evaluation
            self.logger.info("Step 12: Creating ASV pairs for speaker verification")
            asv_pairs_path = create_asv_pairs(
                config=self.config,
                logger=self.logger,
                split_chunks_path=split_chunks_path
            )

            # Use updated chunk path with ASV roles if ASV pairs were created
            if asv_pairs_path:
                self.logger.info(f"ASV pairs created: {asv_pairs_path}")
                chunks_with_asv_filename = self.config['processing']['chunks_with_asv_roles_filename']
                chunks_with_asv_path = self.temp_dir / chunks_with_asv_filename
            else:
                self.logger.warning("ASV pairs creation failed or skipped")
                chunks_with_asv_path = split_chunks_path

            # Step 13: Create and save final combined CSV with all information
            self.logger.info("Step 13: Creating final combined dataset")
            final_combined_path = create_final_combined_csv(
                config=self.config,
                logger=self.logger,
                transcript_cleaned_path=chunks_with_asv_path,
                filtered_metadata_path=completed_metadata_path
            )
            self.logger.info(f"Final combined dataset created: {final_combined_path}")

            import shutil
            index_file_path = self.final_dir / self.config['output']['final_index_filename']
            metadata_file_path = self.final_dir / self.config['output']['final_metadata_filename']

            if os.path.exists(chunks_with_asv_path):
                shutil.copy2(chunks_with_asv_path, index_file_path)
                self.logger.info(f"Final index data copied to: {index_file_path}")
            elif os.path.exists(split_chunks_path):
                shutil.copy2(split_chunks_path, index_file_path)
                self.logger.info(f"Final index data copied to: {index_file_path}")

            if os.path.exists(completed_metadata_path):
                shutil.copy2(completed_metadata_path, metadata_file_path)
                self.logger.info(f"Final metadata copied to: {metadata_file_path}")

            self.logger.info("=== Initial Data Processing Pipeline Completed ===")

        except Exception as e:
            self.logger.error(f"Error in initial data processing: {str(e)}")
            raise

    def run_asr_training(self):
        """
        Pipeline 2: ASR Training
        """
        self.logger.info("=== Starting ASR Training Pipeline ===")

        try:
            # Check required input file
            temp_dir = Path(self.config['output']['temporary_data_dir'])
            split_chunks_filename = self.config['processing']['split_chunks_filename']
            split_chunks_path = temp_dir / split_chunks_filename

            if not split_chunks_path.exists():
                self.logger.error(f"Required input file not found: {split_chunks_path}")
                self.logger.error("Please run initial_processing pipeline first!")
                raise FileNotFoundError(f"Split chunks file not found: {split_chunks_path}")

            self.logger.info(f"Using stratified data from: {split_chunks_path}")

            # Log training configuration
            audio_type = self.config['model_training']['audio_type']
            data_percentage = self.config['model_training']['data_percentage']
            model_suffix = self.config['model_training']['model_suffix']
            
            self.logger.info(f"Training Configuration:")
            self.logger.info(f"  - Audio Type: {audio_type}")
            self.logger.info(f"  - Data Percentage: {data_percentage}%")
            self.logger.info(f"  - Model Suffix: {model_suffix}")

            # Run ASR training
            model_path, final_wer = train_asr_model(
                config=self.config,
                logger=self.logger,
                split_chunks_path=str(split_chunks_path)
            )

            self.logger.info(f"ASR model training completed successfully!")
            self.logger.info(f"Model saved to: {model_path}")
            self.logger.info(f"Final WER: {final_wer:.4f}")

            # Save model path for evaluation pipeline
            model_info = {
                'model_path': model_path,
                'training_completed': True,
                'model_type': 'wav2vec2-ctc',
                'audio_type': audio_type,
                'model_suffix': model_suffix,
                'data_percentage': data_percentage,
                'final_wer': final_wer
            }

            model_info_path = temp_dir / f'trained_model_info_{model_suffix}.json'
            import json
            with open(model_info_path, 'w') as f:
                json.dump(model_info, f, indent=2)

            self.logger.info(f"Model info saved to: {model_info_path}")

        except Exception as e:
            self.logger.error(f"ASR training failed: {str(e)}")
            raise

    def run_asr_grid_search(self):
        """
        Pipeline 3: ASR Grid Search
        """
        self.logger.info("=== Starting ASR Grid Search Pipeline ===")

        try:
            # Check required input file
            temp_dir = Path(self.config['output']['temporary_data_dir'])
            split_chunks_filename = self.config['processing']['split_chunks_filename']
            split_chunks_path = temp_dir / split_chunks_filename

            if not split_chunks_path.exists():
                self.logger.error(f"Required input file not found: {split_chunks_path}")
                self.logger.error("Please run initial_processing pipeline first!")
                raise FileNotFoundError(f"Split chunks file not found: {split_chunks_path}")

            # Define grid search parameters
            batch_sizes = [2, 8, 32]
            epochs_list = [10, 30, 50]
            learning_rates = [5e-6, 1e-5, 2e-5]
            
            total_combinations = len(batch_sizes) * len(epochs_list) * len(learning_rates)
            self.logger.info(f"Grid Search Configuration:")
            self.logger.info(f"  - Batch sizes: {batch_sizes}")
            self.logger.info(f"  - Epochs: {epochs_list}")
            self.logger.info(f"  - Learning rates: {learning_rates}")
            self.logger.info(f"  - Total combinations: {total_combinations}")

            # Initialize results storage
            grid_search_results = []
            combination_counter = 0

            # Grid search loop
            for batch_size in batch_sizes:
                for epochs in epochs_list:
                    for learning_rate in learning_rates:
                        combination_counter += 1
                        
                        self.logger.info("="*80)
                        self.logger.info(f"GRID SEARCH: Combination {combination_counter}/{total_combinations}")
                        self.logger.info(f"Batch Size: {batch_size}, Epochs: {epochs}, Learning Rate: {learning_rate}")
                        self.logger.info("="*80)
                        
                        start_time = datetime.now()
                        
                        try:
                            # Train model with current hyperparameters
                            model_path, final_wer = train_asr_model(
                                config=self.config,
                                logger=self.logger,
                                split_chunks_path=str(split_chunks_path),
                                batch_size=batch_size,
                                epochs=epochs,
                                learning_rate=learning_rate
                            )
                            
                            end_time = datetime.now()
                            training_duration = (end_time - start_time).total_seconds() / 60
                            
                            result = {
                                'combination': combination_counter,
                                'batch_size': batch_size,
                                'epochs': epochs,
                                'learning_rate': learning_rate,
                                'final_wer': final_wer,
                                'model_path': model_path,
                                'training_duration_minutes': round(training_duration, 2),
                                'start_time': start_time.isoformat(),
                                'end_time': end_time.isoformat(),
                                'status': 'success'
                            }
                            
                            self.logger.info(f"Combination {combination_counter} completed successfully!")
                            self.logger.info(f"Final WER: {final_wer:.4f}")
                            self.logger.info(f"Training Duration: {training_duration:.1f} minutes")
                            
                        except Exception as e:
                            self.logger.error(f"Combination {combination_counter} failed: {str(e)}")
                            end_time = datetime.now()
                            training_duration = (end_time - start_time).total_seconds() / 60
                            
                            result = {
                                'combination': combination_counter,
                                'batch_size': batch_size,
                                'epochs': epochs,
                                'learning_rate': learning_rate,
                                'final_wer': float('inf'),
                                'model_path': None,
                                'training_duration_minutes': round(training_duration, 2),
                                'start_time': start_time.isoformat(),
                                'end_time': end_time.isoformat(),
                                'status': 'failed',
                                'error': str(e)
                            }
                        
                        grid_search_results.append(result)
                        
                        # Save intermediate results after each combination
                        self._save_grid_search_results(grid_search_results)

            # Final results analysis and logging
            self._analyze_grid_search_results(grid_search_results)
            
            self.logger.info("=== ASR Grid Search Pipeline Completed ===")

        except Exception as e:
            self.logger.error(f"ASR Grid Search failed: {str(e)}")
            raise

    def _save_grid_search_results(self, results):
        """Save grid search results to CSV"""
        
        results_df = pd.DataFrame(results)
        output_path = Path("./data/grid_search_x.csv")
        output_path.parent.mkdir(exist_ok=True)
        
        results_df.to_csv(output_path, index=False)
        self.logger.info(f"Grid search results saved to: {output_path}")

    def _analyze_grid_search_results(self, results):
        """Analyze and log grid search results"""
        
        results_df = pd.DataFrame(results)
        successful_results = results_df[results_df['status'] == 'success']
        
        if len(successful_results) == 0:
            self.logger.error("No successful training runs in grid search!")
            return
        
        # Find best combination
        best_result = successful_results.loc[successful_results['final_wer'].idxmin()]
        
        self.logger.info("="*80)
        self.logger.info("GRID SEARCH RESULTS SUMMARY")
        self.logger.info("="*80)
        self.logger.info(f"Total combinations: {len(results)}")
        self.logger.info(f"Successful runs: {len(successful_results)}")
        self.logger.info(f"Failed runs: {len(results) - len(successful_results)}")
        self.logger.info("")
        self.logger.info("BEST COMBINATION:")
        self.logger.info(f"Batch Size: {best_result['batch_size']}")
        self.logger.info(f"Epochs: {best_result['epochs']}")
        self.logger.info(f"Learning Rate: {best_result['learning_rate']}")
        self.logger.info(f"Final WER: {best_result['final_wer']:.4f}")
        self.logger.info(f"Training Duration: {best_result['training_duration_minutes']:.1f} minutes")
        self.logger.info("")
        
        # Show top 5 combinations
        top_5 = successful_results.nsmallest(5, 'final_wer')
        self.logger.info("TOP 5 COMBINATIONS:")
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            self.logger.info(f"  {i}. WER={row['final_wer']:.4f} | BS={row['batch_size']} | E={row['epochs']} | LR={row['learning_rate']}")

    def run_asr_evaluation(self):
        """
        Pipeline 4: ASR Evaluation
        """
        self.logger.info("=== Starting ASR Evaluation Pipeline ===")

        try:
            # Check required input files
            temp_dir = Path(self.config['output']['temporary_data_dir'])
            
            # Get setup from config
            model_to_evaluate = self.config['asr_evaluation']['model_to_evaluate']
            self.logger.info(f"Model to evaluate: {model_to_evaluate}")
            model_path = self._resolve_asr_model_path(model_to_evaluate)
            self.logger.info(f"Resolved model path: {model_path}")
            
            split_chunks_filename = self.config['processing']['split_chunks_filename']
            split_chunks_path = temp_dir / split_chunks_filename

            if not split_chunks_path.exists():
                self.logger.error("No test data found. Please run initial_processing pipeline first!")
                raise FileNotFoundError(f"Split chunks file not found: {split_chunks_path}")

            self.logger.info(f"Using test data from: {split_chunks_path}")

            # Start ASR evaluation run
            self.logger.info("Starting ASR model evaluation with statistical analysis...")

            asr_results = evaluate_asr_model(
                config=self.config,
                logger=self.logger,
                model_path=model_path,
                split_chunks_path=str(split_chunks_path)
            )

            # Log ASR results
            asr_stats = asr_results['statistics']
            self.logger.info("FINAL ASR EVALUATION RESULTS:")
            self.logger.info(f"Model: {model_to_evaluate}")
            self.logger.info(f"Test Set WER: {asr_stats['wer_mean']:.4f} ± {asr_stats['wer_std']:.4f}")
            self.logger.info(f"Confidence Interval: [{asr_stats['wer_mean'] - asr_stats['wer_std']:.4f}, {asr_stats['wer_mean'] + asr_stats['wer_std']:.4f}]")
            self.logger.info(f"Based on {asr_stats['num_runs']} evaluation runs")
            self.logger.info(f"Test samples: {asr_stats['num_samples']}")

            self.logger.info("=== ASR Evaluation Pipeline Completed ===")

        except Exception as e:
            self.logger.error(f"ASR evaluation failed: {str(e)}")
            raise

    def run_asv_evaluation(self):
        """
        Pipeline 5: ASV Evaluation
        """
        self.logger.info("=== Starting ASV Evaluation Pipeline ===")

        try:
            # Check required input files
            temp_dir = Path(self.config['output']['temporary_data_dir'])
            final_dir = Path(self.config['output']['final_data_dir'])
            
            # Check for ASV pairs
            asv_pairs_filename = self.config['output']['asv_pairs_filename']
            asv_pairs_path = final_dir / asv_pairs_filename

            if not asv_pairs_path.exists():
                self.logger.error("No ASV pairs found. Please run initial_processing pipeline first!")
                raise FileNotFoundError(f"ASV pairs file not found: {asv_pairs_path}")

            self.logger.info(f"Using ASV pairs from: {asv_pairs_path}")

            # Check for chunks with ASV roles
            chunks_with_asv_filename = self.config['processing']['chunks_with_asv_roles_filename']
            chunks_with_asv_path = temp_dir / chunks_with_asv_filename

            if not chunks_with_asv_path.exists():
                split_chunks_filename = self.config['processing']['split_chunks_filename']
                chunks_with_asv_path = temp_dir / split_chunks_filename
                self.logger.info(f"Using fallback chunks data: {chunks_with_asv_path}")
            else:
                self.logger.info(f"Using chunks with ASV roles: {chunks_with_asv_path}")

            # Start ASV evaluation run
            self.logger.info("Starting ASV evaluation across privacy attack levels...")

            asv_results = evaluate_asv_model(
                config=self.config,
                logger=self.logger,
                asv_pairs_path=str(asv_pairs_path),
                chunks_path=str(chunks_with_asv_path)
            )

            # Log ASV results
            self.logger.info("FINAL ASV EVALUATION RESULTS (Statistical):")
            attack_level_names = {
                'oo': 'Unprotected (Original vs Original)',
                'oa': 'Ignorant Attack (Original vs Anonymized)',
                'aa': 'Lazy-Informed Attack (Anonymized vs Anonymized)'
            }

            for attack_level, results in asv_results.items():
                if results is not None and 'statistics' in results:
                    stats = results['statistics']
                    attack_name = attack_level_names[attack_level]
                    self.logger.info(f"{attack_name}:")
                    self.logger.info(f"EER: {stats['eer_mean']:.4f} ± {stats['eer_std']:.4f}")
                    self.logger.info(f"AUC: {stats['auc_mean']:.4f} ± {stats['auc_std']:.4f}")
                    self.logger.info(f"Runs: {stats['num_runs']}")
                    self.logger.info(f"EER Range: [{stats['eer_min']:.4f}, {stats['eer_max']:.4f}]")

            self.logger.info("=== ASV Evaluation Pipeline Completed ===")

        except Exception as e:
            self.logger.error(f"ASV evaluation failed: {str(e)}")
            raise

    def _resolve_asr_model_path(self, model_to_evaluate: str) -> str:
        """
        Resolve the ASR model path based on config specification.
        """
        temp_dir = Path(self.config['output']['temporary_data_dir'])
        
        if model_to_evaluate == "pretrained":
            # Use the base pretrained model
            model_path = self.config['model_training']['huggingface_model_name']
            self.logger.info(f"Using pretrained model: {model_path}")
            
        elif model_to_evaluate == "pretrained_anonymized":
            # Use the base pretrained model but on anonymized data
            model_path = self.config['model_training']['huggingface_model_name']
            self.logger.info(f"Using pretrained model on anonymized audio: {model_path}")
            
        elif model_to_evaluate == "trained_original":
            # Look for trained model on original data
            model_info_path = temp_dir / 'trained_model_info_original.json'
            if not model_info_path.exists():
                raise FileNotFoundError(f"No trained original model found. Train model first or check: {model_info_path}")
            
            import json
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
            model_path = model_info['model_path']
            
        elif model_to_evaluate == "trained_anonymized":
            # Look for trained model on anonymized data
            model_info_path = temp_dir / 'trained_model_info_anonymized.json'
            if not model_info_path.exists():
                raise FileNotFoundError(f"No trained anonymized model found. Train model first or check: {model_info_path}")
            
            import json
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
            model_path = model_info['model_path']
            
        elif model_to_evaluate.startswith('/') or model_to_evaluate.startswith('./'):
            # Direct path to model
            model_path = model_to_evaluate
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Custom model path does not exist: {model_path}")
                
        else:
            model_path = model_to_evaluate
            
        return model_path

    def run_pipeline(self):
        """Execute the specified pipeline based on the config file"""
        pipelines_to_run = self.config['execution']['pipelines_to_run']
        
        self.logger.info(f"Pipelines to run: {pipelines_to_run}")

        # Handle 'all' option
        if 'all' in pipelines_to_run:
            self.run_initial_data_processing()
            self.run_asr_training()
            self.run_asr_evaluation()
            self.run_asv_evaluation()
            
        else:
            # Run individual pipelines
            if 'initial_processing' in pipelines_to_run:
                self.run_initial_data_processing()

            if 'asr_training' in pipelines_to_run:
                self.run_asr_training()

            if 'asr_grid_search' in pipelines_to_run:
                self.run_asr_grid_search()

            if 'asr_evaluation' in pipelines_to_run:
                self.run_asr_evaluation()

            if 'asv_evaluation' in pipelines_to_run:
                self.run_asv_evaluation()

        self.logger.info("=== PIPELINE EXECUTION COMPLETED ===")


def main():
    """Main entry point."""
    try:
        # Initialize and run pipeline
        pipeline = ExperimentPipeline()
        pipeline.run_pipeline()

    except Exception as e:
        print(f"Pipeline failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()