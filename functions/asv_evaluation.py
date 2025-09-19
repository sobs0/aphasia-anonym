"""
ASV Evaluation Module
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
from datetime import datetime
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.metric_stats import EER
from sklearn.metrics import roc_curve, auc
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")


class ASVEvaluator:
    """
    ASV Evaluator class
    """
    
    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger
        self.asv_config = config['asv_evaluation']
        self.setup_directories()
        self.encoder_model = None
        self.evaluation_results = {}
        
        # Attack level mapping
        self.attack_levels = {
            'oo': 'unprotected',
            'oa': 'ignorant',
            'aa': 'lazy_informed'
        }
    
    def setup_directories(self):
        """Setup output directories for ASV evaluation"""
        self.temp_dir = Path(self.config['output']['temporary_data_dir'])
        self.final_dir = Path(self.config['output']['final_data_dir'])
        self.results_dir = self.temp_dir / 'asv_evaluation_results'
        self.models_dir = self.temp_dir / 'asv_models'
        
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"ASV evaluation results directory: {self.results_dir}")
        self.logger.info(f"ASV models directory: {self.models_dir}")
    
    def load_speechbrain_model(self):
        """
        Load SpeechBrain ECAPA-TDNN EncoderClassifier for embedding extraction
        """
        self.logger.info("Loading SpeechBrain ECAPA-TDNN EncoderClassifier...")
        
        try:
            self.encoder_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(self.models_dir / "spkrec-ecapa-voxceleb")
            )
            self.logger.info("ECAPA-TDNN EncoderClassifier loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load ECAPA-TDNN EncoderClassifier: {str(e)}")
            raise
    
    def load_asv_pairs_and_chunks(self, asv_pairs_path: str, chunks_path: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Load ASV pairs and chunks data, applying optional data percentage sampling
        """
        self.logger.info("Loading ASV pairs and chunks data")
        
        # Load ASV pairs
        asv_pairs = pd.read_csv(asv_pairs_path)
        self.logger.info(f"Loaded {len(asv_pairs)} ASV trial pairs")
        
        # Apply data percentage sampling if configured
        asv_data_percentage = self.config['asv_evaluation'].get('asv_data_percentage', 100) 
        
        if 0 < asv_data_percentage < 100:
            original_num_pairs = len(asv_pairs)
            asv_pairs = asv_pairs.sample(frac=asv_data_percentage / 100, random_state=42).reset_index(drop=True)
            self.logger.info(f"Sampled {asv_data_percentage}% of ASV trial pairs: {len(asv_pairs)} out of {original_num_pairs}")
        elif asv_data_percentage <= 0 or asv_data_percentage > 100:
            self.logger.warning(f"Invalid 'asv_data_percentage' specified ({asv_data_percentage}%). Using full dataset.")
        
        # Load chunks data for audio path mapping
        chunks_df = pd.read_csv(chunks_path)
        self.logger.info(f"Loaded {len(chunks_df)} chunks for audio path mapping")
        
        # Create mapping from chunk_id to audio paths
        chunk_audio_mapping = {}
        for _, row in chunks_df.iterrows():
            chunk_id = row['chunk_id']
            chunk_audio_mapping[chunk_id] = {
                'original': row['chunk_wav_path'],
                'anonymized': row['mcadams_anonymized_chunk_wav_path']
            }
        
        self.logger.info(f"Created audio path mapping for {len(chunk_audio_mapping)} chunks")
        return asv_pairs, chunk_audio_mapping

    def extract_embedding_from_audio(self, audio_path: str) -> np.ndarray:
        """
        Extract embedding from audio file
        """
        try:
            import torchaudio
            
            # Load and resample audio to 16kHz
            signal, fs = torchaudio.load(audio_path)
            if fs != 16000:
                signal = torchaudio.functional.resample(signal, fs, 16000)
            
            # Convert to mono if stereo
            if signal.shape[0] > 1:
                signal = torch.mean(signal, dim=0, keepdim=True)
            
            # Ensure batch dimension for encode_batch
            if signal.dim() == 1:
                signal = signal.unsqueeze(0)
            
            # Extract embedding
            with torch.no_grad():
                embeddings = self.encoder_model.encode_batch(signal)
            
            # Convert to numpy
            if isinstance(embeddings, torch.Tensor):
                embedding = embeddings.detach().cpu().numpy()
            else:
                embedding = np.array(embeddings)
            
            expected_embedding_dim = 192
            
            if embedding.shape == (1, expected_embedding_dim):
                embedding = embedding.squeeze(0)
            elif embedding.shape == (1, 1, expected_embedding_dim):
                embedding = embedding.squeeze()
            elif embedding.shape == (expected_embedding_dim,):
                pass
            else:
                # Reject invalid shapes
                raise ValueError(
                    f"Unexpected embedding shape: {embedding.shape}. "
                    f"Expected (1, {expected_embedding_dim}), (1, 1, {expected_embedding_dim}), or ({expected_embedding_dim},)"
                )
            
            # Ensure correct 1D embedding
            if embedding.shape != (expected_embedding_dim,):
                raise ValueError(
                    f"Failed to process embedding to correct shape. "
                    f"Got {embedding.shape}, expected ({expected_embedding_dim},)"
                )
            
            self.logger.debug(f"Successfully extracted embedding shape: {embedding.shape}")
            return embedding
            
        except Exception as e:
            self.logger.error(f"Failed to extract embedding from {audio_path}: {str(e)}")
            return None
    
    def create_enrollment_embeddings(self, asv_pairs: pd.DataFrame, chunk_audio_mapping: Dict) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Create average enrollment embeddings for each speaker and attack level
        """
        self.logger.info("Creating enrollment embeddings...")
        
        # Get unique enrollment speakers
        enrollment_speakers = asv_pairs['enrollment_speaker_id'].unique()
        self.logger.info(f"Processing {len(enrollment_speakers)} enrollment speakers")
        
        speaker_embeddings = {}
        
        for speaker_id in tqdm(enrollment_speakers, desc="Creating enrollment embeddings"):
            speaker_embeddings[speaker_id] = {}
            
            # Get enrollment utterance IDs for this speaker
            speaker_rows = asv_pairs[asv_pairs['enrollment_speaker_id'] == speaker_id]
            enrollment_utterance_ids = speaker_rows.iloc[0]['enrollment_utterance_ids'].split(',')
            
            # Create embeddings for both original and anonymized audio
            for audio_type in ['original', 'anonymized']:
                try:
                    # Get audio paths and extract embeddings for enrollment utterances
                    embeddings = []
                    valid_paths = 0
                    
                    for utterance_id in enrollment_utterance_ids:
                        if utterance_id in chunk_audio_mapping:
                            audio_path = chunk_audio_mapping[utterance_id][audio_type]
                            if audio_path and Path(audio_path).exists():
                                embedding = self.extract_embedding_from_audio(audio_path)
                                if embedding is not None:
                                    if embedding.ndim != 1:
                                        self.logger.warning(f"Embedding for {utterance_id} has {embedding.ndim} dimensions, flattening")
                                        embedding = embedding.flatten()
                                    embeddings.append(embedding)
                                    valid_paths += 1
                    
                    if embeddings:
                        
                        embedding_shapes = [emb.shape for emb in embeddings]
                        unique_shapes = list(set(embedding_shapes))
                        
                        if len(unique_shapes) > 1:
                            self.logger.warning(f"Speaker {speaker_id} ({audio_type}): Inconsistent embedding shapes: {unique_shapes}")
                            from collections import Counter
                            most_common_shape = Counter(embedding_shapes).most_common(1)[0][0]
                            filtered_embeddings = [emb for emb in embeddings if emb.shape == most_common_shape]
                            embeddings = filtered_embeddings
                            self.logger.info(f"Filtered to {len(embeddings)} embeddings with shape {most_common_shape}")
                        
                        if embeddings:
                            # Stack embeddings and calculate mean
                            embeddings_stack = np.stack(embeddings, axis=0)
                            avg_embedding = np.mean(embeddings_stack, axis=0)
                            
                            if avg_embedding.ndim != 1:
                                self.logger.warning(f"Average embedding for {speaker_id} ({audio_type}) has {avg_embedding.ndim} dimensions, flattening")
                                avg_embedding = avg_embedding.flatten()
                            
                            speaker_embeddings[speaker_id][audio_type] = avg_embedding
                            
                            self.logger.debug(f"Speaker {speaker_id} ({audio_type}): {len(embeddings)} embeddings -> avg shape: {avg_embedding.shape}")
                        else:
                            self.logger.warning(f"No valid embeddings after filtering for speaker {speaker_id} ({audio_type})")
                            speaker_embeddings[speaker_id][audio_type] = None
                    else:
                        self.logger.warning(f"No valid embeddings for speaker {speaker_id} ({audio_type})")
                        speaker_embeddings[speaker_id][audio_type] = None
                        
                except Exception as e:
                    self.logger.error(f"Error creating embedding for speaker {speaker_id} ({audio_type}): {str(e)}")
                    speaker_embeddings[speaker_id][audio_type] = None
        
        self.logger.info(f"Created enrollment embeddings for {len(speaker_embeddings)} speakers")
        return speaker_embeddings
    
    def compute_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        """
        try:
            # Ensure both embeddings are 1D
            if embedding1.ndim != 1:
                self.logger.warning(f"embedding1 has {embedding1.ndim} dimensions, flattening")
                embedding1 = embedding1.flatten()
            
            if embedding2.ndim != 1:
                self.logger.warning(f"embedding2 has {embedding2.ndim} dimensions, flattening")
                embedding2 = embedding2.flatten()
            
            # Check if embeddings have same length
            if len(embedding1) != len(embedding2):
                self.logger.error(f"Embedding dimension mismatch: {len(embedding1)} vs {len(embedding2)}")
                return None
            
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                self.logger.warning("Zero norm embedding detected")
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            similarity = np.clip(similarity, -1.0, 1.0)
            
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Error computing cosine similarity: {str(e)}")
            return None
    
    def compute_eer_speechbrain(self, positive_scores: List[float], negative_scores: List[float]) -> float:
        """
        Compute EER
        """
        try:
            positive_tensor = torch.tensor(positive_scores)
            negative_tensor = torch.tensor(negative_scores)
            eer, threshold = EER(positive_tensor, negative_tensor)
            return float(eer), float(threshold)
        except Exception as e:
            self.logger.error(f"Error computing EER with SpeechBrain: {e}")
            return self.compute_eer_sklearn(positive_scores, negative_scores)
    
    def compute_eer_sklearn(self, positive_scores: List[float], negative_scores: List[float]) -> Tuple[float, float]:
        """
        Fallback EER computation
        """
        # Combine scores and labels
        all_scores = positive_scores + negative_scores
        all_labels = [1] * len(positive_scores) + [0] * len(negative_scores)
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores, pos_label=1)
        fnr = 1 - tpr
        
        # Find EER point
        abs_diffs = np.abs(fnr - fpr)
        idx_eer = np.nanargmin(abs_diffs)
        eer = (fpr[idx_eer] + fnr[idx_eer]) / 2
        threshold = thresholds[idx_eer]
        
        return eer, threshold



    def bootstrap_asv_analysis(self, asv_pairs: pd.DataFrame, chunk_audio_mapping: Dict, 
                              speaker_embeddings: Dict, n_bootstrap: int = 1000) -> Dict[str, Any]:
        """
        Bootstrap resampling (speaker based) for EER comparison
        """
        self.logger.info("="*60)
        self.logger.info(f"BOOTSTRAP ASV ANALYSIS")
        self.logger.info(f"Number of bootstrap iterations: {n_bootstrap}")
        self.logger.info("="*60)
        
        # Extract unique enrollment speakers
        enrollment_speakers = asv_pairs['enrollment_speaker_id'].unique()
        self.logger.info(f"Total enrollment speakers available: {len(enrollment_speakers)}")
        
        # Group trial pairs by enrollment speaker
        speaker_trial_pairs = {}
        for speaker_id in enrollment_speakers:
            speaker_pairs = asv_pairs[asv_pairs['enrollment_speaker_id'] == speaker_id]
            speaker_trial_pairs[speaker_id] = speaker_pairs
        
        bootstrap_results = {}
        
        # Bootstrap analysis for each attack level
        for attack_level in ['aa']:
            self.logger.info("-"*60)
            self.logger.info(f"Starting bootstrap for attack level: {attack_level}")
            
            # Setup checkpoint file for this attack level
            checkpoint_file = self.results_dir / f'bootstrap_checkpoint_{attack_level}.csv'
            
            # Load existing results if checkpoint exists
            existing_results = []
            start_iteration = 0
            
            if checkpoint_file.exists():
                try:
                    existing_df = pd.read_csv(checkpoint_file)
                    existing_results = existing_df.to_dict('records')
                    start_iteration = len(existing_results)
                    self.logger.info(f"Resuming from checkpoint: {start_iteration} iterations completed")
                except Exception as e:
                    self.logger.warning(f"Could not load checkpoint {checkpoint_file}: {e}")
                    existing_results = []
                    start_iteration = 0
            
            attack_bootstrap_results = existing_results.copy()
            successful_iterations = len(existing_results)
            failed_iterations = 0
            
            # Create progress bar starting from checkpoint
            with tqdm(range(start_iteration, n_bootstrap), 
                     desc=f"Bootstrap {attack_level}", unit="iteration",
                     initial=start_iteration, total=n_bootstrap) as progress_bar:
                
                for bootstrap_iter in progress_bar:
                    iteration_start_time = datetime.now()
                    
                    try:
                        self.logger.info(f"Starting iteration {bootstrap_iter + 1}/{n_bootstrap}")
                        
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        

                        np.random.seed(42 + bootstrap_iter)

                        resampled_speakers = np.random.choice(
                            enrollment_speakers, 
                            size=len(enrollment_speakers), 
                            replace=True

                        )
                        
                        # Log details for first 3 iterations
                        if bootstrap_iter < 3:
                            from collections import Counter
                            speaker_counts = Counter(resampled_speakers)
                            unique_speakers = len(set(resampled_speakers))
                            max_repeats = max(speaker_counts.values()) if speaker_counts else 0
                            
                            self.logger.info(f"  Bootstrap iteration {bootstrap_iter + 1}:")
                            self.logger.info(f"    - Unique speakers: {unique_speakers}/{len(enrollment_speakers)}")
                            self.logger.info(f"    - Max speaker repeats: {max_repeats}")
                        
                        # Collect trial pairs from resampled speakers
                        bootstrap_trial_pairs = []
                        for speaker in resampled_speakers:
                            speaker_pairs = speaker_trial_pairs[speaker]
                            bootstrap_trial_pairs.append(speaker_pairs)
                        
                        bootstrap_asv_pairs = pd.concat(bootstrap_trial_pairs, ignore_index=True)
                        
                        # Evaluate this attack level on bootstrap sample
                        attack_result = self.evaluate_attack_level_bootstrap(
                            bootstrap_asv_pairs, chunk_audio_mapping, speaker_embeddings, attack_level
                        )
                        
                        if attack_result is not None:
                            iteration_duration = (datetime.now() - iteration_start_time).total_seconds() / 60
                            
                            result_entry = {
                                'bootstrap_iteration': bootstrap_iter + 1,
                                'attack_level': attack_level,
                                'eer': attack_result['eer'],
                                'auc': attack_result['auc'],
                                'threshold': attack_result['threshold'],
                                'num_trials': attack_result['num_trials'],
                                'num_speakers': len(resampled_speakers),
                                'num_unique_speakers': len(set(resampled_speakers)),
                                'iteration_duration_minutes': round(iteration_duration, 2),
                                'timestamp': datetime.now().isoformat()
                            }
                            attack_bootstrap_results.append(result_entry)
                            successful_iterations += 1
                            
                            # Save checkpoint
                            try:
                                checkpoint_df = pd.DataFrame(attack_bootstrap_results)
                                checkpoint_df.to_csv(checkpoint_file, index=False)
                                self.logger.info(f"Checkpoint saved: {bootstrap_iter + 1} iterations completed")
                            except Exception as checkpoint_error:
                                self.logger.error(f"Failed to save checkpoint: {checkpoint_error}")
                            
                            current_eers = [r['eer'] for r in attack_bootstrap_results]
                            current_mean = np.mean(current_eers) if current_eers else 0
                            progress_bar.set_postfix({
                                'EER': f"{attack_result['eer']:.4f}",
                                'Mean': f"{current_mean:.4f}",
                                'Success': successful_iterations,
                                'Duration': f"{iteration_duration:.1f}min"
                            })
                            
                            self.logger.info(f"Iteration {bootstrap_iter + 1} completed: "
                                           f"EER = {attack_result['eer']:.4f}, "
                                           f"Duration = {iteration_duration:.1f} min")
                        else:
                            failed_iterations += 1
                            self.logger.warning(f"Bootstrap iteration {bootstrap_iter + 1} failed - no result")
                        
                    except KeyboardInterrupt:
                        self.logger.info("Bootstrap interrupted by user")
                        break
                    except Exception as e:
                        failed_iterations += 1
                        iteration_duration = (datetime.now() - iteration_start_time).total_seconds() / 60
                        self.logger.error(f"Bootstrap {attack_level} iteration {bootstrap_iter + 1} failed after {iteration_duration:.1f} min: {str(e)}")
                        
                        try:
                            checkpoint_df = pd.DataFrame(attack_bootstrap_results)
                            checkpoint_df.to_csv(checkpoint_file, index=False)
                        except:
                            pass
                        continue
            
            # Calculate final statistics for this attack level
            if attack_bootstrap_results:
                eers = [r['eer'] for r in attack_bootstrap_results]
                aucs = [r['auc'] for r in attack_bootstrap_results]
                
                statistics = {
                    'attack_level': attack_level,
                    'attack_level_name': self.attack_levels[attack_level],
                    'method': 'bootstrap_speaker_level',
                    'n_bootstrap_iterations': len(attack_bootstrap_results),
                    'n_bootstrap_attempted': n_bootstrap,
                    'success_rate': len(attack_bootstrap_results) / n_bootstrap,
                    'eer_mean': np.mean(eers),
                    'eer_std': np.std(eers),
                    'eer_min': np.min(eers),
                    'eer_max': np.max(eers),
                    'eer_median': np.median(eers),
                    'eer_25th_percentile': np.percentile(eers, 25),
                    'eer_75th_percentile': np.percentile(eers, 75),
                    'confidence_interval_95': np.percentile(eers, [2.5, 97.5]).tolist(),
                    'auc_mean': np.mean(aucs),
                    'auc_std': np.std(aucs),
                    'total_duration_hours': sum([r.get('iteration_duration_minutes', 0) for r in attack_bootstrap_results]) / 60,
                    'evaluation_date': datetime.now().isoformat()
                }
                
                bootstrap_results[attack_level] = {
                    'individual_results': attack_bootstrap_results,
                    'statistics': statistics
                }
                
                self.logger.info(f"Attack {attack_level} completed:")
                self.logger.info(f"  - Successful iterations: {successful_iterations}/{n_bootstrap}")
                self.logger.info(f"  - EER: {statistics['eer_mean']:.4f} ± {statistics['eer_std']:.4f}")
                self.logger.info(f"  - 95% CI: [{statistics['confidence_interval_95'][0]:.4f}, "
                               f"{statistics['confidence_interval_95'][1]:.4f}]")
                self.logger.info(f"  - Total duration: {statistics['total_duration_hours']:.1f} hours")
            else:
                self.logger.error(f"All bootstrap iterations failed for attack level {attack_level}")
                bootstrap_results[attack_level] = None
        
        return bootstrap_results


    def evaluate_attack_level_bootstrap(self, bootstrap_asv_pairs: pd.DataFrame, chunk_audio_mapping: Dict, 
                                       speaker_embeddings: Dict, attack_level: str) -> Dict[str, Any]:
        """
        Evaluate ASV performance on bootstrap sample
        """
        # Determine audio types
        if attack_level == 'oo':
            enrollment_type = 'original'
            trial_type = 'original'
        elif attack_level == 'oa':
            enrollment_type = 'original'
            trial_type = 'anonymized'
        elif attack_level == 'aa':
            enrollment_type = 'anonymized'
            trial_type = 'anonymized'
        else:
            raise ValueError(f"Unknown attack level: {attack_level}")
        
        positive_scores = []  
        negative_scores = []
        valid_trials = 0
        failed_trials = 0
        
        batch_size = 1000
        total_pairs = len(bootstrap_asv_pairs)
        
        for batch_start in range(0, total_pairs, batch_size):
            batch_end = min(batch_start + batch_size, total_pairs)
            batch_pairs = bootstrap_asv_pairs.iloc[batch_start:batch_end]
            
            for _, row in batch_pairs.iterrows():
                try:
                    # Get enrollment embedding
                    enrollment_speaker_id = row['enrollment_speaker_id']
                    if enrollment_speaker_id not in speaker_embeddings:
                        failed_trials += 1
                        continue
                    
                    enrollment_embedding = speaker_embeddings[enrollment_speaker_id][enrollment_type]
                    if enrollment_embedding is None:
                        failed_trials += 1
                        continue
                    
                    # Get trial embedding
                    trial_utterance_id = row['trial_utterance_id']
                    if trial_utterance_id not in chunk_audio_mapping:
                        failed_trials += 1
                        continue
                    
                    trial_audio_path = chunk_audio_mapping[trial_utterance_id][trial_type]
                    if not trial_audio_path or not Path(trial_audio_path).exists():
                        failed_trials += 1
                        continue
                    
                    # Extract trial embedding
                    trial_embedding = self.extract_embedding_from_audio(trial_audio_path)
                    if trial_embedding is None:
                        failed_trials += 1
                        continue
                    
                    # Compute similarity score
                    score = self.compute_cosine_similarity(enrollment_embedding, trial_embedding)
                    if score is None:
                        failed_trials += 1
                        continue
                    
                    # Collect scores for EER calculation
                    same_speaker_label = row['same_speaker_label']
                    if same_speaker_label == 1:
                        positive_scores.append(score)
                    else:
                        negative_scores.append(score)
                    
                    valid_trials += 1
                    
                except Exception as e:
                    failed_trials += 1
                    continue
            
            # Clean up
            del batch_pairs
            
            # Log progress every 10 batches
            if (batch_start // batch_size) % 10 == 0:
                progress = (batch_end / total_pairs) * 100
                self.logger.debug(f"Batch processing: {progress:.1f}% complete, "
                                f"valid_trials: {valid_trials}, failed: {failed_trials}")
        
        if valid_trials == 0 or len(positive_scores) == 0 or len(negative_scores) == 0:
            return None
        
        # Calculate EER
        try:
            eer, threshold = self.compute_eer_speechbrain(positive_scores, negative_scores)
            
            # Calculate AUC
            all_scores = positive_scores + negative_scores
            all_labels = [1] * len(positive_scores) + [0] * len(negative_scores)
            fpr, tpr, _ = roc_curve(all_labels, all_scores, pos_label=1)
            auc_score = auc(fpr, tpr)
            
            return {
                'attack_level': attack_level,
                'eer': eer,
                'auc': auc_score,
                'threshold': threshold,
                'num_trials': valid_trials,
                'num_same_speaker': len(positive_scores),
                'num_different_speaker': len(negative_scores),
                'failed_trials': failed_trials
            }
            
        except Exception as e:
            self.logger.error(f"Error computing EER for {attack_level}: {e}")
            return None
    
    
    def run_multiple_asv_evaluations(self, asv_pairs: pd.DataFrame, chunk_audio_mapping: Dict) -> Dict[str, Any]:
        """
        Run multiple ASV evaluations
        """
        num_runs = self.config['asv_model_evaluation']['num_evaluation_runs']
        self.logger.info(f"Starting {num_runs} ASV evaluation runs with fresh embeddings per run")
        
        # Collect all run results
        all_run_results = []
        
        with tqdm(range(num_runs), desc="ASV Evaluation Runs", unit="run") as progress_bar:
            for run_num in progress_bar:
                try:
                    torch.manual_seed(42 + run_num)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(42 + run_num)
                    
                    self.logger.info(f"=== Starting Run {run_num + 1}/{num_runs} ===")
                    
                    # Create embeddings
                    self.logger.info(f"Computing fresh enrollment embeddings for run {run_num + 1}")
                    speaker_embeddings = self.create_enrollment_embeddings(asv_pairs, chunk_audio_mapping)
                    
                    # Evaluate all attack levels
                    run_results = {
                        'run_number': run_num + 1,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    for attack_level in ['oo', 'oa', 'aa']:
                        self.logger.info(f"Run {run_num + 1}: Evaluating attack level {attack_level}")
                        
                        attack_result = self.evaluate_attack_level(
                            asv_pairs, chunk_audio_mapping, speaker_embeddings, attack_level
                        )
                        
                        if attack_result is not None:
                            run_results[attack_level] = {
                                'eer': attack_result['eer'],
                                'auc': attack_result['auc'],
                                'threshold': attack_result['threshold'],
                                'num_trials': attack_result['num_trials'],
                                'num_same_speaker': attack_result['num_same_speaker'],
                                'num_different_speaker': attack_result['num_different_speaker'],
                                'score_statistics': attack_result['score_statistics']
                            }
                        else:
                            run_results[attack_level] = None
                            self.logger.error(f"Run {run_num + 1}: Failed to evaluate attack level {attack_level}")
                    
                    all_run_results.append(run_results)
                    
                    if all(run_results.get(level) is not None for level in ['oo', 'oa', 'aa']):
                        current_eers = {level: run_results[level]['eer'] for level in ['oo', 'oa', 'aa']}
                        progress_bar.set_postfix({
                            'oo': f"{current_eers['oo']:.4f}",
                            'oa': f"{current_eers['oa']:.4f}", 
                            'aa': f"{current_eers['aa']:.4f}"
                        })
                    
                    self.logger.info(f"=== Completed Run {run_num + 1}/{num_runs} ===")
                    
                except Exception as e:
                    self.logger.error(f"Error in ASV run {run_num + 1}: {str(e)}")
                    continue
        
        # Reorganize results by attack level for statistics
        self.logger.info("Computing statistics across all runs...")
        
        all_attack_results = {}
        for attack_level in ['oo', 'oa', 'aa']:
            # Collect results for this attack level across all runs
            individual_results = []
            eers = []
            aucs = []
            
            for run_result in all_run_results:
                if attack_level in run_result and run_result[attack_level] is not None:
                    attack_data = run_result[attack_level]
                    individual_result = {
                        'run_number': run_result['run_number'],
                        'attack_level': attack_level,
                        'eer': attack_data['eer'],
                        'auc': attack_data['auc'],
                        'threshold': attack_data['threshold'],
                        'num_trials': attack_data['num_trials'],
                        'timestamp': run_result['timestamp']
                    }
                    individual_results.append(individual_result)
                    eers.append(attack_data['eer'])
                    aucs.append(attack_data['auc'])
            
            # Calculate statistics for this attack level
            if eers:
                statistics = {
                    'attack_level': attack_level,
                    'attack_level_name': self.attack_levels[attack_level],
                    'num_runs': len(eers),
                    'eer_mean': np.mean(eers),
                    'eer_std': np.std(eers),
                    'eer_min': np.min(eers),
                    'eer_max': np.max(eers),
                    'eer_median': np.median(eers),
                    'eer_25th_percentile': np.percentile(eers, 25),
                    'eer_75th_percentile': np.percentile(eers, 75),
                    'auc_mean': np.mean(aucs),
                    'auc_std': np.std(aucs),
                    'evaluation_date': datetime.now().isoformat()
                }
                
                attack_results = {
                    'individual_results': individual_results,
                    'statistics': statistics
                }
                
                self.logger.info(f"Attack {attack_level} completed: "
                               f"EER = {statistics['eer_mean']:.4f} ± {statistics['eer_std']:.4f}")
            else:
                self.logger.error(f"No successful runs for attack level {attack_level}")
                attack_results = None
            
            all_attack_results[attack_level] = attack_results
        
        return all_attack_results
    
    def evaluate_attack_level(self, asv_pairs: pd.DataFrame, chunk_audio_mapping: Dict, 
                             speaker_embeddings: Dict, attack_level: str) -> Dict[str, Any]:
        """
        Evaluate ASV performance for specific attack level
        """
        self.logger.debug(f"Evaluating attack level: {attack_level}")
        
        # Determine enrollment and trial audio types based on attack level
        if attack_level == 'oo':
            enrollment_type = 'original'
            trial_type = 'original'
        elif attack_level == 'oa':
            enrollment_type = 'original'
            trial_type = 'anonymized'
        elif attack_level == 'aa':
            enrollment_type = 'anonymized'
            trial_type = 'anonymized'
        else:
            raise ValueError(f"Unknown attack level: {attack_level}")
        
        self.logger.debug(f"Attack level {attack_level}: enrollment_type={enrollment_type}, trial_type={trial_type}")
        
        results = []
        positive_scores = []  
        negative_scores = [] 
        valid_trials = 0
        failed_trials = 0
        
        all_scores = []
        same_speaker_count = 0
        diff_speaker_count = 0
        
        for idx, (_, row) in enumerate(asv_pairs.iterrows()):
            trial_id = row['trial_id']
            enrollment_speaker_id = row['enrollment_speaker_id']
            trial_utterance_id = row['trial_utterance_id']
            same_speaker_label = row['same_speaker_label']
            
            try:
                # Get enrollment embedding
                if enrollment_speaker_id not in speaker_embeddings:
                    failed_trials += 1
                    continue
                
                enrollment_embedding = speaker_embeddings[enrollment_speaker_id][enrollment_type]
                if enrollment_embedding is None:
                    failed_trials += 1
                    continue
                
                # Get trial audio path and extract embedding
                if trial_utterance_id not in chunk_audio_mapping:
                    failed_trials += 1
                    continue
                
                trial_audio_path = chunk_audio_mapping[trial_utterance_id][trial_type]
                if not trial_audio_path or not Path(trial_audio_path).exists():
                    failed_trials += 1
                    continue
                
                # Extract trial embedding
                trial_embedding = self.extract_embedding_from_audio(trial_audio_path)
                if trial_embedding is None:
                    failed_trials += 1
                    continue
                
                # Cosine similarity computation
                score = self.compute_cosine_similarity(enrollment_embedding, trial_embedding)
                if score is None:
                    failed_trials += 1
                    continue
                
                results.append({
                    'trial_id': trial_id,
                    'enrollment_speaker_id': enrollment_speaker_id,
                    'same_speaker_label': same_speaker_label,
                    'score': float(score),
                    'attack_level': attack_level
                })
                
                # Collect scores for EER calculation
                if same_speaker_label == 1:
                    positive_scores.append(score)
                    same_speaker_count += 1
                else:
                    negative_scores.append(score)
                    diff_speaker_count += 1
                
                all_scores.append(score)
                valid_trials += 1
                
            except Exception as e:
                self.logger.warning(f"Failed to process trial {trial_id}: {str(e)}")
                failed_trials += 1
                continue
        
        self.logger.debug(f"Attack level {attack_level}: {valid_trials} valid trials, {failed_trials} failed trials")
        
        if valid_trials == 0:
            self.logger.error(f"No valid trials for attack level {attack_level}")
            return None
        
        if len(positive_scores) == 0 or len(negative_scores) == 0:
            self.logger.error(f"No positive or negative scores for attack level {attack_level}")
            self.logger.error(f"Same speaker trials: {same_speaker_count}, Different speaker trials: {diff_speaker_count}")
            return None
        
        # Calculate EER
        eer, threshold = self.compute_eer_speechbrain(positive_scores, negative_scores)
        
        # Calculate AUC for additional metrics
        all_scores = positive_scores + negative_scores
        all_labels = [1] * len(positive_scores) + [0] * len(negative_scores)
        fpr, tpr, _ = roc_curve(all_labels, all_scores, pos_label=1)
        auc_score = auc(fpr, tpr)
        
        # Calculate confusion matrix
        predictions = [1 if score >= threshold else 0 for score in all_scores]
        tp = sum((pred == 1 and label == 1) for pred, label in zip(predictions, all_labels))
        tn = sum((pred == 0 and label == 0) for pred, label in zip(predictions, all_labels))
        fp = sum((pred == 1 and label == 0) for pred, label in zip(predictions, all_labels))
        fn = sum((pred == 0 and label == 1) for pred, label in zip(predictions, all_labels))
        
        return {
            'attack_level': attack_level,
            'eer': eer,
            'auc': auc_score,
            'threshold': threshold,
            'num_trials': valid_trials,
            'num_same_speaker': len(positive_scores),
            'num_different_speaker': len(negative_scores),
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'results_df': pd.DataFrame(results),
            'fpr': fpr,
            'tpr': tpr,
            'positive_scores': positive_scores,
            'negative_scores': negative_scores,
            'score_statistics': {
                'same_speaker_mean': np.mean(positive_scores),
                'same_speaker_std': np.std(positive_scores),
                'diff_speaker_mean': np.mean(negative_scores),
                'diff_speaker_std': np.std(negative_scores),
                'separation': np.mean(positive_scores) - np.mean(negative_scores)
            }
        }
    
    def save_results_csv(self, all_results: Dict[str, Any]):
        """
        Save detailed ASV results
        """
        self.logger.info("Saving ASV results to CSV files")
        
        is_bootstrap = False
        if isinstance(all_results, dict):
            # Check for bootstrap results structure
            for attack_level in ['oo', 'oa', 'aa']:
                if attack_level in all_results and all_results[attack_level]:
                    if 'statistics' in all_results[attack_level]:
                        if all_results[attack_level]['statistics'].get('method') == 'bootstrap_speaker_level':
                            is_bootstrap = True
                            break
        
        if is_bootstrap:
            self.logger.info("Saving BOOTSTRAP results")
            
            # Save individual bootstrap iterations
            all_bootstrap_iterations = []
            for attack_level in ['oo', 'oa', 'aa']:
                if attack_level in all_results and all_results[attack_level]:
                    results = all_results[attack_level]
                    if 'individual_results' in results:
                        # Bootstrap iterations are stored here
                        for iteration in results['individual_results']:
                            iteration_data = {
                                'method': 'bootstrap_speaker_level',
                                'attack_level': attack_level,
                                'attack_level_name': self.attack_levels[attack_level],
                                'bootstrap_iteration': iteration.get('bootstrap_iteration', 0),
                                'eer': iteration['eer'],
                                'auc': iteration['auc'],
                                'threshold': iteration['threshold'],
                                'num_trials': iteration['num_trials'],
                                'num_speakers': iteration.get('num_speakers', 0),
                                'num_unique_speakers': iteration.get('num_unique_speakers', 0)
                            }
                            all_bootstrap_iterations.append(iteration_data)
            
            if all_bootstrap_iterations:
                # Save individual bootstrap iterations
                bootstrap_df = pd.DataFrame(all_bootstrap_iterations)
                bootstrap_iterations_path = self.final_dir / 'asv_bootstrap_iterations.csv'
                bootstrap_df.to_csv(bootstrap_iterations_path, index=False)
                self.logger.info(f"Bootstrap iterations saved to: {bootstrap_iterations_path}")
                
                # Log summary of bootstrap iterations per attack level
                self.logger.info("Bootstrap Iterations Summary:")
                for attack_level in ['oo', 'oa', 'aa']:
                    level_data = bootstrap_df[bootstrap_df['attack_level'] == attack_level]
                    if not level_data.empty:
                        self.logger.info(f"  {self.attack_levels[attack_level]} ({attack_level}):")
                        self.logger.info(f"    Iterations: {len(level_data)}")
                        eer_values = level_data['eer'].tolist()
                        self.logger.info(f"    EER values: {[f'{e:.4f}' for e in eer_values]}")
                        self.logger.info(f"    EER range: [{min(eer_values):.4f}, {max(eer_values):.4f}]")
        
        else:
            self.logger.info("Saving STANDARD multiple-run results")
            
            # Save individual results from standard runs
            all_individual_results = []
            for attack_level, results in all_results.items():
                if results is not None and 'individual_results' in results:
                    all_individual_results.extend(results['individual_results'])
            
            if all_individual_results:
                individual_df = pd.DataFrame(all_individual_results)
                individual_results_path = self.final_dir / 'asv_individual_results.csv'
                individual_df.to_csv(individual_results_path, index=False)
                self.logger.info(f"Individual ASV results saved to: {individual_results_path}")
        
        # Create and save summary statistics (for both bootstrap and standard)
        summary_data = []
        for attack_level, results in all_results.items():
            if results is not None and 'statistics' in results:
                stats = results['statistics']
                summary_entry = {
                    'attack_level': attack_level,
                    'attack_level_name': stats.get('attack_level_name', self.attack_levels.get(attack_level, 'unknown')),
                    'method': stats.get('method', 'standard'),
                    'num_runs': stats.get('num_runs', stats.get('n_bootstrap_iterations', 0)),
                    'eer_mean': stats['eer_mean'],
                    'eer_std': stats['eer_std'],
                    'eer_min': stats['eer_min'],
                    'eer_max': stats['eer_max'],
                    'eer_median': stats['eer_median'],
                    'eer_25th_percentile': stats['eer_25th_percentile'],
                    'eer_75th_percentile': stats['eer_75th_percentile'],
                    'auc_mean': stats['auc_mean'],
                    'auc_std': stats['auc_std'],
                    'evaluation_date': stats['evaluation_date']
                }
                
                # Add confidence intervals if available (bootstrap)
                if 'confidence_interval_95' in stats:
                    summary_entry['ci_95_lower'] = stats['confidence_interval_95'][0]
                    summary_entry['ci_95_upper'] = stats['confidence_interval_95'][1]
                
                summary_data.append(summary_entry)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = self.final_dir / 'asv_results.csv'
            summary_df.to_csv(summary_path, index=False)
            self.logger.info(f"ASV summary results saved to: {summary_path}")
    
    def create_evaluation_plots(self, all_results: Dict[str, Any]):
        """
        Create visualization plots for ASV evaluation results
        """
        self.logger.info("Creating ASV evaluation visualization plots")
        
        is_bootstrap = False
        for attack_level in ['oo', 'oa', 'aa']:
            if attack_level in all_results and all_results[attack_level]:
                if 'statistics' in all_results[attack_level]:
                    if all_results[attack_level]['statistics'].get('method') == 'bootstrap_speaker_level':
                        is_bootstrap = True
                        break
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        title_suffix = "Bootstrap" if is_bootstrap else "Multiple Runs"
        fig.suptitle(f'ASV Evaluation Results - Privacy Attack Analysis ({title_suffix})', fontsize=16)
        
        # Color mapping for attack levels
        colors = {'oo': 'blue', 'oa': 'orange', 'aa': 'red'}
        labels = {
            'oo': 'Unprotected (Original vs Original)',
            'oa': 'Ignorant Attack (Original vs Anonymized)', 
            'aa': 'Lazy-Informed Attack (Anonymized vs Anonymized)'
        }
        
        # 1. EER Comparison with Error Bars
        attack_levels = []
        eer_means = []
        eer_stds = []
        colors_list = []
        
        for attack_level, results in all_results.items():
            if results is not None and 'statistics' in results:
                stats = results['statistics']
                attack_levels.append(labels[attack_level])
                eer_means.append(stats['eer_mean'])
                eer_stds.append(stats['eer_std'])
                colors_list.append(colors[attack_level])
        
        if eer_means:
            bars = axes[0, 0].bar(range(len(attack_levels)), eer_means, yerr=eer_stds, 
                                 color=colors_list, alpha=0.7, capsize=5)
            axes[0, 0].set_xlabel('Attack Level')
            axes[0, 0].set_ylabel('Equal Error Rate (EER)')
            axes[0, 0].set_title('EER Comparison with Standard Deviation')
            axes[0, 0].set_xticks(range(len(attack_levels)))
            axes[0, 0].set_xticklabels(attack_levels, rotation=45, ha='right')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, mean, std in zip(bars, eer_means, eer_stds):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.001,
                               f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        # 2. EER Distribution Box Plots
        eer_data = []
        eer_labels = []
        for attack_level, results in all_results.items():
            if results is not None and 'individual_results' in results:
                individual_results = results['individual_results']

                if is_bootstrap:
                    eers = [r['eer'] for r in individual_results]
                else:
                    eers = [r['eer'] for r in individual_results if 'eer' in r]
                
                if eers:
                    eer_data.append(eers)
                    eer_labels.append(labels[attack_level])
        
        if eer_data:
            bp = axes[0, 1].boxplot(eer_data, labels=eer_labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors_list[:len(eer_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            axes[0, 1].set_ylabel('Equal Error Rate (EER)')
            axes[0, 1].set_title('EER Distribution Across Iterations')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. EER Progression Over Iterations
        for attack_level, results in all_results.items():
            if results is not None and 'individual_results' in results:
                individual_results = results['individual_results']
                if individual_results:
                    if is_bootstrap:
                        iteration_numbers = [r['bootstrap_iteration'] for r in individual_results]
                        eers = [r['eer'] for r in individual_results]
                        x_label = 'Bootstrap Iteration'
                        title = 'EER Progression Over Bootstrap Iterations'
                    else:
                        iteration_numbers = [r['run_number'] for r in individual_results]
                        eers = [r['eer'] for r in individual_results]
                        x_label = 'Run Number'
                        title = 'EER Progression Over Runs'
                    
                    axes[0, 2].plot(iteration_numbers, eers, color=colors[attack_level], 
                                  alpha=0.6, linewidth=1, label=labels[attack_level], marker='o', markersize=3)
                    
                    if len(eers) > 5:
                        running_avg = pd.Series(eers).rolling(window=min(5, len(eers)//2), min_periods=1).mean()
                        axes[0, 2].plot(iteration_numbers, running_avg, color=colors[attack_level], 
                                      linewidth=2, linestyle='--', alpha=0.8)
        
        axes[0, 2].set_xlabel(x_label if 'x_label' in locals() else 'Iteration')
        axes[0, 2].set_ylabel('EER')
        axes[0, 2].set_title(title if 'title' in locals() else 'EER Progression')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. AUC Comparison (if available)
        auc_means = []
        auc_stds = []
        
        for attack_level, results in all_results.items():
            if results is not None and 'statistics' in results:
                stats = results['statistics']
                auc_means.append(stats.get('auc_mean', 0))
                auc_stds.append(stats.get('auc_std', 0))
        
        if auc_means and any(auc > 0 for auc in auc_means):
            bars = axes[1, 0].bar(range(len(attack_levels)), auc_means, yerr=auc_stds, 
                                 color=colors_list, alpha=0.7, capsize=5)
            axes[1, 0].set_xlabel('Attack Level')
            axes[1, 0].set_ylabel('Area Under Curve (AUC)')
            axes[1, 0].set_title('AUC Comparison with Standard Deviation')
            axes[1, 0].set_xticks(range(len(attack_levels)))
            axes[1, 0].set_xticklabels(attack_levels, rotation=45, ha='right')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add value labels
            for bar, mean, std in zip(bars, auc_means, auc_stds):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                               f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        else:
            axes[1, 0].text(0.5, 0.5, 'AUC data not available', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('AUC Comparison')
        
        # 5. Statistics Summary Table
        axes[1, 1].axis('off')
        table_data = []
        
        if is_bootstrap:
            headers = ['Attack Level', 'EER (Mean±Std)', 'Iterations', '95% CI', 'Duration (h)']
            
            for attack_level, results in all_results.items():
                if results is not None and 'statistics' in results:
                    stats = results['statistics']
                    ci = stats.get('confidence_interval_95', [0, 0])
                    duration = stats.get('total_duration_hours', 0)
                    
                    table_data.append([
                        labels[attack_level].replace(' (', '\n('),
                        f"{stats['eer_mean']:.3f}±{stats['eer_std']:.3f}",
                        f"{stats['n_bootstrap_iterations']}",
                        f"[{ci[0]:.3f}, {ci[1]:.3f}]",
                        f"{duration:.1f}h"
                    ])
        else:
            headers = ['Attack Level', 'EER (Mean±Std)', 'AUC (Mean±Std)', 'Runs', 'Min-Max EER']
            
            for attack_level, results in all_results.items():
                if results is not None and 'statistics' in results:
                    stats = results['statistics']
                    table_data.append([
                        labels[attack_level].replace(' (', '\n('),
                        f"{stats['eer_mean']:.3f}±{stats['eer_std']:.3f}",
                        f"{stats.get('auc_mean', 0):.3f}±{stats.get('auc_std', 0):.3f}",
                        f"{stats.get('num_runs', stats.get('n_bootstrap_iterations', 0))}",
                        f"{stats['eer_min']:.3f}-{stats['eer_max']:.3f}"
                    ])
        
        if table_data:
            table = axes[1, 1].table(cellText=table_data, colLabels=headers,
                                   cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.scale(1, 2)
            method_name = "Bootstrap" if is_bootstrap else "Statistical"
            axes[1, 1].set_title(f'ASV Performance Summary ({method_name})', pad=20)
        
        # 6. Privacy Protection Analysis
        if all_results.get('oo') and all_results.get('oa') and all_results.get('aa'):
            oo_stats = all_results['oo']['statistics']
            oa_stats = all_results['oa']['statistics']
            aa_stats = all_results['aa']['statistics']
            
            # Calculate privacy protection (EER degradation)
            privacy_protection_oa = oa_stats['eer_mean'] - oo_stats['eer_mean']
            privacy_protection_aa = aa_stats['eer_mean'] - oo_stats['eer_mean']
            
            protection_data = [privacy_protection_oa, privacy_protection_aa]
            protection_labels = ['Ignorant Attack\n(Original vs Anon)', 'Lazy-Informed Attack\n(Anon vs Anon)']
            protection_colors = ['orange', 'red']
            
            bars = axes[1, 2].bar(range(len(protection_labels)), protection_data, 
                                 color=protection_colors, alpha=0.7)
            axes[1, 2].set_xlabel('Attack Type')
            axes[1, 2].set_ylabel('Privacy Protection (ΔEER)')
            axes[1, 2].set_title('Privacy Protection Analysis')
            axes[1, 2].set_xticks(range(len(protection_labels)))
            axes[1, 2].set_xticklabels(protection_labels)
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # Add value labels
            for bar, value in zip(bars, protection_data):
                axes[1, 2].text(bar.get_x() + bar.get_width()/2, 
                               bar.get_height() + (0.01 if value >= 0 else -0.01),
                               f'{value:.3f}', ha='center', 
                               va='bottom' if value >= 0 else 'top', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        suffix = "bootstrap" if is_bootstrap else "standard"
        plot_path = self.results_dir / f'asv_{suffix}_evaluation_results.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"ASV evaluation plots saved to {plot_path}")
    
    
    def run_asv_evaluation_pipeline(self, asv_pairs_path: str, chunks_path: str) -> Dict[str, Any]:
        """
        Run complete ASV evaluation pipeline
        """
        self.logger.info("="*80)
        self.logger.info("ASV EVALUATION PIPELINE STARTING")
        self.logger.info("="*80)
        
        use_bootstrap = self.config['asv_evaluation'].get('use_bootstrap_analysis', False)
        n_bootstrap = self.config['asv_evaluation'].get('n_bootstrap_iterations', 1000)
        
        self.logger.info("Configuration:")
        self.logger.info(f"  - Bootstrap Enabled: {use_bootstrap}")
        if use_bootstrap:
            self.logger.info(f"  - Bootstrap Iterations: {n_bootstrap}")
            self.logger.info(f"  - Method: Speaker-level resampling with replacement")
        else:
            num_runs = self.config['asv_model_evaluation']['num_evaluation_runs']
            self.logger.info(f"  - Standard Evaluation Runs: {num_runs}")
            self.logger.info(f"  - Method: Independent runs with fresh embeddings")
        
        try:
            # 1. Load SpeechBrain EncoderClassifier model
            self.logger.info("\nStep 1: Loading SpeechBrain ECAPA-TDNN model...")
            self.load_speechbrain_model()
            
            # 2. Load ASV pairs and chunks data
            self.logger.info("\nStep 2: Loading ASV pairs and chunks data...")
            asv_pairs, chunk_audio_mapping = self.load_asv_pairs_and_chunks(asv_pairs_path, chunks_path)
            
            # 3. Choose evaluation method based on config
            if use_bootstrap:
                self.logger.info("\n" + "="*60)
                self.logger.info("USING BOOTSTRAP SPEAKER-LEVEL ANALYSIS")
                self.logger.info("="*60)
                
                # Create enrollment embeddings
                self.logger.info("\nStep 3: Computing enrollment embeddings (once for all bootstrap iterations)...")
                speaker_embeddings = self.create_enrollment_embeddings(asv_pairs, chunk_audio_mapping)
                
                # Run bootstrap analysis
                self.logger.info(f"\nStep 4: Running bootstrap analysis with {n_bootstrap} iterations per attack level...")
                all_results = self.bootstrap_asv_analysis(
                    asv_pairs, chunk_audio_mapping, speaker_embeddings, n_bootstrap
                )
                
                # Mark results as bootstrap
                for attack_level in all_results:
                    if all_results[attack_level] and 'statistics' in all_results[attack_level]:
                        all_results[attack_level]['statistics']['evaluation_method'] = 'bootstrap_speaker_level'
                
            else:
                self.logger.info("\n" + "="*60)
                self.logger.info("USING STANDARD MULTIPLE-RUNS ANALYSIS")
                self.logger.info("="*60)
                
                # Run standard multiple evaluations
                self.logger.info(f"\nStep 3: Running {self.config['asv_model_evaluation']['num_evaluation_runs']} independent evaluation runs...")
                all_results = self.run_multiple_asv_evaluations(asv_pairs, chunk_audio_mapping)
                
                # Mark results as standard
                for attack_level in all_results:
                    if all_results[attack_level] and 'statistics' in all_results[attack_level]:
                        all_results[attack_level]['statistics']['evaluation_method'] = 'standard_multiple_runs'
            
            # 4. Save results to CSV
            self.logger.info("\nStep 5: Saving results to CSV files...")
            self.save_results_csv(all_results)
            
            # 5. Create visualization plots
            self.logger.info("\nStep 6: Creating visualization plots...")
            self.create_evaluation_plots(all_results)
            
            # 6. Log final results
            self.logger.info("\n" + "="*80)
            self.logger.info("ASV EVALUATION RESULTS SUMMARY")
            self.logger.info("="*80)
            
            method_used = "Bootstrap (Speaker-level)" if use_bootstrap else "Standard (Multiple Runs)"
            self.logger.info(f"Evaluation Method: {method_used}")
            
            for attack_level in ['oo', 'oa', 'aa']:
                if attack_level in all_results and all_results[attack_level] is not None:
                    if 'statistics' in all_results[attack_level]:
                        stats = all_results[attack_level]['statistics']
                        attack_name = self.attack_levels[attack_level]
                        
                        self.logger.info(f"\n{attack_name} ({attack_level}):")
                        
                        if use_bootstrap:
                            self.logger.info(f"  Bootstrap iterations: {stats['n_bootstrap_iterations']}/{stats['n_bootstrap_attempted']}")
                            self.logger.info(f"  Success rate: {stats['success_rate']:.2%}")
                            self.logger.info(f"  EER: {stats['eer_mean']:.4f} ± {stats['eer_std']:.4f}")
                            if 'confidence_interval_95' in stats:
                                self.logger.info(f"  95% CI: [{stats['confidence_interval_95'][0]:.4f}, "
                                               f"{stats['confidence_interval_95'][1]:.4f}]")
                        else:
                            self.logger.info(f"  Runs: {stats['num_runs']}")
                            self.logger.info(f"  EER: {stats['eer_mean']:.4f} ± {stats['eer_std']:.4f}")
                            self.logger.info(f"  EER Range: [{stats['eer_min']:.4f}, {stats['eer_max']:.4f}]")
                        
                        self.logger.info(f"  AUC: {stats['auc_mean']:.4f} ± {stats['auc_std']:.4f}")
            
            self.logger.info("\n" + "="*80)
            self.logger.info("ASV EVALUATION PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("="*80)
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"ASV evaluation failed: {str(e)}")
            raise


def evaluate_asv_model(config: dict, logger, asv_pairs_path: str, chunks_path: str) -> Dict[str, Any]:
    """
    Main function to evaluate ASV performance for all ASV attack levels
    """
    evaluator = ASVEvaluator(config, logger)
    return evaluator.run_asv_evaluation_pipeline(asv_pairs_path, chunks_path)