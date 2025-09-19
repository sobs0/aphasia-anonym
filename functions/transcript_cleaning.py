import re
from typing import Tuple
import pandas as pd
from pathlib import Path
import nltk
from nltk.corpus import words

try:
    from nltk.corpus import words
    _ = words.words()
except LookupError:
    nltk.download('words')

ENGLISH_WORDS = set(w.lower() for w in words.words())

PHONETIC_REGEX = re.compile(r"[\u0250-\u02AF]")

CHAT_MAPPING = {
    "coulda": "could have", "gotta": "got to", "wanna": "want to", "mighta": "might have",
    "hadta": "had to", "needa": "need to", "musta": "must have", "hafta": "have to",
    "gonna": "going to", "shoulda": "should have", "hasta": "has to", "sposta": "supposed to",
    "woulda": "would have", "oughta": "ought to", "useta": "used to", "dunno": "don't know",
    "kinda": "kind of", "dyou": "do you", "sorta": "sort of", "gimme": "give me",
    "whyntcha": "why didn't you", "lemme": "let me", "wassup": "what's up", "lotsa": "lots of",
    "whaddya": "what did you", "caint": "can't", "hows about": "how about", "da": "the",
    "nutin": "nothing", "dan": "than", "sumpin": "something", "dat": "that", "ta": "to",
    "de": "the", "tagether": "together", "dese": "these", "tamorrow": "tomorrow", "deir": "their",
    "weunz": "we", "deirselves": "themselves", "whad": "what", "dem": "them", "wif": "with",
    "demselves": "themselves", "ya": "you", "den": "then", "yall": "you all", "dere": "there",
    "yer": "your", "dey": "they", "youse": "you all", "dis": "this", "yinz": "you all",
    "dose": "those", "younz": "you all", "fer": "for", "ze": "the", "git": "get",
    "zis": "this", "gon": "going", "zat": "that", "hisself": "himself"
}

def is_english_word(word: str) -> bool:
    return word.lower() in ENGLISH_WORDS

def contains_phonetic(word: str) -> bool:
    return bool(PHONETIC_REGEX.search(word))

def clean_ground_truth_transcript(original_string: str) -> Tuple[bool, str]:
    s = original_string

    # Rule 1: Reject strings with problematic CHAT markers
    if any(p in s for p in ["[< ", "[> ", "+<", "&*"]):
        return False, original_string

    # Rule 2: Handle [: ] annotations with phonetic or invalid words
    matches = list(re.finditer(r"\b(\S+)\s\[:\s([^\[\]]+?)\]", s))
    for match in matches:
        pre_word = match.group(1)
        post_word = match.group(2)
        if "@" in pre_word or contains_phonetic(pre_word) or not is_english_word(pre_word):
            if "@" in post_word:
                s = s.replace(match.group(0), "xxx")
                return False, s
            else:
                s = s.replace(match.group(0), post_word)

    # Rule 3: Remove all square bracket annotations
    s = re.sub(r"\[[^\[\]]*\]", "", s)

    # Rule 4: Clean @ words
    def clean_at_word(word):
        if "@x" in word:
            return "xxx"
        if "@" in word:
            return word.split("@")[0]
        return word
    s = " ".join(clean_at_word(w) for w in s.split())

    # Rule 5: Handle words starting with &- / &+ / &~ (remove prefix), 
    # all others with & are removed entirely
    def clean_amp_word(word):
        if word.startswith("&-") or word.startswith("&+") or word.startswith("&~"):
            return word[2:]
        elif word.startswith("&"):
            return None
        return word
    s = " ".join(filter(None, (clean_amp_word(w) for w in s.split())))

    # Rule 6: Remove parenthesis that contain only punctuation/numbers
    s = re.sub(r"\(([0-9.:,;+\-*/=^%]+)\)", "", s)

    # Rule 7: Remove words starting with 0 followed by a letter
    s = " ".join([w for w in s.split() if not re.match(r"^0[a-zA-Z]", w)])

    # Rule 8: Remove remaining parentheses
    s = s.replace("(", "").replace(")", "")

    # Rule 9: Replace informal CHAT words with canonical forms
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in CHAT_MAPPING.keys()) + r")\b")
    def replace_match(match):
        return CHAT_MAPPING.get(match.group(0), match.group(0))
    s = pattern.sub(replace_match, s)

    # Rule 10: Clean non-letter characters but preserve apostrophes and join letter:letter
    def clean_non_letters(text):
        # Protect number-based patterns (e.g. 3:25, 4.5)
        protected = re.findall(r"\d+[:.,]\d+", text)
        for i, p in enumerate(protected):
            text = text.replace(p, f"PROT{i}X")

        # Join any characters split by colon, like ha:llo → hallo
        text = re.sub(r"(?<=\w):(?=\w)", "", text)

        # Remove all but letters, digits, apostrophes (both ' and ')
        text = re.sub(r"[^a-zA-Z0-9'']", " ", text)

        for i, p in enumerate(protected):
            text = text.replace(f"PROT{i}X", p)

        return text

    s = clean_non_letters(s)

    # Rule 11: Normalize whitespace
    s = re.sub(r"\s+", " ", s)

    # Rule 12: Trim leading/trailing whitespace
    s = s.strip()

    # Rule 13: Reject if it contains placeholder tokens
    if any(word in s for word in ["xxx", "yyy", "www"]):
        return False, s

    return True, s


def clean_transcripts_and_update_usage(config, logger, length_marked_path):
    """
    Clean transcripts and update usage based on ground truth availability
    """
    logger.info("Starting transcript cleaning and usage update")
    
    # Load chunks data
    chunks_df = pd.read_csv(length_marked_path)
    logger.info(f"Processing {len(chunks_df)} chunks for transcript cleaning")
    
    # Initialize cleaned transcript column
    chunks_df['cleaned_transcript'] = None
    
    # Track usage changes
    usage_changes = {
        'ONLY_ASR_to_NO': 0,
        'BOTH_to_ONLY_ASV': 0,
        'valid_ground_truth': 0,
        'invalid_ground_truth': 0,
        'no_changes': 0
    }
    
    # Process each chunk
    for idx, row in chunks_df.iterrows():
        original_transcript = row['original_transcript']
        current_use = row['use']
        
        # Skip if usage is already NO
        if current_use == 'NO':
            chunks_df.at[idx, 'cleaned_transcript'] = ""
            usage_changes['no_changes'] += 1
            continue
        
        # Clean the transcript
        try:
            has_valid_ground_truth, cleaned_transcript = clean_ground_truth_transcript(original_transcript)
            chunks_df.at[idx, 'cleaned_transcript'] = cleaned_transcript
            
            if has_valid_ground_truth:
                # Keep original usage - transcript is valid
                usage_changes['valid_ground_truth'] += 1
            else:
                # No valid ground truth - update usage according to rules
                usage_changes['invalid_ground_truth'] += 1
                
                if current_use == 'ONLY_ASR':
                    # ONLY_ASR + no ground truth → NO
                    chunks_df.at[idx, 'use'] = 'NO'
                    usage_changes['ONLY_ASR_to_NO'] += 1
                elif current_use == 'BOTH':
                    # BOTH + no ground truth → ONLY_ASV
                    chunks_df.at[idx, 'use'] = 'ONLY_ASV'
                    usage_changes['BOTH_to_ONLY_ASV'] += 1
                    
        except Exception as e:
            logger.error(f"Failed to clean transcript for chunk {row['chunk_id']}: {str(e)}")
            # Treat as invalid ground truth
            chunks_df.at[idx, 'cleaned_transcript'] = ""
            usage_changes['invalid_ground_truth'] += 1
            
            if current_use == 'ONLY_ASR':
                chunks_df.at[idx, 'use'] = 'NO'
                usage_changes['ONLY_ASR_to_NO'] += 1
            elif current_use == 'BOTH':
                chunks_df.at[idx, 'use'] = 'ONLY_ASV'
                usage_changes['BOTH_to_ONLY_ASV'] += 1
    
    # Log detailed statistics
    total_chunks = len(chunks_df)
    logger.info(f"Transcript cleaning completed:")
    logger.info(f"  - Valid ground truth: {usage_changes['valid_ground_truth']} ({usage_changes['valid_ground_truth']/total_chunks*100:.1f}%)")
    logger.info(f"  - Invalid ground truth: {usage_changes['invalid_ground_truth']} ({usage_changes['invalid_ground_truth']/total_chunks*100:.1f}%)")
    
    logger.info(f"Usage changes due to missing ground truth:")
    logger.info(f"  - ONLY_ASR → NO: {usage_changes['ONLY_ASR_to_NO']}")
    logger.info(f"  - BOTH → ONLY_ASV: {usage_changes['BOTH_to_ONLY_ASV']}")
    logger.info(f"  - No changes (already NO): {usage_changes['no_changes']}")
    
    # Count final usage distribution
    usage_counts = chunks_df['use'].value_counts()
    logger.info(f"Final usage distribution:")
    for usage_type in ['NO', 'ONLY_ASR', 'ONLY_ASV', 'BOTH']:
        count = usage_counts.get(usage_type, 0)
        logger.info(f"  - {usage_type}: {count} ({count/total_chunks*100:.1f}%)")
    
    # Calculate final usable chunks
    asr_usable = usage_counts.get('ONLY_ASR', 0) + usage_counts.get('BOTH', 0)
    asv_usable = usage_counts.get('ONLY_ASV', 0) + usage_counts.get('BOTH', 0)
    
    logger.info(f"Final usable chunks:")
    logger.info(f"  - For ASR training: {asr_usable} ({asr_usable/total_chunks*100:.1f}%)")
    logger.info(f"  - For ASV training: {asv_usable} ({asv_usable/total_chunks*100:.1f}%)")
    
    # Save updated chunks data
    temp_dir = Path(config['output']['temporary_data_dir'])
    transcript_cleaned_filename = config['processing']['transcript_cleaned_filename']
    output_path = temp_dir / transcript_cleaned_filename
    
    chunks_df.to_csv(output_path, index=False)
    logger.info(f"Transcript-cleaned chunks data saved to: {output_path}")
    
    return str(output_path)