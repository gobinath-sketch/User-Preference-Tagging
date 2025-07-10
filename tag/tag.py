import os
import csv
import re
import pickle
import torch
import hashlib
import spacy
import string
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from spacy.lang.en.stop_words import STOP_WORDS
from difflib import SequenceMatcher

# === Constants ===
TAGS_DIR = 'tags'
LOG_FILE = 'learned_log.txt'
CACHE_FILE = 'embedding_cache.pkl'
SEMANTIC_LOG = 'semantic_log.txt'
TAG_EMBEDDINGS_FILE = 'tag_embeddings.pkl'
TAG_EMBEDDINGS_HASH = 'tag_embeddings.hash'

EMBEDDING_MODEL = SentenceTransformer('intfloat/e5-base-v2')
nlp = spacy.load("en_core_web_sm")


# === Load or init embedding cache ===
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'rb') as f:
        embedding_cache = pickle.load(f)
else:
    embedding_cache = {}
def get_cached_embedding(text):
    key = text.strip().lower()
    if key not in embedding_cache:
        embedding_cache[key] = EMBEDDING_MODEL.encode(key, convert_to_tensor=True)
    return embedding_cache[key]

def is_complete_noun_phrase(phrase):
    doc = nlp(phrase)
    # Accept only if the phrase is a full noun chunk and not a fragment
    if len(doc) < 2:
        return False
    for chunk in doc.noun_chunks:
        if chunk.text.strip().lower() == phrase.strip().lower():
            # Avoid fragments like 'a strong sense', prefer 'strong sense of identity'
            if chunk.root.head.pos_ in {"VERB", "ADJ", "NOUN"} or chunk.root.dep_ == "ROOT":
                return True
    return False

def is_valid_phrase(phrase, vague_terms, min_words=1):
    # Use NLP to check for completeness and context
    if not is_complete_noun_phrase(phrase):
        return False
    if phrase in vague_terms or phrase in STOP_WORDS:
        return False
    if len(phrase.split()) < min_words:
        return False
    if len(phrase) <= 2:
        return False
    return True

# List of generic single-word verbs/values to filter from hobbies_interests and profession
GENERIC_PROFESSION = {"currently", "work", "working", "teaching", "school", "dedication", "enterprising"}
GENERIC_HOBBIES = {"searching"}

def is_valid_for_tag(tag, phrase, tag_dict):
    # Only allow values present in the CSV for that tag
    # For learning, allow if it's a noun phrase and not too generic
    if tag == "language" and phrase == "modern":
        return False
    if tag == "profession" and (phrase == "working" or phrase in GENERIC_PROFESSION):
        return False
    if tag == "hobbies_interests" and phrase in GENERIC_HOBBIES:
        return False
    if phrase in tag_dict.get(tag, set()):
        return True
    # Allow learning new values for certain tags
    if tag in {"profession", "education", "location", "relocation", "spiritual_religious", "values_personality_traits", "hobbies_interests"}:
        # Accept multi-word noun phrases or contextually relevant phrases
        if len(phrase.split()) > 1:
            return True
    return False

def is_near_duplicate(phrase, existing_values):
    for val in existing_values:
        ratio = SequenceMatcher(None, phrase, val).ratio()
        if ratio > 0.85:
            return True
    return False

def compute_tag_files_hash(tags_dir):
    sha = hashlib.sha256()
    for filename in sorted(os.listdir(tags_dir)):
        if filename.endswith('.csv'):
            path = os.path.join(tags_dir, filename)
            with open(path, 'rb') as f:
                sha.update(f.read())
    return sha.hexdigest()

def load_tag_embeddings(tag_dict):
    current_hash = compute_tag_files_hash(TAGS_DIR)
    if os.path.exists(TAG_EMBEDDINGS_FILE) and os.path.exists(TAG_EMBEDDINGS_HASH):
        with open(TAG_EMBEDDINGS_HASH, 'r') as f:
            cached_hash = f.read().strip()
        if cached_hash == current_hash:
            with open(TAG_EMBEDDINGS_FILE, 'rb') as f:
                return pickle.load(f)
    print("🔁 Rebuilding tag embedding cache...")
    tag_embeddings = {
        tag: EMBEDDING_MODEL.encode(list(values), convert_to_tensor=True)
        for tag, values in tag_dict.items() if values
    }
    with open(TAG_EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(tag_embeddings, f)
    with open(TAG_EMBEDDINGS_HASH, 'w') as f:
        f.write(current_hash)
    return tag_embeddings

def load_tags(tags_dir):
    tag_dict = {}
    for filename in os.listdir(tags_dir):
        if filename.endswith('.csv'):
            tag = os.path.splitext(filename)[0]
            values = set()
            with open(os.path.join(tags_dir, filename), 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    for item in row:
                        for chunk in re.split(r'[;,.\(\)\[\]]+', item):
                            cleaned = chunk.strip().lower()
                            if cleaned and cleaned not in STOP_WORDS and len(cleaned) > 2:
                                values.add(cleaned)
            tag_dict[tag] = values
    return tag_dict

def save_to_csv(tag, phrase):
    path = os.path.join(TAGS_DIR, f"{tag}.csv")
    with open(path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([phrase.lower()])

def log_learning(phrase, tag, is_duplicate=False):
    if is_duplicate:
        return
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    category = tag.replace("_", " ").title()
    line = f'{timestamp} ✅ Learned: "{phrase}" → {category}'
    with open(LOG_FILE, 'a', encoding='utf-8') as log:
        log.write(line + '\n')


def log_semantic_match(phrase, tag, score):
    with open(SEMANTIC_LOG, 'a', encoding='utf-8') as log:
        log.write(f'{datetime.now()} | "{phrase}" matched "{tag}" with score {round(score, 3)}\n')

def normalize_age(age_str):
    # Extract just the number (e.g., '30 years old' -> '30')
    match = re.search(r'(\d{1,3})', age_str)
    if match:
        return match.group(1)
    return age_str.strip().lower()

def normalize_height(height_str):
    # Convert word-based heights to numeric (e.g., 'five feet eleven inches' -> '5 ft 11 in')
    word2num = {'zero':0, 'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9, 'ten':10, 'eleven':11, 'twelve':12}
    s = height_str.lower()
    # Replace word numbers with digits
    for word, num in word2num.items():
        s = re.sub(r'\b'+word+r'\b', str(num), s)
    # Standardize format
    s = s.replace('feet', 'ft').replace('foot', 'ft').replace('inches', 'in').replace('inch', 'in')
    s = re.sub(r'\s+', ' ', s)
    # Try to extract patterns like '5 ft 11 in'
    match = re.search(r'(\d+)\s*ft\s*(\d+)?\s*in', s)
    if match:
        ft = match.group(1)
        inch = match.group(2) if match.group(2) else '0'
        return f"{ft} ft {inch} in"
    # Try to extract patterns like 5'11
    match = re.search(r'(\d+)[\'′′’](\d+)', s)
    if match:
        return f"{match.group(1)} ft {match.group(2)} in"
    return s.strip()

def extract_matches(user_input, tag_dict):
    matches = {}
    input_lower = user_input.lower()
    global_spans = []

    # --- Height extraction ---
    # Expanded patterns for all real-world, ambiguous, rare, informal, and range-based height expressions
    height_patterns = [
        # Standard feet & inches
        r"\b([4-7])['′]\s?(\d{1,2})[\"″]?\b",  # 5'11", 6'2"
        r"\b([4-7])\s?ft\.?\s?(\d{1,2})\s?(in|inches)?\b",  # 5 ft 10 in
        r"\b([4-7])\s?feet\s?(\d{1,2})\s?(inches|inch)?\b",  # 5 feet 8 inches
        # Inches only
        r"\b([5-8][0-9])\s?(in|inches)\b",  # 72 inches
        # Centimeters
        r"\b(1[0-9]{2}|2[0-4][0-9])\s?(cm|centimeters?)\b",  # 180 cm
        # Meters
        r"\b([1-2](?:[.,]\d{1,2}))\s?(m|meters?)\b",  # 1.75 m, 1,80 m
        # Written out: six feet two, five foot eleven
        r"\b(four|five|six|seven)\s?(feet|foot|ft)\s?(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)?\b",
        # Rare/informal: five and a half feet, five eleven
        r"\b(four|five|six|seven)\s?(and a half|and half)\s?(feet|foot|ft)\b",  # five and a half feet
        r"\b(four|five|six|seven)\s?eleven\b",  # five eleven
        r"\b(\d)\s?foot[s]?\s?(\d{1,2})?\b",  # 5 foot 11, 5 foot
        # Slang: five eleven, six two
        r"\b(four|five|six|seven)\s?(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",  # five eleven
        # Creative spacing/typos
        r"\b(\d)\s?['′]\s?(\d{1,2})\b",  # 5 ' 11
        # Ranges: 5'10" to 6'0", between 5'8" and 6'0"
        r"\b([4-7]['′]\d{1,2})\s?(to|\-|and|–)\s?([4-7]['′]\d{1,2})\b",  # 5'10" to 6'0"
        r"\bbetween\s([4-7]['′]\d{1,2})\s?(and|to|-)\s?([4-7]['′]\d{1,2})\b",  # between 5'8" and 6'0"
    ]
    ambiguous_height_pattern = r"\b(tall|short|average height|medium height|medium tall|medium short|about \d{1,2}|around \d{1,2}|approximately \d{1,2})\b"
    height_matches = []
    explicit_spans = []
    for pattern in height_patterns:
        for match in re.finditer(pattern, input_lower):
            span = match.span()
            if any(start < span[1] and end > span[0] for start, end in global_spans):
                continue
            matched = user_input[span[0]:span[1]]
            global_spans.append(span)
            height_matches.append(matched.strip())
            explicit_spans.append(span)
    # Only add ambiguous/approximate if no explicit match overlaps or is adjacent
    for match in re.finditer(ambiguous_height_pattern, input_lower):
        span = match.span()
        # Check for overlap or adjacency with explicit spans
        if any(abs(span[0] - e[1]) <= 1 or abs(span[1] - e[0]) <= 1 or (span[0] < e[1] and span[1] > e[0]) for e in explicit_spans):
            continue
        matched = user_input[span[0]:span[1]]
        global_spans.append(span)
        height_matches.append(matched.strip())
    if height_matches:
        matches.setdefault("height", []).extend(height_matches)

    # --- Age extraction (exclude gender/role words) ---
    # Comprehensive patterns for numeric, ranges, written, and open/flexible age
    age_patterns = [
        # Numeric age: 25 years old, 32 y/o, age 40, I'm 29
        r"\b([1-9][0-9])\s?(years? old|y/o|yrs? old|yo)\b",  # 25 years old, 32 y/o
        r"\bage\s?([1-9][0-9])\b",  # age 40
        r"\bI'?m\s?([1-9][0-9])\b",  # I'm 29
        # Age ranges: 20-25, between 30 and 35, in my 40s
        r"\b([1-9][0-9])\s?[-–]\s?([1-9][0-9])\b",  # 20-25
        r"\bbetween\s([1-9][0-9])\s?(and|to)\s?([1-9][0-9])\b",  # between 30 and 35
        r"\bin my (\d{2})s\b",  # in my 40s
        # Written out: twenty five years old
        r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)([-\s]?(one|two|three|four|five|six|seven|eight|nine))?\s?(years? old|y/o|yrs? old|yo)?\b",
        # Open/flexible age phrases
        r"\b(whatever the age|any age|no age bar|no age limit|open age|all ages|no bar for age|no age restriction|no age preference)\b"
    ]
    age_matches = []
    for pattern in age_patterns:
        for match in re.finditer(pattern, input_lower):
            span = match.span()
            if any(start < span[1] and end > span[0] for start, end in global_spans):
                continue
            matched = user_input[span[0]:span[1]]
            global_spans.append(span)
            age_matches.append(matched.strip())
    if age_matches:
        matches.setdefault('age', []).extend(age_matches)

    # --- Context-aware extraction for smoking and alcohol ---
    # Improved negation/avoidance patterns to handle 'avoid alcohol and smoking', etc.
    non_smoking_patterns = [
        r"avoid[\w\s,]*smoking", r"no[\w\s,]*smoking", r"does not smoke", r"never smokes?", r"non[- ]smoking", r"doesn't smoke", r"don't smoke", r"not smoke", r"without smoking"
    ]
    non_drinking_patterns = [
        r"avoid[\w\s,]*alcohol", r"no[\w\s,]*alcohol", r"does not drink", r"never drinks?", r"non[- ]drinking", r"doesn't drink", r"don't drink", r"not drink", r"without alcohol"
    ]
    input_lower = user_input.lower()
    found_non_smoking = any(re.search(p, input_lower) for p in non_smoking_patterns)
    found_non_drinking = any(re.search(p, input_lower) for p in non_drinking_patterns)

    # Remove any existing matches for 'smoking_drinking habits' before tag dict matching
    if 'smoking_drinking habits' in matches:
        del matches['smoking_drinking habits']

    # Add non-smoking/non-drinking if negation/avoidance is found
    non_habits = []
    if found_non_smoking:
        non_habits.append('non-smoking')
    if found_non_drinking:
        non_habits.append('non-drinking')
    if non_habits:
        matches['smoking_drinking habits'] = non_habits

    # Only if not negated, allow normal tag dict matching for 'smoking_drinking habits'
    for tag, values in tag_dict.items():
        if tag == "height":
            continue  # Only match height via regex, not tag dict
        if tag == "smoking_drinking habits" and (found_non_smoking or found_non_drinking):
            continue  # Skip normal matching if negation/avoidance found
        found_phrases = []
        for value in sorted(values, key=lambda x: -len(x)):
            pattern = r'(?<!\w)' + re.escape(value) + r'(?!\w)'
            for match in re.finditer(pattern, input_lower):
                start, end = match.start(), match.end()
                if any(start < e and end > s for s, e in global_spans):
                    continue
                matched_phrase = user_input[start:end].strip(string.punctuation + " ")
                global_spans.append((start, end))
                if is_valid_for_tag(tag, matched_phrase.lower(), tag_dict):
                    if matched_phrase not in found_phrases:
                        found_phrases.append(matched_phrase)
        if found_phrases:
            matches[tag] = found_phrases

    return matches, global_spans

def semantic_learn(user_input, tag_dict, used_spans, all_existing_embeddings):
    vague_terms = {'open', 'available', 'nice', 'good', 'well', 'fine', 'any', 'some', 'many',
                   'looking', 'decent', 'cool', 'okay', 'basic', 'standard'}
    new_phrases = []
    doc = nlp(user_input)
    # Add NER for locations and orgs
    for ent in doc.ents:
        if ent.label_ in {"GPE", "LOC", "ORG"}:
            phrase = ent.text.strip().lower()
            if phrase not in STOP_WORDS and len(phrase) > 2:
                new_phrases.append((phrase, ent.start_char, ent.end_char, ent.label_))
    # Noun chunks
    for chunk in doc.noun_chunks:
        start, end = chunk.start_char, chunk.end_char
        if any(start < e and end > s for s, e in used_spans):
            continue
        phrase = chunk.text.strip(string.punctuation + " ").lower()
        if is_valid_phrase(phrase, vague_terms, min_words=2):
            new_phrases.append((phrase, start, end, None))
    if not new_phrases:
        return {}
    learned = {}
    for phrase, _, _, ent_label in new_phrases:
        phrase_emb = get_cached_embedding(phrase)
        best_tag = None
        best_score = -1
        for tag, embeddings in all_existing_embeddings.items():
            if not embeddings.shape[0]:
                continue
            sim_scores = util.cos_sim(phrase_emb, embeddings)[0]
            max_score = float(torch.max(sim_scores))
            if max_score > best_score:
                best_score = max_score
                best_tag = tag
        # If it's a location/entity, force tag
        if ent_label in {"GPE", "LOC"}:
            best_tag = "location"
        elif ent_label == "ORG":
            best_tag = "profession"  # or a new tag if you want
        if best_score > 0.75 and best_tag:
            existing_values = tag_dict[best_tag]
            if not is_near_duplicate(phrase, existing_values) and is_valid_for_tag(best_tag, phrase, tag_dict):
                learned.setdefault(best_tag, []).append(phrase)
                tag_dict[best_tag].add(phrase)
                save_to_csv(best_tag, phrase)
                log_learning(phrase, best_tag, is_duplicate=False)
                log_semantic_match(phrase, best_tag, best_score)
            else:
                log_learning(phrase, best_tag, is_duplicate=True)
    return learned

def main():
    tag_dict = load_tags(TAGS_DIR)
    all_existing_embeddings = load_tag_embeddings(tag_dict)

    print("⚡ Semantic Tag Matcher & Real-Time Learner")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_input = input('📝 Enter your preferences or description: ')
        if user_input.strip().lower() == 'exit':
            with open(CACHE_FILE, 'wb') as f:
                pickle.dump(embedding_cache, f)
            with open(TAG_EMBEDDINGS_FILE, 'wb') as f:
                pickle.dump(all_existing_embeddings, f)
            print("👋 Exiting. All learned data and cache have been saved.")
            break

        matches, spans = extract_matches(user_input, tag_dict)
        learned = semantic_learn(user_input, tag_dict, spans, all_existing_embeddings)

        if learned:
            for tag, phrases in learned.items():
                for phrase in phrases:
                    if tag in all_existing_embeddings:
                        emb = get_cached_embedding(phrase)
                        all_existing_embeddings[tag] = torch.cat(
                            (all_existing_embeddings[tag], emb.unsqueeze(0)), dim=0
                        )

        for tag, phrases in learned.items():
            matches.setdefault(tag, []).extend(phrases)

        if matches:
            print('\n✅ Matched tags:')
            for tag, values in matches.items():
                print(f'  - {tag}: {", ".join(values)}')

        if learned:
            print('\n🧠 Learned new matches:')
            for tag, phrases in learned.items():
                print(f'  - {tag}: {", ".join(phrases)}')

        if not matches and not learned:
            print('❌ No matches found.')

        print("\n────────────────────────────────────\n")

if __name__ == '__main__':
    main()
