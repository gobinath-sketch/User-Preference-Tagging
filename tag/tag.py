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

    height_patterns = [
        r'\b(?:above|over|at least|minimum|around|about)?\s*(\d{1})[\'′′’]\s*(\d{1,2})(?:[\"″]?)\b',
        r'\b(\d{1})\s*(?:feet|ft)\s*(\d{1,2})?\s*(?:inches|in)?\b',
        r'\b(?:above|over|at least|minimum)?\s*(\d{1}\.\d{1,2})\s*(?:feet|ft|inches|in)?\b',
        r'\b(five|six|seven|eight|nine|ten|eleven|twelve)\s*(?:feet|foot)\s*(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)?\s*(?:inches|in)?\b'
    ]
    height_matches = []
    for pattern in height_patterns:
        for match in re.finditer(pattern, input_lower):
            span = match.span()
            if any(start < span[1] and end > span[0] for start, end in global_spans):
                continue
            matched = user_input[span[0]:span[1]]
            global_spans.append(span)
            height_matches.append(matched.strip())

    if height_matches:
        matches.setdefault("height", []).extend(height_matches)

    # --- Age extraction (keep improved logic) ---
    age_patterns = [
        r'\b(?:age|aged|at the age of|at the age|is|turning|turns|will be|she is|he is|they are|i am|i\'m|she\'s|he\'s|they\'re)?\s*(\d{1,3})\s*(?:years old|yrs old|yrs|years)?\b',
        r'\b(\d{1,3})\s*(?:years old|yrs old|yrs|years)\b',
        r'\b(?:age|aged)\s*(\d{1,3})\b'
    ]
    age_matches = []
    for pattern in age_patterns:
        for match in re.finditer(pattern, input_lower):
            span = match.span()
            if any(start < span[1] and end > span[0] for start, end in global_spans):
                continue
            matched = user_input[span[0]:span[1]]
            global_spans.append(span)
            norm = normalize_age(matched)
            age_matches.append(norm)
    if age_matches:
        csv_values = set([normalize_age(v) for v in tag_dict.get('age', set())])
        for a in age_matches:
            if a in csv_values:
                matches.setdefault('age', []).append(a)
            else:
                # Fuzzy: allow +/- 1 year
                for v in csv_values:
                    try:
                        if abs(int(v)-int(a)) <= 1:
                            matches.setdefault('age', []).append(v)
                            break
                    except:
                        continue
    # --- Existing tag matching logic ---
    for tag, values in tag_dict.items():
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
        if best_score > 0.70 and best_tag:
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
