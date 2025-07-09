import os
import csv
import re
import string
from spacy.lang.en.stop_words import STOP_WORDS

TAGS_DIR = 'tags'

# You can add more vague/unwanted terms here if needed
VAGUE_TERMS = {'open', 'available', 'nice', 'good', 'well', 'fine', 'any', 'some', 'many',
               'looking', 'decent', 'cool', 'okay', 'basic', 'standard', 'interest', 'old', 'simple'}

def is_valid_phrase(phrase):
    return (
        len(phrase) > 2 and
        phrase not in STOP_WORDS and
        phrase not in VAGUE_TERMS and
        any(c.isalpha() for c in phrase)
    )

def clean_csv_file(filepath):
    cleaned_values = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            for item in row:
                for chunk in re.split(r'[;,\.\(\)\[\]]+', item):
                    cleaned = chunk.strip().lower()
                    if is_valid_phrase(cleaned):
                        cleaned_values.add(cleaned)
    # Overwrite the file with cleaned, deduplicated values
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for value in sorted(cleaned_values):
            writer.writerow([value])

if __name__ == '__main__':
    for filename in os.listdir(TAGS_DIR):
        if filename.endswith('.csv'):
            path = os.path.join(TAGS_DIR, filename)
            print(f'Cleaning {filename}...')
            clean_csv_file(path)
    print('All CSV files cleaned and deduplicated.') 