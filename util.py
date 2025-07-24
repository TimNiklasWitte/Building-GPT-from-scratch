import random

def extract_test_set(text, test_size = 0.1):
    total_length = len(text)
    test_length = int(total_length * test_size)

    start_index = random.randint(0, total_length - test_length)
    end_index = start_index + test_length

    test_text = text[start_index:end_index]
    train_text = text[:start_index] + text[end_index:]

    return train_text, test_text

def save_vocabulary(vocabulary, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for token in vocabulary:
            file.write(f"{token}\n")

def save_document(text, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)

def open_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()