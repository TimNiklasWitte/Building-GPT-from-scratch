from util import  open_text_file, save_document
from bpe import segmenter

vocab_path = 'data/vocabulary.txt'
test_corpus_path = 'data/test_corpus.txt'
save_path = 'data/segmented_test_corpus.txt'

vocabulary = open_text_file(vocab_path)
test_corpus = open_text_file()

segmented_text = segmenter(test_corpus, vocabulary)
save_document(segmented_text, save_path)
