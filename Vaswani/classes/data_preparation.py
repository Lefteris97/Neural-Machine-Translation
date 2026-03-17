# This Python class provides fundamental data preprocessing functionality for 
# the machine tranaslation task at the word level.

# Import all required python modules.
import unicodedata
import re
import os
import tensorflow as tf

class DataPreparation:
    
    def __init__(self,datapath,datafile,sentence_pairs,batch_size,testing_factor):
        self.datapath = datapath
        self.datafile = datafile
        self.sentence_pairs = sentence_pairs
        self.batch_size = batch_size
        self.testing_factor = testing_factor
        self.create_dataset()
        self.tokenize_dataset()
        self.partition_training_testing_datasets()
    
    # This function "ascifies" the characters pertaining to a given sentence.
    def unicode_to_ascii(self,sentence):
        # NFD normalization performs a combatibility decomposition of the 
        # input sentence. Moreover, by identifying the category of each 
        # character it is possible to exclude the "Mn" class of character which 
        # contains all combining accents.
        sentence = "".join([c for c in unicodedata.normalize("NFD", sentence)
                            if unicodedata.category(c) != "Mn"])
        return sentence
    
    # This function preprocesses each given sentence.    
    # Each input sentence is preprocessed character by character through
    # seperating out punctuations from neighbouring characters and by removing 
    # all characters other than alphabets and these particular punctuation 
    # symbols.
    def preprocess_sentence(self,sentence):
        # Clean each sentence.
        sentence = self.unicode_to_ascii(sentence)
        # Create a space between word and the punctuation following it
        sentence = re.sub(r"([!.?])",r" \1",sentence)
        # Replace everything with space except (a-z, A-Z, ".", "?", "!", ",")
        sentence = re.sub(r"[^a-zA-Z?.!]+", r" ", sentence)
        # Strip leading and following white spaces.
        sentence = re.sub(r"\s+"," ",sentence)
        sentence = sentence.lower()
        return sentence
    
    # This function prepares a dataset out of the raw data.
    # Each English sentence is converted to a sequence of words.
    # Each French sentene is converted to two sequences of words.
    # The first sequence is preceded by the "BOS " token indicating the begining
    # of the sentence. This sequence starts at position 0 which contains the BOS 
    # tokenand stops one position short of the final word in the sentence, which 
    # is the EOS token. The second sequence is followed by the " EOS" token 
    # indicating the ending of the sentence. This sequence starts at position 1
    # and goes all the way to the end of the sentence, which is the BOS token.
    def create_dataset(self):
        self.input_english_sentences,self.input_french_sentences,self.target_french_sentences = [],[],[]
        local_file = os.path.join(self.datapath,self.datafile)
        # with open(local_file, "r") as fin:
        with open(os.path.join(self.datapath, self.datafile), 'r', encoding='utf-8') as fin:
            for i,line in enumerate(fin):
                en_sent,fr_sent,_ = line.strip().split("\t")
                en_sent = [w for w in self.preprocess_sentence(en_sent).split()]
                fr_sent = self.preprocess_sentence(fr_sent)
                fr_sent_in = [w for w in ("BOS "+fr_sent).split()]
                fr_sent_out = [w for w in (fr_sent+" EOS").split()]
                self.input_english_sentences.append(en_sent)
                self.input_french_sentences.append(fr_sent_in)
                self.target_french_sentences.append(fr_sent_out)
                if i >= self.sentence_pairs:
                    break
    
    # This function tokenizes the input sequences for the English language and
    # the input and target sequences for the French language. The Tokenizer class
    # provided by the Keras framework will be employed. In particular, filters
    # are set to the empty string and lower is set to False since all the 
    # necessary preprocessing steps are already conducted by the previous 
    # functions. The aforementioned Tokenizer class creates various data 
    # structures from which it is possible to compute the vocabulary sizes for
    # both languages and acquire lookup tables for word to index and index to 
    # word transitions. Different length sequences can be handled by padding 
    # zeros at the end of each sequence when necessary.
    def tokenize_dataset(self):
        # Define the tokenizer for the English language and apply it on the set
        # of input English sentences.
        english_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="",lower=False)
        english_tokenizer.fit_on_texts(self.input_english_sentences)
        english_data = english_tokenizer.texts_to_sequences(self.input_english_sentences)
        self.input_data_english = tf.keras.preprocessing.sequence.pad_sequences(english_data,padding="post", value=0)
        # Define the tokenizer for the French language and apply it on the sets
        # of input and target french sentences.
        french_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="",lower=False)
        french_tokenizer.fit_on_texts(self.input_french_sentences)
        french_tokenizer.fit_on_texts(self.target_french_sentences)
        french_data_in = french_tokenizer.texts_to_sequences(self.input_french_sentences)
        self.input_data_french = tf.keras.preprocessing.sequence.pad_sequences(french_data_in,padding="post", value=0)
        french_data_out = french_tokenizer.texts_to_sequences(self.target_french_sentences)
        self.target_data_french = tf.keras.preprocessing.sequence.pad_sequences(french_data_out,padding="post", value=0)
        self.english_vocabulary_size = len(english_tokenizer.word_index)
        self.french_vocabulary_size = len(french_tokenizer.word_index)
        self.english_word2idx = english_tokenizer.word_index
        self.english_idx2word = {v:k for k,v in self.english_word2idx.items()}
        self.french_word2idx = french_tokenizer.word_index
        self.french_idx2word = {v:k for k,v in self.french_word2idx.items()}
        print("=======================================================")
        print("English vocabulary size: {:d}".format(self.english_vocabulary_size))
        print("French vocabulary size: {:d}".format(self.french_vocabulary_size))
        self.english_maxlen = self.input_data_english.shape[1]
        self.french_maxlen = self.target_data_french.shape[1]
        print("Maximum English sequence length: {:d}".format(self.english_maxlen))
        print("Maximum French sequence length: {:d}".format(self.french_maxlen))

    # This function creates the Tensorflow-based training and testing subsets
    # of data. The test size will be equal to the 25% of the loaded pairs of 
    # sentences.
    def partition_training_testing_datasets(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.input_data_english,
                                                     self.input_data_french,
                                                     self.target_data_french))
        dataset = dataset.shuffle(self.sentence_pairs)
        test_size = self.sentence_pairs //self.testing_factor
        self.test_dataset = dataset.take(test_size).batch(self.batch_size,drop_remainder=True)
        self.train_dataset = dataset.skip(test_size).batch(self.batch_size,drop_remainder=True)

