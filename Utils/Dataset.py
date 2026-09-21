# This method loads and processes the captions based on the type of dataset specified by the user
import re
import pickle as pkl
import unicodedata
from tqdm import tqdm
import tensorflow as tf
from Utils.Identifier import Identifier as id
from Utils.Vectorization import generate_text_vectorization
from Utils.FeatureExtractors import generate_features_filename


def load_and_preprocess_dataset(dataset_name=None, filepath=None):
    # First load the unprocessed captions from the main file
    unprocessed_captions = load_dataset_file(filepath)
    # Now process the captions in the format we use them
    processed_captions = preprocess_dataset(captions=unprocessed_captions, dataset=dataset_name)
    #  Captions have been processed, now return them in the form of a dataset
    return Dataset(dataset_name=dataset_name, image_captions=processed_captions)


# This method loads the caption file of the dataset into the main memory and then reads all the text contained in it
def load_dataset_file(filepath):
    with open(filepath, 'r', encoding="utf8") as caption_file:
        captions = caption_file.read()
        print('>>> Dataset source file loading completed...')
        return captions


# This method packs the individual captions and associates them with the images they are associated with by creating a dictionary
def preprocess_dataset(captions=None, dataset=None):
    processed_captions = dict()
    all_lines = captions.split('\n')

    if dataset is id.dataset_pascal_hin:
        # PASCAL Hindi: first line is a header; format is `image.jpg#N caption`
        for line in tqdm(all_lines[1:], desc="PROCESSING CAPTIONS", ascii=False, ncols=100):
            if not line.strip():
                continue
            image_name, caption = re.split('#[0-9] ', line[:-1])
            if image_name in processed_captions:
                processed_captions[image_name].append(caption)
            else:
                processed_captions[image_name] = [caption]

    elif dataset is id.dataset_flickr8k_hin:
        # Flickr8K Hindi: no header; format is `image_id caption` (space-separated, no .jpg in ID)
        for line in tqdm(all_lines, desc="PROCESSING CAPTIONS", ascii=False, ncols=100):
            if not line.strip():
                continue
            parts = line.strip().split(' ', 1)
            if len(parts) < 2:
                continue
            image_id, caption = parts
            image_name = image_id + '.jpg'
            if image_name in processed_captions:
                processed_captions[image_name].append(caption)
            else:
                processed_captions[image_name] = [caption]

    print('>>> Dataset preprocessing completed...')
    return processed_captions


# Pre-compiled token pattern for Hindi/Devanagari, Latin words, and numbers.
_TOKEN_PATTERN = re.compile(r"[\u0900-\u097F]+|[a-zA-Z]+|[0-9]+")


# This method deals with all the dataset related manipulations
def clean_image_captions(image_captions=None):
    word_frequencies = dict()
    freq_get = word_frequencies.get
    # Iterate over all the training images and fetch the captions associated with them
    for image_name, captions in tqdm(image_captions.items(), desc="CLEANING CAPTIONS", ascii=False,
                                     ncols=100):
        # This fetches the list of captions associated with each image, now iterate over them as well as access them individually
        for index, caption in enumerate(captions):
            clean_caption = unicodedata.normalize("NFC", caption.lower())
            word_tokens = _TOKEN_PATTERN.findall(clean_caption)
            # Update the word use frequencies inline to avoid function call overhead
            for word in word_tokens:
                word_frequencies[word] = freq_get(word, 0) + 1
            # Generate a new cleaned caption from the above received word tokens
            new_caption = ' '.join(word_tokens)
            # Now, replace the old caption text with the new cleaned caption text
            captions[index] = new_caption
    # Cleaning of captions is now complete, now return the cleaned captions along with their word frequencies
    return image_captions, word_frequencies


# This method loads the given object into the main memory
def load_preprocessed_dataset(meta_paths=None, dataset_choice=None, finetuning=False):
    if finetuning:
        filename = meta_paths.preprocessed_finetuning_dataset_abs_path(dataset_choice=dataset_choice)
    else:
        filename = meta_paths.preprocessed_dataset_abs_path(dataset_choice=dataset_choice)
    with open(filename, "rb") as filepath:
        instance = pkl.load(filepath)
        print(f">>> Successfully loaded the preprocessed dataset instance from memory...")
        return instance


# This class holds all the functions related to dataset manipulation
class Dataset:
    # The name of the dataset we are currently working with
    dataset_name = None
    # The images name of the training, validation and testing splits
    training_image_names = None
    training_image_captions = None
    training_ic = None
    validation_image_names = None
    validation_image_captions = None
    testing_image_names = None
    testing_image_captions = None
    all_image_captions = None
    flat_training_captions = None
    word_threshold = None
    word_frequencies = None
    vocabulary = None
    text_vectorization = None
    word_to_index = None
    index_to_word = None
    word_bias_vector = None
    average_caption_length = None
    max_caption_length = None
    tokenizer = None
    x = None
    y = None
    y_tokenized = None
    validation_x = None
    keras_dataset = None
    validation_keras_dataset = None

    def __init__(self, dataset_name=None, image_captions=None):
        self.dataset_name = dataset_name
        self.all_image_captions = image_captions
        print('>>> Dataset successfully initialized...')

    # This method generates a user specified testing, validation and training split
    def split(self, training_split=None, validation_split=None, testing_split=None):
        # Fetch all the image names from the dataset
        all_image_names = list(self.all_image_captions.keys())
        # Find the total number of images and split the data into the user specified ratio
        data_length = len(all_image_names)
        # Calculate the training, validation and testing split sizes
        training_data_size = int(data_length * training_split)
        validation_data_size = int(data_length * validation_split)
        # Calculate the splitting index based on the training, validation and testing lengths
        validation_data_start_index, validation_data_end_index = training_data_size, training_data_size + validation_data_size
        # Save the training, validation and testing split image names in the dataset instance
        self.training_image_names = all_image_names[:training_data_size]
        self.validation_image_names = all_image_names[validation_data_start_index:validation_data_end_index]
        self.testing_image_names = all_image_names[validation_data_end_index:]
        # Return the training_split, validation_split and testing_split size images
        print('>>> Dataset splitting completed...')

    # This method preprocesses the dataset for its use in the deep model
    def preprocess_data_splits(self, meta_paths=None, word_threshold=None):
        # Save the word threshold value for future references
        self.word_threshold = word_threshold
        # First, we fetch all  the captions associated with the training and testing images
        self.training_image_captions = self.fetch_captions(self.training_image_names)
        self.testing_image_captions = self.fetch_captions(self.testing_image_names)
        self.validation_image_captions = self.fetch_captions(self.validation_image_names)

        # Next, append the start and end tokens at the end in the training image captions
        self.training_image_captions = self.append_start_end_tokens(image_captions=self.training_image_captions)

        # Next, we iterate over all the training and testing captions and convert them into lowercase and remove special symbol
        # This way, we get a cleaned version of captions and also the document frequencies of all the words
        self.training_image_captions, self.word_frequencies = clean_image_captions(self.training_image_captions)
        self.testing_image_captions, _ = clean_image_captions(self.testing_image_captions)
        self.validation_image_captions, _ = clean_image_captions(self.validation_image_captions)

        # Now, we choose only those words for our vocabulary that are at least used for user threshold times
        # Using a set for O(1) membership lookups during token filtering
        self.vocabulary = [word for word in self.word_frequencies if self.word_frequencies[word] >= word_threshold]
        self.vocabulary_set = set(self.vocabulary)
        print(f">>> Filtered {len(self.vocabulary)} words for vocabulary from the training text corpus")
        # Filter the tokens that are not in the vocabulary from the training captions
        self.filter_banned_tokens(self.training_image_captions)
        # Convert the captions into the vectorization tokens as well
        self.vectorize_captions(self.training_image_captions)
        # Separate the (image, caption) pair so that they can be used while training the pipeline
        self.generate_x_and_y(meta_paths, self.training_image_captions, self.training_ic)

    # This method preprocesses the validation split for finetuning purposes
    def preprocess_finetuning_split(self, meta_paths=None, word_threshold=None):
        # Save the word threshold value for future references
        self.word_threshold = word_threshold
        # First, we fetch all  the captions associated with the training and testing images
        self.training_image_captions = self.fetch_captions(self.training_image_names)
        self.testing_image_captions = self.fetch_captions(self.testing_image_names)
        self.validation_image_captions = self.fetch_captions(self.validation_image_names)

        # Next, append the start and end tokens at the end in the training and validation image captions
        self.training_image_captions = self.append_start_end_tokens(image_captions=self.training_image_captions)
        self.validation_image_captions = self.append_start_end_tokens(image_captions=self.validation_image_captions)

        # Next, we iterate over all the training and testing captions and convert them into lowercase and remove special symbol
        # This way, we get a cleaned version of captions and also the document frequencies of all the words
        self.training_image_captions, self.word_frequencies = clean_image_captions(self.training_image_captions)
        self.testing_image_captions, _ = clean_image_captions(self.testing_image_captions)
        self.validation_image_captions, _ = clean_image_captions(self.validation_image_captions)

        # Now, we choose only those words for our vocabulary that are at least used for user threshold times
        self.vocabulary = [word for word in self.word_frequencies if self.word_frequencies[word] >= word_threshold]
        self.vocabulary_set = set(self.vocabulary)
        print(f">>> Filtered {len(self.vocabulary)} words for vocabulary from the training text corpus")

        # Filter the tokens that are not in the vocabulary from the training captions
        self.filter_banned_tokens(self.training_image_captions)
        self.filter_banned_tokens(self.validation_image_captions, finetuning=True)
        # Convert the captions into the BERT embeddings now
        self.vectorize_captions(self.validation_image_captions)

        # Separate the (image, caption) pair so that they can be used while training the pipeline
        self.generate_x_and_y(meta_paths, self.validation_image_captions, self.training_ic)

    # This method fetches the training captions from the overall caption dataset based on the training image names
    def fetch_captions(self, image_names=None):
        image_captions_pairs = dict()
        # Iterate over the training images and fetch their processed captions
        for image_name in tqdm(image_names, desc="LOADING IMAGE CAPTION PAIRS", ascii=False, ncols=100):
            # Fetch the processed captions of the current training image and shallow copy the list
            # (strings are immutable so shallow copy is sufficient and much faster than deepcopy)
            image_captions_pairs[image_name] = list(self.all_image_captions[image_name])
        # Return the generated training caption dict
        return image_captions_pairs

    # This method appends the <start_token> and <end_token> token into the training captions
    def append_start_end_tokens(self, image_captions=None):
        # Iterate over the training images and fetch their processed captions
        for image_name in tqdm(image_captions, desc="APPENDING START AND END TOKENS TO TRAINING CAPTIONS",
                               ascii=False, ncols=100):
            caption_pair = image_captions[image_name]
            for index, caption in enumerate(caption_pair):
                image_captions[image_name][index] = 'startsen ' + caption + ' endsen'
        return image_captions
    # This method associates the index with each of the word token in our training library
    def filter_banned_tokens(self, image_captions, finetuning=False):
        # Calculate the overall number of word tokens in all the training samples
        word_tokens_count = 0
        max_sentence_length = 0
        # Pre-built set for O(1) membership checks instead of dict.keys() lookup each iteration
        allowed_words = self.vocabulary_set
        # Remove all the token words from training captions that are below the threshold
        for image_name, captions in tqdm(image_captions.items(), desc='REMOVING BANNED TOKENS FROM TRAINING SAMPLES',
                                         ncols=100):
            for index, caption in enumerate(captions):
                caption_word_tokens = [word for word in caption.split() if word in allowed_words]
                captions[index] = ' '.join(caption_word_tokens)
                if not finetuning:
                    token_count = len(caption_word_tokens)
                    word_tokens_count += token_count
                    if token_count > max_sentence_length:
                        max_sentence_length = token_count
        if not finetuning:
            # Now update the filtered training captions
            self.training_image_captions = image_captions
            # Flat all the training captions
            self.flat_training_captions = [caption for captions in self.training_image_captions.values() for caption in captions]
            # Calculate the average caption length for the model training
            self.average_caption_length = word_tokens_count // len(self.flat_training_captions)
            self.max_caption_length = max_sentence_length
        else:
            # Now update the filtered training captions
            self.validation_image_captions = image_captions

    # This method converts the given input captions into the text vectorization form
    def vectorize_captions(self, image_captions):
        self.training_ic = dict()
        # Generate the text vectorization layer
        text_vectorization = generate_text_vectorization(vocabulary=self.vocabulary,
                                                         max_caption_length=self.max_caption_length)
        # Batch all captions into a single list for one vectorization call (avoids repeated TF kernel launches)
        image_names = []
        caption_counts = []
        all_captions = []
        for image_name, captions in image_captions.items():
            image_names.append(image_name)
            caption_counts.append(len(captions))
            all_captions.extend(captions)

        # Single batched vectorization call
        print(">>> Vectorizing all captions in a single batch...")
        all_vectorized = text_vectorization(all_captions)

        # Split the vectorized results back per image
        offset = 0
        for image_name, count in zip(image_names, caption_counts):
            self.training_ic[image_name] = all_vectorized[offset:offset + count]
            offset += count

    # This method generates the x instance for the training and validation of our pipeline
    def generate_x_and_y(self, meta_paths=None, image_caption_pairs=None, embedding_caption_pairs=None):
        x, y, y_tokenized = [], [], []
        # Directory of the image features
        directory = meta_paths.features_abs_path()
        # Fetch all the training samples one by one and append their information in x and y
        for tuple_1, tuple_2 in tqdm(zip(image_caption_pairs.items(), embedding_caption_pairs.items()),
                                                   desc="PREPARING LABELED DATASET", ncols=100):
            image_name, image_captions = tuple_1
            caption_tokenized = list(tuple_2[1])
            # Pre-compute feature filename once per image instead of once per caption
            feature_path = generate_features_filename(directory_path=directory, file_name=image_name)
            num_captions = len(image_captions)
            # Use extend for batch appending instead of repeated append calls
            x.extend([feature_path] * num_captions)
            y.extend(image_captions)
            y_tokenized.extend(caption_tokenized)

        # Replace the instance x and y with the currently created labeled dataset
        self.x = x
        self.y = y
        self.y_tokenized = y_tokenized

    # This method saves the dataset into the external directory for future reference
    def save_preprocessed_dataset(self, meta_paths=None, finetuning=False):
        if finetuning:
            filename = meta_paths.preprocessed_finetuning_dataset_abs_path(dataset_choice=self.dataset_name)
        else:
            filename = meta_paths.preprocessed_dataset_abs_path(dataset_choice=self.dataset_name)

        with open(filename, 'wb') as filepath:
            pkl.dump(self, filepath)
            print(">>> Preprocessed labeled dataset saved successfully...")

    # This method generates the keras version of dataset
    def batch_dataset(self, batch_size=None, buffer_size=1000):
        # Now, generate a new tf.dataset instance from the above list slices
        image_name_dataset = tf.data.TextLineDataset.from_tensor_slices(self.x)
        image_caption_dataset = tf.data.Dataset.from_tensor_slices(self.y)
        caption_tokenized_dataset = tf.data.Dataset.from_tensor_slices(self.y_tokenized)

        # Create a similar tf.dataset instance for validation instances as well
        # The batch size here is irrespective of hyperparameters as it is used for evaluation only
        validation_image_name_dataset = tf.data.Dataset.from_tensor_slices(self.validation_image_names).batch(batch_size=100)
        validation_image_name_dataset = validation_image_name_dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)

        # Generate a batched dataset by shuffling the dataset into a user defined buffer size and then filtering the batch from there
        image_captioning_dataset = tf.data.Dataset.zip((image_name_dataset, image_caption_dataset, caption_tokenized_dataset))
        image_captioning_dataset = image_captioning_dataset.shuffle(buffer_size=buffer_size).batch(
            batch_size=batch_size)

        image_captioning_dataset = image_captioning_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        # Dataset preparation has been complete, now return
        self.keras_dataset = image_captioning_dataset
        self.validation_keras_dataset = validation_image_name_dataset

        # self.validation_keras_dataset = validation_image_name_dataset
        print('>>> Dataset batched successfully...')


# This method generates an instance of the dataset that will be used in our framework
def initialize_dataset(dataset_name=None, filepath=None):
    dataset = load_and_preprocess_dataset(dataset_name=dataset_name, filepath=filepath)
    return dataset
