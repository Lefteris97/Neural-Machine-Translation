# Bahdanau Model

# Import all required Python frameworks.
from classes.data_preparation import DataPreparation
from classes.encoder import Encoder
from classes.decoder import Decoder
import tensorflow as tf
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import os
import time
# -----------------------------------------------------------------------------
#                       FUNCTIONS DEFINITION
# -----------------------------------------------------------------------------
# This function reports fundamental tensor shape configurations for the encoder
# and decoder objects.
def report_encoder_decoder():
    for encoder_in, decoder_in, decoder_out in data_preparation.train_dataset:
        encoder_state = encoder.init_state(data_preparation.batch_size)
        encoder_out,encoder_state = encoder(encoder_in,encoder_state)
        decoder_state = encoder_state
        decoder_pred, decoder_state = decoder(decoder_in,decoder_state, encoder_out)
        break
    print("=======================================================")
    print("Encoder Input:           :{}".format(encoder_in.shape))
    print("Encoder Output:          :{}".format(encoder_out.shape))
    print("Encoder State:           :{}".format(encoder_state.shape))
    print("=======================================================")
    print("Decoder Input:           :{}".format(decoder_in.shape))
    print("Decoder Output           :{}".format(decoder_pred.shape))
    print("Decoder State            :{}".format(decoder_state.shape))
    print("=======================================================")
# -----------------------------------------------------------------------------
# This function defines the loss function to be minimized during training.
# Given the fact that input and target sequences for both langauges have been
# padded in order to reflect the maximum sequence length for the respective
# language, it is imperative to avoid considering equality of pad words between
# the true labels and the estimated predictions. To this end, the utilized loos
# function masks the estimated predictions with the true labels, so that padded
# positions on the label are also removed from the predictions. Thus, the loss
# function is actually evaluated exclusively on the non-zero elements of both
# labels and predictions.
def loss_fn(ytrue,ypred):
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mask = tf.math.logical_not(tf.math.equal(ytrue,0))
    mask = tf.cast(mask,dtype=tf.int64)
    loss = scce(ytrue,ypred,sample_weight=mask)
    return loss, loss
# -----------------------------------------------------------------------------
# This function implements the actual training process for the neural model 
# according to the Teacher Forcing technique where the input to the decoder
# is the actual ground truth output instead of the prediction from the previous
# timestep.
# -----------------------------------------------------------------------------
@tf.function
def train_step(encoder_in,decoder_in,decoder_out,encoder_state):
    with tf.GradientTape() as tape:
        encoder_out,encoder_state = encoder(encoder_in,encoder_state)
        decoder_state = encoder_state
        decoder_pred, decoder_state = decoder(decoder_in, decoder_state, encoder_out)
        loss = loss_fn(decoder_out,decoder_pred)
    variables = (encoder.trainable_variables+decoder.trainable_variables)
    gradients = tape.gradient(loss,variables)
    optimizer.apply_gradients(zip(gradients,variables))
    return loss
# -----------------------------------------------------------------------------
# This function is used to randomly sample a single English sentence from the
# dataset and used the model trained so far to predict the French sentence. Mind
# that the sampling process does not discriminate between training and testing
# patterns.
# -----------------------------------------------------------------------------
def predict(encoder,decoder):
    random_id = np.random.choice(len(data_preparation.input_english_sentences))
    print("Input Sentence: {}".format(" ".join(data_preparation.input_english_sentences[random_id])))
    print("Target Sentence: {}".format(" ".join(data_preparation.target_french_sentences[random_id])))
    encoder_in = tf.expand_dims(data_preparation.input_data_english[random_id],axis=0)
    encoder_state = encoder.init_state(1)
    encoder_out,encoder_state = encoder(encoder_in,encoder_state)
    decoder_state = encoder_state
    decoder_in = tf.expand_dims(tf.constant([data_preparation.french_word2idx["BOS"]]),axis=0)
    pred_sent_fr = []
    while True:
        decoder_pred, decoder_state = decoder(decoder_in, decoder_state, encoder_out)
        decoder_pred = tf.argmax(decoder_pred,axis=-1)
        pred_word = data_preparation.french_idx2word[decoder_pred.numpy()[0][0]]
        pred_sent_fr.append(pred_word)
        if pred_word == "EOS" or len(pred_sent_fr)>=data_preparation.french_maxlen:
            break
        decoder_in = decoder_pred
    print("Predicted Sentence: {}".format(" ".join(pred_sent_fr)))
    
# -----------------------------------------------------------------------------
# This function computes the BiLingual Evaluation Understudy (BLEU) score between  
# the label and the prediction across all the records in the test set. BLUE 
# scores are generally used where multiple ground truth labels exist (in this 
# there exists only one), but compares up to 4-grams in both reference and 
# candidate sentences.
def evaluate_bleu_score(encoder,decoder):
    bleu_scores = []
    smooth_fn = SmoothingFunction()
    for encoder_in,decoder_in,decoder_out in data_preparation.test_dataset:
        # Initialize the encoder state accroding to the selected batchsize.
        encoder_state = encoder.init_state(data_preparation.batch_size)
        # Get the encoder final state and output for the given the encoder input   
        # according to the initialized encoder state.
        encoder_out,encoder_state = encoder(encoder_in,encoder_state)
        # Set the initial state of the decoder to be equal to the final state
        # of the encoder.
        decoder_state = encoder_state
        # Get the decoder final state and output for the given decoder input 
        # according to the initialized decoder state.
        decoder_pred, decoder_state = decoder(decoder_in, decoder_state, encoder_out)
        # Connvert the expected decoder output to a nunpy array.
        decoder_out = decoder_out.numpy()
        # Get the maximum index for each element of the decoder_pred tensor.
        # decoder_pred is initialy of shape:
        # [batch_size x french_maxlen x french_vocabulary_size] and will be 
        # converted to a tensor of shape [batch_size x french_maxlen]
        decoder_pred = tf.argmax(decoder_pred,axis=-1).numpy()
        # Loop through the various patterns in the current decoder_out batch.
        for i in range(decoder_out.shape[0]):
            # Compose the correct sequence of target french words as a list of 
            # strings.
            target_sent = [data_preparation.french_idx2word[j] 
                        for j in decoder_out[i].tolist() if j > 0]
            # Compose the estimated sequence of target french words as a list of
            # strings.
            pred_sent = [data_preparation.french_idx2word[j] for j in 
                         decoder_pred[i].tolist() if j > 0]
            # Remove trailing EOS tokens from both target and predicted 
            # sentences. Mind that for predicted sentences during the earlier  
            # training stages an EOS token may have not been generated.
            target_sent = target_sent[0:-1]
            pred_sent = pred_sent[0:-1]
            bleu_score = sentence_bleu([target_sent],pred_sent, smoothing_function=smooth_fn.method1)
            bleu_scores.append(bleu_score)
        return np.mean(np.array(bleu_scores))
# -----------------------------------------------------------------------------
# This function cleans the older checkpoint files from the corresponding 
# directory.
def clean_checkpoints():
    last_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIRECTORY)
    # Retrieve the prefix of the latest checkpoint file.
    prefix = last_checkpoint.split("/")[-1]
    # Get the list of files contained within the checkpoint directory.
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIRECTORY)]
    # Remove all file that do not contain the prefix of the latest checkpoint
    # file.
    for file in checkpoint_files:
        status = file.find(prefix)
        if status == -1:
            if file != 'checkpoint':
                remove_file = os.path.join(CHECKPOINT_DIRECTORY,file)
                os.remove(remove_file)
# -----------------------------------------------------------------------------
# This function computes token-level accuracy.
def evaluate_token_accuracy(encoder, decoder):
    total_tokens = 0
    correct_tokens = 0
    for encoder_in, decoder_in, decoder_out in data_preparation.test_dataset:
        encoder_state = encoder.init_state(data_preparation.batch_size)
        encoder_out, encoder_state = encoder(encoder_in, encoder_state)
        decoder_state = encoder_state
        decoder_pred, decoder_state = decoder(decoder_in, decoder_state, encoder_out)
        decoder_out = decoder_out.numpy()
        decoder_pred = tf.argmax(decoder_pred, axis=-1).numpy()
        mask = (decoder_out != 0)
        correct_tokens += np.sum((decoder_out == decoder_pred) * mask)
        total_tokens += np.sum(mask)
    return correct_tokens / total_tokens
# -----------------------------------------------------------------------------
# This function is for saving checkpoints. If the save is going to fail it will retry.
# If the retries dont work it will continue its training to the next epoch
def save_checkpoint(checkpoint_manager, epoch, retry_count=5, delay=5):
    for i in range(retry_count):
        try:
            checkpoint_manager.save()
            print(f"Checkpoint saved at epoch {epoch}")
            break
        except tf.errors.FailedPreconditionError as e:
            print(f"Failed to save checkpoint: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    else:
        print("Failed to save checkpoint after multiple attempts.")
# -----------------------------------------------------------------------------
import re
# This function provides the actual training process for the neural model.
def train_model(num_epochs,delta_epochs,encoder,decoder):
    # Initialize the list of evaluation scores for the current training session.
    eval_scores = []
    # Initialize the list of token_accuracies
    token_accuracies = []
    # Initialize the list of losses for the current training session.
    losses = []

    # Retrieve the starting training epoch from the last checkpoint file.
    last_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIRECTORY)
    print(f"Last checkpoint: {last_checkpoint}")  # For Debugging

    # Restore from the latest checkpoint if it exists.
    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        print(f"Restored from checkpoint: {checkpoint_manager.latest_checkpoint}")
        start_epoch = int(checkpoint_manager.latest_checkpoint.split("-")[-1]) + 1
    else:
        print("No checkpoint found, starting training from epoch 1")
        start_epoch = 1

    # Set the ending training epoch.
    finish_epoch = min(num_epochs, start_epoch + delta_epochs - 1)

    for epoch in range(start_epoch, finish_epoch + 1):
        start_time = time.time()  # Start the timer
        encoder_state = encoder.init_state(data_preparation.batch_size)
        # Get the total number of batches
        total_batches = len(data_preparation.train_dataset)
        for batch_idx, (encoder_in, decoder_in, decoder_out) in enumerate(data_preparation.train_dataset):
            loss = train_step(encoder_in, decoder_in, decoder_out, encoder_state)
            elapsed_time = time.time() - start_time
            print(f"Epoch {epoch} | Batch {batch_idx + 1}/{total_batches} | Loss: {loss[0].numpy():.4f} | Time elapsed: {elapsed_time:.2f}s")

        # Evaluate at the end of each epoch
        eval_score = evaluate_bleu_score(encoder, decoder)
        token_accuracy = evaluate_token_accuracy(encoder, decoder)
        print("Eval Score (BLEU): {}".format(eval_score))
        print("Token-Level Accuracy: {}".format(token_accuracy))
        eval_scores.append(eval_score)
        token_accuracies.append(token_accuracy)
        losses.append(loss[0].numpy())
        # End of epoch - measure time taken
        epoch_time = time.time() - start_time
        print(f"Time taken for epoch {epoch}: {epoch_time:.2f} seconds")

        # Call the prediction function.
        predict(encoder, decoder)

        # Save the checkpoint
        save_checkpoint(checkpoint_manager, epoch)

    return eval_scores, token_accuracies, losses
# -----------------------------------------------------------------------------
#                       MAIN PROGRAM
# -----------------------------------------------------------------------------
# Set the path to the data folder.
DATAPATH = "../datasets"
# Set the name of the data file.
DATAFILE = "fra.txt"
# Set the number of English-French sentence pairs to be retrieved.
SENTENCE_PAIRS = 15000
# Set the bath size for training.
# BATCH_SIZE = 64
BATCH_SIZE = 150
# Set the portion of the available data to be used for testing.
TESTING_FACTOR = 10
# Set the bahdanau_checkpoints directory.
CHECKPOINT_DIRECTORY = "bahdanau_checkpoints"
# Set the total number of training epochs.
EPOCHS_NUMBER = 250
# Set the number of training epochs to be conducted during the current session.
DELTA_EPOCHS = 100
# Instantiate the DataPreparation class.
data_preparation = DataPreparation(DATAPATH,DATAFILE,SENTENCE_PAIRS,BATCH_SIZE,
                                   TESTING_FACTOR)

# Set the embedding dimension for the encoder and the decoder.
EMBEDDING_DIM = 256
# Set the encoder and decoder RNNs hidden dimensions.
ENCODER_DIM, DECODER_DIM = 1024, 1024
# Instantiate the encoder class.
# encoder = Encoder(data_preparation.english_vocabulary_size+1,EMBEDDING_DIM,
#                   data_preparation.english_maxlen,ENCODER_DIM)
encoder = Encoder(data_preparation.english_vocabulary_size+1, data_preparation.english_maxlen, EMBEDDING_DIM, ENCODER_DIM)
# Instantiate the decoder class.
# decoder = Decoder(data_preparation.french_vocabulary_size+1,EMBEDDING_DIM,
#                   data_preparation.french_maxlen,DECODER_DIM)
decoder = Decoder(data_preparation.french_vocabulary_size+1, data_preparation.french_maxlen, EMBEDDING_DIM, DECODER_DIM)
# Note that vocabulary sizes for both languages was extended by one in order to 
# take into account the fact that a PAD character was added during the call to 
# the pad_sequences() method.
report_encoder_decoder()
# Set the optimizer to be used during the training process.
optimizer = tf.keras.optimizers.Adam()
# Set the checkpoint directory prefix.
checkpoint_prefix = os.path.join(CHECKPOINT_DIRECTORY,"ckpt")
# Setup a checkpoint so that the neural model can be saved every 10 epochs.
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

# Create a CheckpointManager to automatically manage the bahdanau_checkpoints.
checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=CHECKPOINT_DIRECTORY, max_to_keep=2)

# Call the training function.
eval_scores, token_accuracies, losses = train_model(EPOCHS_NUMBER,DELTA_EPOCHS,encoder,decoder)


# Visualize training history for BLEU Score.
plt.plot(eval_scores)
plt.title('Model Accuracy in terms of the BLEU Score.')
plt.ylabel('BLEU Score')
plt.xlabel('epoch')
plt.legend(['BLEU Score'], loc='lower right')
plt.grid()
plt.show()
#Visualize Token-Level Accuracy
plt.plot(token_accuracies)
plt.title('Token-Level Accuracy of Model')
plt.ylabel('Token-Level Accuracy Score')
plt.xlabel('epoch')
plt.legend(['Token-Level Accuracy Score'], loc='lower right')
plt.grid()
plt.show()
# Visualize training history for Loss.
plt.plot(losses)
plt.title('Model Accuracy in terms of the Loss.')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['LOSS'], loc='upper right')
plt.grid()
plt.show()