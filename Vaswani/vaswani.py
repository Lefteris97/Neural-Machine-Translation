# TRANSFORMER MODEL
# Import all required Python frameworks.
from classes.data_preparation import DataPreparation
from classes.transformer import Transformer
import tensorflow as tf
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import os
import time

# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)


# -----------------------------------------------------------------------------
#                       FUNCTIONS DEFINITION
# -----------------------------------------------------------------------------

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

def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss

# -----------------------------------------------------------------------------
# This function implements the actual training process for the neural model 
# according to the Teacher Forcing technique where the input to the decoder
# is the actual ground truth output instead of the prediction from the previous
# timestep.
# -----------------------------------------------------------------------------
@tf.function
def train_step(encoder_in, decoder_in, decoder_out):
    with tf.GradientTape() as tape:
        transformer_out = transformer((encoder_in, decoder_in), training=True)
        loss = masked_loss(decoder_out, transformer_out)
    variables = transformer.trainable_variables
    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))
    return loss
# -----------------------------------------------------------------------------
# This function is used to randomly sample a single English sentence from the
# dataset and used the model trained so far to predict the French sentence. Mind
# that the sampling process does not discriminate between training and testing
# patterns.
# -----------------------------------------------------------------------------
def predict(transformer):
    random_id = np.random.choice(len(data_preparation.input_english_sentences))
    print("Input Sentence: {}".format(" ".join(data_preparation.input_english_sentences[random_id])))
    print("Target Sentence: {}".format(" ".join(data_preparation.target_french_sentences[random_id])))

    encoder_in = tf.expand_dims(data_preparation.input_data_english[random_id], axis=0)
    decoder_in = tf.expand_dims(tf.constant([data_preparation.french_word2idx["BOS"]]), axis=0)

    pred_sent_fr = []

    while True:
        transformer_out = transformer((encoder_in, decoder_in), training=False)
        transformer_pred = tf.argmax(transformer_out[:, -1:, :], axis=-1).numpy()[0][0]
        pred_word = data_preparation.french_idx2word[transformer_pred]
        pred_sent_fr.append(pred_word)
        if pred_word == "EOS" or len(pred_sent_fr) >= data_preparation.french_maxlen:
            break
        decoder_in = tf.concat([decoder_in, tf.expand_dims([transformer_pred], 0)], axis=-1)
    print("Predicted Sentence: {}".format(" ".join(pred_sent_fr)))
# -----------------------------------------------------------------------------
# This function computes the BiLingual Evaluation Understudy (BLEU) score between
# the label and the prediction across all the records in the test set. BLUE
# scores are generally used where multiple ground truth labels exist (in this
# there exists only one), but compares up to 4-grams in both reference and
# candidate sentences.
def evaluate_bleu_score(transformer):
    bleu_scores = []
    smooth_fn = SmoothingFunction()
    for encoder_in, decoder_in, decoder_out in data_preparation.test_dataset:
        # Get the decoder final state and output for the given decoder input
        # according to the initialized decoder state.
        decoder_pred = transformer((encoder_in, decoder_in), training=False)
        # Convert the expected decoder output to a nunpy array.
        decoder_out = decoder_out.numpy()
        # Get the maximum index for each element of the decoder_pred tensor.
        # decoder_pred is initialy of shape:
        # [batch_size x french_maxlen x french_vocabulary_size] and will be
        # converted to a tensor of shape [batch_size x french_maxlen]
        decoder_pred = tf.argmax(decoder_pred, axis=-1).numpy()
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
            bleu_score = sentence_bleu([target_sent], pred_sent, smoothing_function=smooth_fn.method1)
            bleu_scores.append(bleu_score)
        return np.mean(np.array(bleu_scores))
# -----------------------------------------------------------------------------
# This function computes token-level accuracy.
def evaluate_token_accuracy(transformer):
    total_tokens = 0
    correct_tokens = 0

    for encoder_in, decoder_in, decoder_out in data_preparation.test_dataset:
        decoder_pred = transformer((encoder_in, decoder_in), training=False)

        decoder_out = decoder_out.numpy()
        decoder_pred = tf.argmax(decoder_pred, axis=-1).numpy()

        mask = (decoder_out != 0)
        correct_tokens += np.sum((decoder_out == decoder_pred) * mask)
        total_tokens += np.sum(mask)
    return correct_tokens / total_tokens
# -----------------------------------------------------------------------------
# This function is for saving checkpoints. If the save is going to fail it will retry.
# If the retries don't work it will continue its training to the next epoch
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
# This function provides the actual training process for the neural model.
def train_model(num_epochs, delta_epochs, transformer):
    # Initialize the list of evaluation scores for the current training session.
    eval_scores = []
    # Initialize the list of token_accuracies
    token_accuracies = []
    # Initialize the list of losses for the current training session.
    losses = []

    # Retrieve the starting training epoch from the last checkpoint file.
    last_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIRECTORY)
    print(f"Last checkpoint: {last_checkpoint}")

    # Restore from the latest checkpoint if it exists.
    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
        print(f"Restored from checkpoint: {checkpoint_manager.latest_checkpoint}")
        start_epoch = int(checkpoint_manager.latest_checkpoint.split("-")[-1]) + 1
    else:
        print("No checkpoint found, starting training from epoch 1")
        start_epoch = 1

    # Set the ending training epoch.
    finish_epoch = min(num_epochs, start_epoch + delta_epochs - 1)

    for epoch in range(start_epoch, finish_epoch + 1):
        start_time = time.time()  # Start the timer
        # Get the total number of batches
        total_batches = len(data_preparation.train_dataset)
        for batch_idx, (encoder_in, decoder_in, decoder_out) in enumerate(data_preparation.train_dataset):
            loss = train_step(encoder_in, decoder_in, decoder_out)
            elapsed_time = time.time() - start_time
            print(f"Epoch {epoch} | Batch {batch_idx + 1}/{total_batches} | Loss: {loss.numpy():.4f} | Time elapsed: {elapsed_time:.2f}s")

        # Evaluate at the end of each epoch
        eval_score = evaluate_bleu_score(transformer)
        token_accuracy = evaluate_token_accuracy(transformer)
        print("Eval Score (BLEU): {}".format(eval_score))
        print("Token-Level Accuracy: {}".format(token_accuracy))
        eval_scores.append(eval_score)
        token_accuracies.append(token_accuracy)
        losses.append(loss.numpy())
        # End of epoch - measure time taken
        epoch_time = time.time() - start_time
        print(f"Time taken for epoch {epoch}: {epoch_time:.2f} seconds")

        # Call the prediction function.
        predict(transformer)

        # Save the checkpoints (Doesn't work)
        # save_checkpoint(checkpoint_manager, epoch)

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
BATCH_SIZE = 64
# Set the portion of the available data to be used for testing.
TESTING_FACTOR = 10
# Set the checkpoints directory.
#CHECKPOINT_DIRECTORY = "base_checkpoints"
CHECKPOINT_DIRECTORY = "big_checkpoints"
# Set the total number of training epochs.
EPOCHS_NUMBER = 100
# Set the number of training epochs to be conducted during the current session.
DELTA_EPOCHS = 100
# Instantiate the DataPreparation class.
data_preparation = DataPreparation(DATAPATH, DATAFILE, SENTENCE_PAIRS, BATCH_SIZE, TESTING_FACTOR)
# Set the model dimensions
#MODEL_DIM = 512  # For Base model
MODEL_DIM = 1024  # For Big model

# Instantiate the transformer class.
transformer = Transformer(input_vocab_size=data_preparation.english_vocabulary_size + 1,
                          target_vocab_size=data_preparation.french_vocabulary_size + 1,
                          model_dim=MODEL_DIM)

# Note that vocabulary sizes for both languages was extended by one in order to
# take into account the fact that a PAD character was added during the call to 
# the pad_sequences() method.

# Set the optimizer to be used during the training process.
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# Set the checkpoint directory prefix.
checkpoint_prefix = os.path.join(CHECKPOINT_DIRECTORY, "ckpt")
checkpoint = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

# Create a CheckpointManager to automatically manage the checkpoints.
checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=CHECKPOINT_DIRECTORY, max_to_keep=3)

# Call the training function.
eval_scores, token_accuracies, losses = train_model(EPOCHS_NUMBER, DELTA_EPOCHS, transformer)

# Visualize training history for BLEU Score.
plt.plot(eval_scores)
plt.title('Model Accuracy in terms of the BLEU Score.')
plt.ylabel('BLEU Score')
plt.xlabel('epoch')
plt.legend(['BLEU Score'], loc='lower right')
plt.grid()
plt.show()
# Visualize Token-Level Accuracy
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