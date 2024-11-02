Transformer-based Bigram Language Model

This project implements a Transformer-based bigram language model from scratch using PyTorch. The model leverages multi-head self-attention and feedforward layers, along with positional embeddings, to predict the next token in a sequence given prior context. This code demonstrates the basic principles of a Transformer, utilizing self-attention blocks to capture dependencies in text data.
Model Architecture

The model follows a simplified Transformer architecture:

    Token and Positional Embeddings: Each character in the vocabulary is embedded in a high-dimensional space. Positional embeddings are added to token embeddings to retain sequence order.
    Multi-Head Attention: Implements multiple heads of self-attention in parallel, allowing the model to learn multiple representations of the input.
    Feedforward Layers: Linear layers with ReLU activation, serving as a fully connected network for each position.
    Layer Normalization and Residual Connections: Ensures stable training and allows gradients to flow more effectively across layers.

Hyperparameters

Key parameters include:

    batch_size: Number of sequences processed simultaneously.
    block_size: Maximum length of context used for predictions.
    n_embed: Embedding dimension size.
    n_head and n_layer: Number of attention heads and Transformer layers, respectively.

Training and Evaluation

The training loop runs for a specified number of iterations, saving the model’s loss at regular intervals. A helper function estimate_loss calculates the average loss over a batch for both training and validation data.
Text Generation

The generate method enables text generation based on a given context, producing new tokens one by one. A softmax function is applied to convert logits into probabilities, and new tokens are sampled from the resulting distribution.
Usage

    Load text data from input.txt for training.
    Adjust hyperparameters as desired for specific tasks.
    Run the script to train the model and generate text based on learned patterns.
