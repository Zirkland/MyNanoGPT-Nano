# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import signal
import sys
from DataloaderNano import get_batch, hyperparameters
from transformers import GPT2Tokenizer, GPT2Model
import torch.optim.lr_scheduler as lr_scheduler
import logging
import os
from MyTransformer import MyModel  # Import MyTransformer
import logging
from datetime import datetime

def configure_logging():
    # Configure logging to file (append mode)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='training.log', filemode='a')

    # Add a StreamHandler to output logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(console_handler)

    # Log the start time
    logging.info(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Call the function to configure logging
configure_logging()

# Check if CUDA is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cuda_available = torch.cuda.is_available()
logging.info(f"CUDA available: {cuda_available}")

if cuda_available:
    logging.info(f"Number of CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        logging.info(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")

learning_rate = hyperparameters['learning_rate']
max_iters = hyperparameters['max_iters']
eval_interval = hyperparameters['eval_interval']
log_interval = hyperparameters['log_interval']
save_interval = hyperparameters.get('save_interval', 1000)  # Add save interval

# Initialize MyTransformer with GPT-2's pre-trained parameters
model = MyModel()
model.to(device)

# Load GPT-2 tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Load GPT-2 pre-trained parameters
def load_gpt2_pretrained(model, model_name='gpt2'):
    gpt2_model = GPT2Model.from_pretrained(model_name)
    model_dict = model.state_dict()
    pretrained_dict = gpt2_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

load_gpt2_pretrained(model)

criterion = nn.CrossEntropyLoss()

# Initialize the optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Lists to store loss values for visualization
train_losses = []
val_losses = []

def save_model():
    torch.save(model.state_dict(), 'trained_model.pth')
    logging.info("Model saved to 'trained_model.pth'")

def emergency_stop(signal, frame):
    logging.info("Emergency stop triggered. Saving model...")
    save_model()
    sys.exit(0)

signal.signal(signal.SIGINT, emergency_stop)

def load_model_if_exists(model, filepath='trained_model.pth'):
    if os.path.exists(filepath):
        model.load_state_dict(torch.load(filepath), strict=False)
        logging.info(f"Model weights loaded from '{filepath}'")
    else:
        logging.info(f"No saved model weights found at '{filepath}'")

def train(model, hyperparameters):
    load_model_if_exists(model)  # Load model weights if they exist
    model.train()
    try:
        for iter in range(hyperparameters['max_iters']):
            x, y = get_batch('train')
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)  # Directly get logits from the model
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update the learning rate

            if iter % log_interval == 0:
                logging.info(f"Iteration {iter}, Loss: {loss.item()}")
                train_losses.append(loss.item())

            if iter % eval_interval == 0:
                val_loss = evaluate(model, hyperparameters)
                val_losses.append(val_loss)

            if iter % save_interval == 0:
                save_model()

    except KeyboardInterrupt:
        logging.info("Training interrupted. Saving model...")
        save_model()
        sys.exit(0)

    save_model()
    plot_losses()
    # Log the end time
    logging.info(f"Training ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def evaluate(model, hyperparameters):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        x, y = get_batch('val')
        x, y = x.to(device), y.to(device)
        logits = model(x)  # Directly get logits from the model
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()
    avg_loss = total_loss / len(x)
    logging.info(f"Validation Loss: {avg_loss}")
    model.train()
    return avg_loss

def plot_losses():
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_progress.png')
    plt.show()

if __name__ == "__main__":
    train(model, hyperparameters)
