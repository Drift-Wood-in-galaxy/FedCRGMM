import matplotlib.pyplot as plt
from Setting import *

def train_plt(loss,train_loss):
    epochs = range(ROUNDS)
    plt.plot(epochs, train_loss, 'g', label='Trainisng Loss [Local Model][Train Clients]')
    plt.plot(epochs, loss, 'b', label='Validation Loss [Global Model][All Clients]')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xlim([0, ROUNDS])
    plt.ylim([0, 15])
    plt.legend()
    plt.show()
    plt.savefig('./pic/train_loss.jpg')

def test_plt(test_loss,INDUCTIVE_ROUNDS):
    plt.plot(range(INDUCTIVE_ROUNDS), test_loss, 'r', label='Test Loss [Local Model] [Train Data]')
    plt.title('Inductive Learning Loss')
    plt.xlabel('Rounds')
    plt.ylabel('Loss')
    plt.xlim([0, INDUCTIVE_ROUNDS])
    plt.ylim([0, 5])
    plt.legend()
    plt.show()
    plt.savefig('./pic.test_loss.jpg')