from lstm_ae import LSTM_AE


def main():
    model = LSTM_AE(input_size=10, hidden_size=20, num_epochs=100, optimizer="sgd", lr=0.001, gradient_clipping=1,
                    batch_size=100)


if __name__ == '__main__':
    main()
