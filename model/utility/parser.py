import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run JMP-GCF.")
    parser.add_argument('--data_path', nargs='?', default='../Data/',
                        help='Input data path.')
    parser.add_argument('--epoch', type=int, default=2000,
                        help='Number of epoch.')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]',
                        help='Output sizes of every layer')
    parser.add_argument('--dataset', nargs='?', default='gowalla',
                        help='Choose a dataset from {gowalla, yelp2018, amazon-book}')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--decay', type=float, default=0.0001, help='Coefficient of L2 regularization.')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of graph convolution layers.')

    return parser.parse_args()