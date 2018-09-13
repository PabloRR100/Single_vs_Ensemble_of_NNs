
import argparse

dss = ['SVHN', 'Fruits', 'Fashion MNIST', 'Facial Poses']
msd = 'Posible values: ' + ', '.join(dss)
ens = ['Big', 'Huge']
mse = 'Possible values: ' + ', '.join(ens)

# Function to parse the booleans
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        

parser = argparse.ArgumentParser(description='Evaluate importance of depth, width and ensembling')

# General parameters

parser.add_argument('-s', '--save', 
                    type=str2bool, default=False,
                    help='Activate save the outputs of the script',)

parser.add_argument('-n', '--name', 
                    help='Name of the file to save the outputs')

parser.add_argument('-t', '--testing', 
                    type=str2bool, default=False,
                    help='Activate test mode to train only few interations')

parser.add_argument('-c', '--comments', 
                    type=str2bool, default=False,
                    help='Boolean: show comments while running')

parser.add_argument('-D', '--draws', 
                    type=str2bool, default=False,
                    help='Boolean: draw figures or jsut save them')

# Model parameters
parser.add_argument('-E', '--ensembleSize',
                    type=str, default='Big', help=mse)



# Training parameters

parser.add_argument('-d', '--dataset', 
                    type=str, default='CIFAR10', help=msd)

parser.add_argument('-e', '--epochs', 
                    type=int, default=None, 
                    help='Int: Epochs for training')

parser.add_argument('-i', '--iterations', 
                    type=int, default=64000, 
                    help='Int: Iterations for training')


parser.add_argument('-lr', '--learning_rate', 
                    type=float, default=0.1, 
                    help='Float: Learning Rate')

parser.add_argument('-bs', '--batch_size', 
                    type=int, default=128, 
                    help='Int: Training batch size')

parser.add_argument('-sf', '--save_frequency', 
                    type=int, default=1, 
                    help='Int: Model parameters saving frequency ')

args = parser.parse_args()
