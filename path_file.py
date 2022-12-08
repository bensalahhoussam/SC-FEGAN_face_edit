import argparse

original_dataset = "D://CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img/"

train_path = "D://Deep_Learning_projects/new_projects/computer_vision/project_3/dataset/"

valid_path = "D://CelebAMask-HQ/validation/"

hed_pretrained ="hed_pretrained_bsds.caffemodel"

hed_depoly="deploy.prototxt.txt"

path_bisenet_weights ="D://Deep_Learning_projects/new_projects/computer_vision/project_3/BiSeNet_keras.h5"

batch_size = 4
learning_rate = 1e-4
beta_1=0.1
beta_2=0.999
epochs = 2
batch_size_per_replica = 3



parser = argparse.ArgumentParser()
parser.add_argument('--training_dataset', type = str,default = train_path, help = 'path of training dataset')
parser.add_argument('--validation_dataset', type = str,default = valid_path, help = 'path of validition dataset')
parser.add_argument('--original_dataset', type = str,default = original_dataset, help = 'path of original dataset')
parser.add_argument('--hed_pretrained', type = str,default = hed_pretrained, help = 'pretrained caffe model')
parser.add_argument('--hed_deploy', type = str,default = hed_depoly, help = 'hed deplyment')
parser.add_argument('--batch_size', type = int,default = batch_size, help = 'number of batch size ')
parser.add_argument('--learning_rate', type = float,default = learning_rate, help = 'learning rate ')
parser.add_argument('--beta_1', type = float,default = beta_1, help = 'Adam beta_1 ')
parser.add_argument('--beta_2', type = float,default = beta_2, help = 'Adam beta_2 ')
parser.add_argument('--epochs', type = int,default = epochs, help = 'total epoch number ')
parser.add_argument('--batch_size_per_replica', type=int,default= batch_size_per_replica,help="batch_size_per_replica")
parser.add_argument('--bisenet_weights', type = str,default = path_bisenet_weights, help = 'bisenet_weights ')
args = parser.parse_args()

