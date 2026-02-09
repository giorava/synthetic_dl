import sys
sys.path.insert(0, "../src")
sys.path.insert(0, "../bpnet")
import models, train, logging, torch
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

### specify a model
number_dilated_conv = 9
tasks = 2
number_convolutional_filters = 90
filter_width = 7
learning_rate = 0.004
epochs = 10

logging.info(f"loading motif")
model = models.BPNetSingleHeadProfile(
    numFilters = number_convolutional_filters, 
    widthFilters = filter_width, 
    n_convolutions = number_dilated_conv+1, 
    number_tasks = tasks
).to(device)

logging.info(f"compiling adam")
optimizer = torch.optim.Adam(
    model.parameters(),
    lr = learning_rate
)

logging.info(f"training object")
training = train.trainBPNet(
    model = model,
    optimizer = optimizer, 
    path_train_dataset = "/homes/users/gravanelli/scratch/synthetic_dl/demos/data_dl/train.h5", 
    path_val_dataset = "/homes/users/gravanelli/scratch/synthetic_dl/demos/data_dl/val.h5", 
    model_ouput = "/homes/users/gravanelli/scratch/synthetic_dl/demos/models", 
    fraction_profile = 0.1,
    number_tasks = 2,
    init_lambda = 32.0,
    batch_size = 64
)

logging.info(f"Starting training")
training.train(epochs = epochs)

