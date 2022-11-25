import os
from argparse import ArgumentParser
from cfgyaml import ConfigDir
from datetime import datetime
from pathlib import Path
import logging
import shutil
import random
import numpy as np
import cv2
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn

import albumentations as A
from albumentations.pytorch import ToTensorV2

from data.rellis import RellisImage
import models
from utils import criterions, metrics, optimizers

# region ARGUMENTS
parser = ArgumentParser(description='Implementation of Distilation-based Semantic Segmentation in Off-road.')
parser.add_argument('--config', type=str, help='The configuration filename without extension.')
parser_exp = parser.add_argument_group('Experiment')
parser_exp.add_argument('--exp.directory', type=str, help='The parent directory of the experiment.')
parser_exp.add_argument('--exp.name', type=str, help='The name of the experiment.')
parser_exp.add_argument('--exp.suffix', type=str, help='The suffix follows after the experiment name.')
parser_exp.add_argument('--exp.tensorboard', action='store_true', help='This flag suggests to use tensorboard logging.')
parser_env = parser.add_argument_group('Environments')
parser_env.add_argument('--env.device', type=str, help='The name of device to use. ex) cpu, cuda:0, cuda:1, ...')
parser_env.add_argument('--env.seed', type=int, help='A manual random seed number.')
parser_env.add_argument('--env.deterministic', action='store_true', help='This flag suggests that use only deterministic altorithm.')
parser_env.add_argument('--env.benchmark', action='store_true', help='This flag suggests to find appropriate algorithm for benchmark.')
parser_data = parser.add_argument_group('Dataset')
parser_data.add_argument('--data.input_size', type=int, help='The size of the input image.')
parser_data.add_argument('--data.num_workers', type=int, help='The number of threads to load dataset.')
parser_data.add_argument('--data.batch_size', type=int, help='The size of a batch.')
parser_models = parser.add_argument_group('Models')
parser_models.add_argument('--teacher.params', type=str, help='The filename of the teacher parameter.')
parser_models.add_argument('--student.architect', type=str, help='The architecture of the student.')
parser_models.add_argument('--student.encoder', type=str, help='The encoder of the student.')
parser_models = parser.add_argument_group('Optimization')
parser_models.add_argument('--optim.name', type=str, help='The filename of the teacher parameter.')
parser_models.add_argument('--optim.args.lr', type=str, help='The initial learning rate.')
parser_train = parser.add_argument_group('Training')
parser_train.add_argument('--train.num_epochs', type=int, help='The number of the training epochs.')
parser_train.add_argument('--train.ground_truth_loss', type=str, help='The name of a loss function for the output and the ground truth.')
parser_train.add_argument('--train.teacher_truth_loss', type=str, help='The name of a loss function for the output and the teachers\' output.')
args = parser.parse_args()

config_dir = ConfigDir()
CFG = config_dir.load(config_name=args.config, args=args)
if 'config' in CFG:
    del CFG['config']

if CFG.env.seed is None:
    CFG.env.seed = torch.default_generator.initial_seed() % ((2 << 31) - 1)

now = datetime.now()
CFG.exp.directory = CFG.exp.directory \
    .replace('{date}', now.strftime('%Y%m%d'))[2:] \
    .replace('{time}', now.strftime('%H%M%S'))
# endregion

# region DIRECTORIES
exp_dir = Path('log/').joinpath(CFG.exp.directory, f'{CFG.exp.name}{CFG.exp.suffix}')
best_dir = exp_dir.joinpath('checkpoints')
code_dir = exp_dir.joinpath('codes')
runs_dir = Path('runs/')
for d in (exp_dir, best_dir, code_dir, runs_dir):
    d.mkdir(parents=True, exist_ok=True)
# endregion

# region LOGGING
log_filename = f'{CFG.student.encoder}-{CFG.student.architect}.log'
logger = logging.getLogger('Model')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler = logging.FileHandler(exp_dir.joinpath(log_filename))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def _print(*args, sep=' ', tabs=0, tabchar='\t'):
    args = [str(arg) for arg in args]
    line = (tabchar * tabs) + sep.join(args)
    logger.info(line)
    print(line)
    
if CFG.exp.tensorboard:
    writer = SummaryWriter(runs_dir, filename_suffix=log_filename)

shutil.copy(config_dir.default_file_fullpath, 
            str(code_dir.joinpath(config_dir.default_filename)) + '.yaml')
if args.config is not None:
    shutil.copy(os.path.join(config_dir, args.config.replace('.', os.sep) + '.yaml'), 
                code_dir.joinpath(args.config + '.yaml'))
shutil.copy(__file__, code_dir.joinpath(__file__.split(os.sep)[-1]))

_print(CFG)
# endregion

# region MANUAL SEED & DETERMINISM
use_cuda = CFG.env.device.lower().startswith('cuda')
    
random.seed(CFG.env.seed)
np.random.seed(CFG.env.seed)
torch.manual_seed(CFG.env.seed)
if use_cuda:
    torch.cuda.set_device(CFG.env.device)
    torch.cuda.manual_seed(CFG.env.seed)

cudnn.benchmark = CFG.env.benchmark
cudnn.deterministic = CFG.env.deterministic
# endregion

# region DATASET
train_transform = A.Compose([
    A.HorizontalFlip(p=1.0),
    A.VerticalFlip(p=1.0),
    A.Rotate(limit=[60, 300], p=1.0, interpolation=cv2.INTER_NEAREST),
    A.RandomBrightnessContrast(brightness_limit=[-0.2, 0.3], contrast_limit=0.2, p=1.0),
    A.OneOf([
        A.CLAHE(clip_limit=1.5, tile_grid_size=(8, 8), p=0.5),
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(distort_limit=1, shift_limit=0.5, interpolation=cv2.INTER_NEAREST, p=0.5),
    ], p=1.0),
    A.Resize(width=CFG.data.input_size[0], height=CFG.data.input_size[0]),
    A.Normalize(mean=(0.496588, 0.59493099, 0.53358843), std=(0.496588, 0.59493099, 0.53358843)),
    ToTensorV2(),
])
eval_transform = A.Compose([
    A.Normalize(mean=(0.496588, 0.59493099, 0.53358843), std=(0.496588, 0.59493099, 0.53358843)),
    ToTensorV2(),
])
train_dataset = RellisImage(root='/material/data/Rellis-3D', split='train', transform=train_transform)
eval_dataset = RellisImage(root='/material/data/Rellis-3D', split='val', transform=eval_transform)
train_loader = DataLoader(train_dataset, batch_size=CFG.data.batch_size, num_workers=CFG.data.num_workers)
eval_loader = DataLoader(eval_dataset, batch_size=CFG.data.batch_size, num_workers=CFG.data.num_workers)
# endregion

# region MODELS, OPTIMIZER, CRITERIONS, METRICS
teacher = models.get_model('gscnn', device=CFG.env.device, params=CFG.teacher.params).eval()
student = models.get_model(CFG.student.architect, device=CFG.env.device, encoder_name=CFG.student.encoder, **CFG.student.args).train()
optimizer = optimizers.get_optimizer(CFG.optim.name, params=student.parameters(), **CFG.optim.args)

gt_loss = criterions.get_loss(CFG.train.ground_truth_loss)
tch_loss = criterions.get_loss(CFG.train.teacher_loss)
metric_list = [(item['id'], metrics.get_metrics(item['name']), item['args']) for item in CFG.eval]
# endregion

# region TRAINING
_print('Start training')
for epoch in tqdm(range(1, CFG.train.num_epochs + 1), desc='EPOCH', position=1, leave=False):
    _print('Epoch %03d' % epoch)

    for image, mask in tqdm(train_loader, desc='TRAIN', position=2, leave=False):
        image, mask = image.cuda(), mask.cuda()
        # TODO:

    with torch.no_grad():
        for image, mask in tqdm(eval_loader, desc='EVAL', position=2, leave=False):
            image, mask = image.cuda(), mask.cuda()
            # TODO:

_print('Finish training')
# endregion
