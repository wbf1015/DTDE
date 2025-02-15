import torch
import numpy as np

from DataLoader.RandomSample_Reverse import *
from EmbeddingManager.KDManager_Reverse import *
from KGES.RotatE_Reverse import *
from Models.KDModel_Reverse import *
from Loss.SigmoidLossOrigin import *
from Loss.KDLoss import *
from Optim.Optim import *
from Extracter.Constant import *
from Extracter.RNNExtracter2 import *
from Excuter.StandardExcuter_Reverse import *
from utils import *

args = parse_args()
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
set_logger(args)

np.random.randn(args.seed)
torch.manual_seed(args.seed)

'''
声明数据集
'''
train_triples, valid_triples, test_triples, all_true_triples, nentity, nrelation = read_data_reverse(args)

train_dataloader = DataLoader(
    TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size), 
    batch_size=args.batch_size,
    shuffle=True, 
    num_workers=max(1, args.cpu_num//2),
    collate_fn=TrainDataset.collate_fn
)

test_dataloader = DataLoader(
    TestDataset(
        test_triples, 
        all_true_triples, 
        args.nentity, 
        args.nrelation, 
    ), 
    batch_size=args.test_batch_size,
    num_workers=max(1, args.cpu_num//2), 
    collate_fn=TestDataset.collate_fn
)

logging.info('Successfully init TrainDataLoader and TestDataLoader')

'''
声明Excuter组件
'''

if args.negative_adversarial_sampling is False:
    loss=SigmoidLossOrigin(adv_temperature=None, margin=args.gamma, args=args)
else:
    loss=SigmoidLossOrigin(adv_temperature=args.adversarial_temperature, margin=args.gamma, args=args)

if args.negative_adversarial_sampling is False:
    soft_loss = KDLoss(adv_temperature = None, margin = args.kd_gamma, KGEmargin = args.gamma, args=args)
else:
    soft_loss = KDLoss(adv_temperature = args.adversarial_temperature, margin = args.kd_gamma, KGEmargin = args.gamma, args=args)

KGE=RotatE_Reverse(margin=args.gamma)

entity_pruner=RNNExtracter(args, mode=1)
relation_pruner=Constant()

embedding_manager=KDManager_Reverse(args)

trainDataloader =BidirectionalOneShotIterator(train_dataloader)
testDataLoaders=[test_dataloader]

optimizer=None
scheduler=None

Excuter = StandardExcuter_Reverse(
    KGE=KGE, 
    model=KDModel_Reverse,
    embedding_manager=embedding_manager, 
    entity_pruner=entity_pruner, relation_pruner=relation_pruner, 
    decoder=None,
    loss=loss,
    kdloss=soft_loss,
    ContrastiveLoss=None,
    trainDataloader=trainDataloader, testDataLoaders=testDataLoaders,
    optimizer=optimizer, scheduler=scheduler,
    args=args,
)

Excuter.Run()
