import torch
import numpy as np

from DataLoader.RandomSample_Reverse import *
from EmbeddingManager.KDManager_Reverse_Stastic import *
from KGES.AttH_Reverse import *
from Models.KDModelAttH_Reverse import *
from Optim.Optim import *
from Loss.AttHLossSigmoid import *
from Extracter.Constant import *
from Extracter.GRUExtracter import *
from Excuter.StandardExcuter_Reverse import *
from utils import *

args = parse_args()
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
set_logger(args)

# np.random.seed(args.seed)
# torch.manual_seed(args.seed)

'''
声明数据集
'''
train_triples, valid_triples, test_triples, all_true_triples, nentity, nrelation = read_data_reverse(args)

train_dataloader = DataLoader(
    TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, args=args), 
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

KGE=AttH_Reverse(args)

# entity_pruner=GRUExtracter(args, mode=1)
entity_pruner=Constant()
relation_pruner=Constant()

# 还要想一想对抗负采样怎么加进去
decoder=None
# decoder = SCCF_Decoder(args)
# decoder = MarginSCCF_Decoder(args)

embedding_manager=KDManager_Reverse_Stastic(args)

if args.negative_adversarial_sampling is False:
    loss=AttHLossSigmoid(adv_temperature=None, margin=args.gamma, args=args)
else:
    loss=AttHLossSigmoid(adv_temperature=args.adversarial_temperature, margin=args.gamma, args=args)

trainDataloader =BidirectionalOneShotIterator(train_dataloader)
testDataLoaders=[test_dataloader]

optimizer=None
scheduler=None

Excuter = StandardExcuter_Reverse(
    KGE=KGE, 
    model=KDModelAttH_Reverse,
    embedding_manager=embedding_manager, 
    entity_pruner=entity_pruner, relation_pruner=relation_pruner, 
    decoder=decoder,
    loss=loss,
    kdloss=None,
    ContrastiveLoss=None,
    trainDataloader=trainDataloader, testDataLoaders=testDataLoaders,
    optimizer=optimizer, scheduler=scheduler,
    args=args,
)

Excuter.Run()
