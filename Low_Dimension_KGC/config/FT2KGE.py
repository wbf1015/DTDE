import torch
import numpy as np

from DataLoader.QueryAwareSample_Reverse import *
from EmbeddingManager.KDManager_Reverse_2KGEv2 import *
from KGES.RotatE_Reverse import *
from KGES.AttH_Reverse import *
from KGES.LorentzKG_Reverse.LorentzKG_Reverse import *
from Models.EncoderModel_Reverse_2KGEv2 import *
from Optim.Optim import *
from Decoder.Sim_decoder.Sim_decoderv2 import *
from Extracter.Constant import *
from Excuter.StandardExcuter_Reverse_2KGE import *
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

# train_dataloader2 = DataLoader(
#     TrueTrainDataset(train_triples, nentity, nrelation, args.positive_sample_size, args=args),
#     batch_size=args.batch_size,
#     shuffle=True,
#     num_workers=max(1, args.cpu_num//2),
#     collate_fn=TrueTrainDataset.collate_fn
# )

# query_triples = read_tripels_with_ids('data/FB15k-237/single_query.txt')
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

KGE1 = RotatE_Reverse(teacher_margin=2.0, teacher_embedding_dim=256) # 直接训练的RotatE
# KGE1 = AttH_Reverse(args)

# KGE2 = AttH_Reverse(args)
# KGE2 = HyperNet(args=args)
KGE2 = RotatE_Reverse(teacher_margin=9.0, teacher_embedding_dim=256)

entity_pruner=Constant()
relation_pruner=Constant()

# 还要想一想对抗负采样怎么加进去
decoder = Decoder_2KGEv2(args)

embedding_manager=KDManager_Reverse_2KGEv2(args)

# trainDataloader =BidirectionalOneShotIterator(train_dataloader, train_dataloader2)
trainDataloader =BidirectionalOneShotIterator(train_dataloader)
testDataLoaders=[test_dataloader]

optimizer=None
scheduler=None

Excuter = StandardExcuter_Reverse_2KGE(
    KGE1=KGE1,
    KGE2=KGE2, 
    model=EncoderModel_Reverse_2KGEv2,
    embedding_manager=embedding_manager, 
    entity_pruner=entity_pruner, relation_pruner=relation_pruner, 
    decoder=decoder,
    loss=None,
    kdloss=None,
    ContrastiveLoss=None,
    trainDataloader=trainDataloader, testDataLoaders=testDataLoaders,
    optimizer=optimizer, scheduler=scheduler,
    args=args
)

Excuter.Run()
