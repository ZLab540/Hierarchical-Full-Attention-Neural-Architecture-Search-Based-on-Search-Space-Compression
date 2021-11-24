from    collections import namedtuple



Genotype = namedtuple('Genotype', 'normal1 normal_concat1 normal2 normal_concat2 normal3 normal_concat3 reduce1 reduce_concat1 reduce2 reduce_concat2')



PRIMITIVES = [
    # 'none',
    'attention_3*3_group_4',
    'attention_5*5_group_4',
    'attention_7*7_group_4',
    'attention_3*3_group_8',
    'attention_5*5_group_8',
    'attention_7*7_group_8',
    'NonLocal_Attention'

]
# PRIMITIVES = [
#     'none',
#     'max_pool_3x3',
#     'avg_pool_3x3',
#     'skip_connect',
#     'sep_conv_3x3',
#     'sep_conv_5x5',
#     'dil_conv_3x3',
#     'dil_conv_5x5'
# ]

# NASNet = Genotype(
#     normal=[
#         ('sep_conv_5x5', 1),
#         ('sep_conv_3x3', 0),
#         ('sep_conv_5x5', 0),
#         ('sep_conv_3x3', 0),
#         ('avg_pool_3x3', 1),
#         ('skip_connect', 0),
#         ('avg_pool_3x3', 0),
#         ('avg_pool_3x3', 0),
#         ('sep_conv_3x3', 1),
#         ('skip_connect', 1),
#     ],
#     normal_concat=[2, 3, 4, 5, 6],
#     reduce=[
#         ('sep_conv_5x5', 1),
#         ('sep_conv_7x7', 0),
#         ('max_pool_3x3', 1),
#         ('sep_conv_7x7', 0),
#         ('avg_pool_3x3', 1),
#         ('sep_conv_5x5', 0),
#         ('skip_connect', 3),
#         ('avg_pool_3x3', 2),
#         ('sep_conv_3x3', 2),
#         ('max_pool_3x3', 1),
#     ],
#     reduce_concat=[4, 5, 6],
# )
#
# AmoebaNet = Genotype(
#     normal=[
#         ('avg_pool_3x3', 0),
#         ('max_pool_3x3', 1),
#         ('sep_conv_3x3', 0),
#         ('sep_conv_5x5', 2),
#         ('sep_conv_3x3', 0),
#         ('avg_pool_3x3', 3),
#         ('sep_conv_3x3', 1),
#         ('skip_connect', 1),
#         ('skip_connect', 0),
#         ('avg_pool_3x3', 1),
#     ],
#     normal_concat=[4, 5, 6],
#     reduce=[
#         ('avg_pool_3x3', 0),
#         ('sep_conv_3x3', 1),
#         ('max_pool_3x3', 0),
#         ('sep_conv_7x7', 2),
#         ('sep_conv_7x7', 0),
#         ('avg_pool_3x3', 1),
#         ('max_pool_3x3', 0),
#         ('max_pool_3x3', 1),
#         ('conv_7x1_1x7', 0),
#         ('sep_conv_3x3', 5),
#     ],
#     reduce_concat=[3, 4, 6]
# )
#
# DARTS_V1 = Genotype(
#     normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1),
#             ('skip_connect', 0),
#             ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)],
#     normal_concat=[2, 3, 4, 5],
#     reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0),
#             ('max_pool_3x3', 0),
#             ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)],
#     reduce_concat=[2, 3, 4, 5])
# DARTS_V2 = Genotype(
#     normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
#             ('sep_conv_3x3', 1),
#             ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)],
#     normal_concat=[2, 3, 4, 5],
#     reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1),
#             ('max_pool_3x3', 0),
#             ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)],
#     reduce_concat=[2, 3, 4, 5])
#
#
#
# MyDARTS = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 0)], normal_concat=range(2, 6), reduce=[('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('max_pool_3x3', 2), ('sep_conv_5x5', 1), ('dil_conv_5x5', 3), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))
#
# DARTS_ATTENTION=Genotype(normal=[('attention_3*3_group_8', 0), ('NonLocal_Attention', 1), ('attention_3*3_group_8', 2)], normal_concat=range(3, 4), reduce=[('attention_3*3_group_16', 0), ('attention_7*7_group_16', 1), ('NonLocal_Attention', 2)], reduce_concat=range(3, 4))
# # DARTS_ATTENTION=Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], normal_concat=range(3,4), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], reduce_concat=range(3,4))
# DARTS_ATTENTION_2 = Genotype(normal=[('NonLocal_Attention', 0), ('NonLocal_Attention', 1), ('attention_3*3_group_8', 2)], normal_concat=range(3,4), reduce=[('attention_3*3_group_8', 0), ('attention_7*7_group_8', 1), ('attention_7*7_group_16', 2)], reduce_concat=range(3, 4))

ATTENTION = Genotype(normal1=[('attention_5*5_group_8', 0), ('attention_5*5_group_8', 1), ('attention_3*3_group_4', 2)], normal_concat1=range(3, 4), normal2=[('attention_5*5_group_4', 0), ('attention_5*5_group_8', 1), ('attention_3*3_group_8', 2)], normal_concat2=range(3, 4), normal3=[('NonLocal_Attention', 0), ('NonLocal_Attention', 1), ('NonLocal_Attention', 2)], normal_concat3=range(3, 4), reduce1=[('max_pool_3x3', 0), ('attention_3*3_group_4', 1), ('attention_5*5_group_4', 2)], reduce_concat1=range(3, 4), reduce2=[('NonLocal_Attention', 0), ('attention_5*5_group_8', 1), ('attention_5*5_group_8', 2)], reduce_concat2=range(3, 4))
ATTENTION1 = Genotype(normal1=[('attention_3*3_group_4', 0), ('attention_3*3_group_4', 1), ('attention_5*5_group_4', 2)], normal_concat1=range(3, 4), normal2=[('attention_5*5_group_4', 0), ('attention_3*3_group_8', 1), ('attention_5*5_group_4', 2)], normal_concat2=range(3, 4), normal3=[('NonLocal_Attention', 0), ('NonLocal_Attention', 1), ('NonLocal_Attention', 2)], normal_concat3=range(3, 4), reduce1=[('attention_3*3_group_4', 0), ('NonLocal_Attention', 1), ('attention_5*5_group_4', 2)], reduce_concat1=range(3, 4), reduce2=[('attention_5*5_group_4', 0), ('attention_5*5_group_4', 1), ('NonLocal_Attention', 2)], reduce_concat2=range(3, 4))
ATTENTION2 = Genotype(normal1=[('attention_7*7_group_4', 0), ('attention_5*5_group_4', 1), ('attention_3*3_group_8', 2)], normal_concat1=range(3, 4), normal2=[('attention_3*3_group_8', 0), ('attention_3*3_group_4', 1), ('attention_5*5_group_8', 2)], normal_concat2=range(3, 4), normal3=[('NonLocal_Attention', 0), ('attention_3*3_group_8', 1), ('attention_3*3_group_4', 2)], normal_concat3=range(3, 4), reduce1=[('attention_3*3_group_4', 0), ('attention_7*7_group_8', 1), ('attention_5*5_group_4', 2)], reduce_concat1=range(3, 4), reduce2=[('attention_3*3_group_4', 0), ('attention_3*3_group_4', 1), ('attention_7*7_group_8', 2)], reduce_concat2=range(3, 4))
DARTS = ATTENTION2
