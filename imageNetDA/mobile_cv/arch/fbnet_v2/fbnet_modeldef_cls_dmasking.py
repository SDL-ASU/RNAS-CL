
from .fbnet_modeldef_cls import MODEL_ARCH
from .modeldef_utils import _ex, e1, e6

BASIC_ARGS = {}

IRF_CFG = {"less_se_channels": False}


MODEL_ARCH_DMASKING_NET_l1 = {
    "fbnetv2_supernet_l1": {
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3_hs", 16, 2, 1]],  ###
            # stage 1
            [["ir_k3_hs", 160, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 160, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 160, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 160, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 160, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 160, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 160, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 160, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 160, 1, 1, _ex(6), IRF_CFG],
            ],
            # stage 2
            [
                ["ir_k3_hs", 320, 2, 1, _ex(6), IRF_CFG],  ### stage 2
                ["ir_k3_hs", 320, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 320, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 320, 1, 1, _ex(6), IRF_CFG],  ### stage 2
                ["ir_k3_hs", 320, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 320, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 320, 1, 1, _ex(6), IRF_CFG],

            ],
            # stage 3      
            [
                ["ir_k3_hs", 256, 2, 1, _ex(6), IRF_CFG],  ### stage 3
            ],
        ],
    },
}
MODEL_ARCH.register_dict(MODEL_ARCH_DMASKING_NET_l1)


MODEL_ARCH_DMASKING_NET_l2 = {
    "fbnetv2_supernet_l2": {
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3_hs", 16, 2, 1]],  ###
            # stage 1
            [["ir_k3_hs", 80, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 80, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 80, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 80, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 80, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 80, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 80, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 80, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 80, 1, 1, _ex(6), IRF_CFG],
            ],
            # stage 2
            [
                ["ir_k3_hs", 160, 2, 1, _ex(6), IRF_CFG],  ### stage 2
                ["ir_k3_hs", 160, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 160, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 160, 1, 1, _ex(6), IRF_CFG],  ### stage 2
                ["ir_k3_hs", 160, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 160, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 160, 1, 1, _ex(6), IRF_CFG],

            ],
            # stage 3
            [
                ["ir_k3_hs", 128, 2, 1, _ex(6), IRF_CFG],  ### stage 3
            ],
        ],
    },
}
MODEL_ARCH.register_dict(MODEL_ARCH_DMASKING_NET_l2)

MODEL_ARCH_DMASKING_NET_S3 = {
    "fbnetv2_supernet_s3": {
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3_hs", 16, 2, 1]],  ###
            # stage 1
            [["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
            ],
            # stage 2
            [
                ["ir_k3_hs", 32, 2, 1, _ex(6), IRF_CFG],  ### stage 2
                ["ir_k3_hs", 32, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 32, 1, 1, _ex(6), IRF_CFG],
            ],
            # stage 3      
            [
                ["ir_k3_hs", 64, 2, 1, _ex(6), IRF_CFG],  ### stage 3
                ["ir_k3_hs", 64, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(6), IRF_CFG],
            ],
        ],
    },
}
MODEL_ARCH.register_dict(MODEL_ARCH_DMASKING_NET_S3)

MODEL_ARCH_DMASKING_NET_S5 = {
    "fbnetv2_supernet_s5": {
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3_hs", 16, 2, 1]],  ###
            # stage 1
            [["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
            ],
            # stage 2
            [
                ["ir_k3_hs", 32, 2, 1, _ex(6), IRF_CFG],  ### stage 2
                ["ir_k3_hs", 32, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 32, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 32, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 32, 1, 1, _ex(6), IRF_CFG],
            ],
            # stage 3      
            [
                ["ir_k3_hs", 64, 2, 1, _ex(6), IRF_CFG],  ### stage 3
                ["ir_k3_hs", 64, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(6), IRF_CFG],
            ],
        ],
    },
}
MODEL_ARCH.register_dict(MODEL_ARCH_DMASKING_NET_S5)

'''
MODEL_ARCH_DMASKING_NET_S5 = {
    "fbnetv2_supernet_s5": {
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3_hs", 16, 2, 1]],  ###
            # stage 1
            [["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
            ],
            # stage 2
            [
                ["ir_k3_hs", 20, 2, 1, _ex(6), IRF_CFG],  ### stage 2
                ["ir_k3_hs", 20, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 20, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 20, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 20, 1, 1, _ex(6), IRF_CFG],
            ],
            # stage 3      
            [
                ["ir_k3_hs", 52, 2, 1, _ex(6), IRF_CFG],  ### stage 3
                ["ir_k3_hs", 52, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 52, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 52, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 52, 1, 1, _ex(6), IRF_CFG],
            ],
        ],
    },
}
MODEL_ARCH.register_dict(MODEL_ARCH_DMASKING_NET_S5)

'''
MODEL_ARCH_DMASKING_NET_S7 = {
    "fbnetv2_supernet_s7": {
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3_hs", 16, 2, 1]],  ###
            # stage 1
            [["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
            ],
            # stage 2
            [
                ["ir_k3_hs", 32, 2, 1, _ex(6), IRF_CFG],  ### stage 2
                ["ir_k3_hs", 32, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 32, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 32, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 32, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 32, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 32, 1, 1, _ex(6), IRF_CFG],
            ],
            # stage 3      
            [
                ["ir_k3_hs", 64, 2, 1, _ex(6), IRF_CFG],  ### stage 3
                ["ir_k3_hs", 64, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(6), IRF_CFG],
            ],
        ],
    },
}
MODEL_ARCH.register_dict(MODEL_ARCH_DMASKING_NET_S7)


'''
MODEL_ARCH_DMASKING_NET_S7 = {
    "fbnetv2_supernet_s7": {
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3_hs", 16, 2, 1]],  ###
            # stage 1
            [["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
            ],
            # stage 2
            [
                ["ir_k3_hs", 20, 2, 1, _ex(6), IRF_CFG],  ### stage 2
                ["ir_k3_hs", 20, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 20, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 20, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 20, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 20, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 20, 1, 1, _ex(6), IRF_CFG],
            ],
            # stage 3      
            [
                ["ir_k3_hs", 52, 2, 1, _ex(6), IRF_CFG],  ### stage 3
                ["ir_k3_hs", 52, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 52, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 52, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 56, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 56, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 52, 1, 1, _ex(6), IRF_CFG],
            ],
        ],
    },
}
MODEL_ARCH.register_dict(MODEL_ARCH_DMASKING_NET_S7)

'''

MODEL_ARCH_DMASKING_NET_S9 = {
    "fbnetv2_supernet_s9": {
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3_hs", 16, 2, 1]],  ###
            # stage 1
            [["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
            ],
            # stage 2
            [
                ["ir_k3_hs", 32, 2, 1, _ex(6), IRF_CFG],  ### stage 2
                ["ir_k3_hs", 32, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 32, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 32, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 32, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 32, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 32, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 32, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 32, 1, 1, _ex(6), IRF_CFG],
            ],
            # stage 3      
            [
                ["ir_k3_hs", 64, 2, 1, _ex(6), IRF_CFG],  ### stage 3
                ["ir_k3_hs", 64, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(6), IRF_CFG],
            ],
        ],
    },
}
MODEL_ARCH.register_dict(MODEL_ARCH_DMASKING_NET_S9)

MODEL_ARCH_DMASKING_NET_S18 = {
    "fbnetv2_supernet_s18": {
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3_hs", 16, 2, 1]],  ###
            # stage 1
            [["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
             ["ir_k3_hs", 16, 1, 1, _ex(6), IRF_CFG],
            ],
            # stage 2
            [
                ["ir_k3_hs", 32, 2, 1, _ex(6), IRF_CFG],  ### stage 2
                ["ir_k3_hs", 32, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 32, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 32, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 32, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 32, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 32, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 32, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 32, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 32, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 32, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 32, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 32, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 32, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 32, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 32, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 32, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 32, 1, 1, _ex(6), IRF_CFG],
            ],
            # stage 3      
            [
                ["ir_k3_hs", 64, 2, 1, _ex(6), IRF_CFG],  ### stage 3
                ["ir_k3_hs", 64, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 64, 1, 1, _ex(6), IRF_CFG],
            ],
        ],
    },
}
MODEL_ARCH.register_dict(MODEL_ARCH_DMASKING_NET_S18)


MODEL_ARCH_DMASKING_NET = {
    "fbnetv2_supernet": {
        "basic_args": BASIC_ARGS,
        "blocks": [
            # [c, s, n, ...]
            # stage 0
            [["conv_k3_hs", 16, 2, 1]],  ###
            # stage 1
            [["ir_k3_hs", 16, 1, 1, e1, IRF_CFG]],  ### stage 1
            # stage 2
            [
                ["ir_k3_hs", 28, 2, 1, _ex(6), IRF_CFG],  ### stage 2
                ["ir_k3_hs", 28, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 28, 1, 1, _ex(6), IRF_CFG],
            ],
            # stage 3
            [
                ["ir_k3_hs", 40, 2, 1, _ex(6), IRF_CFG],  ### stage 3
                ["ir_k3_hs", 40, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 40, 1, 1, _ex(6), IRF_CFG],
            ],
            # stage 4
            [
                ["ir_k3_hs", 96, 2, 1, _ex(6), IRF_CFG],  ### stage 4
                ["ir_k3_hs", 96, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 96, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 128, 1, 1, _ex(6), IRF_CFG],   ### stage 5
                ["ir_k3_hs", 128, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 128, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 128, 1, 1, _ex(6), IRF_CFG],
            ],
            # stage 5
            [
                ["ir_k3_hs", 216, 2, 1, _ex(6), IRF_CFG],  ### stage 6
                ["ir_k3_hs", 216, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 216, 1, 1, _ex(6), IRF_CFG],
                ["ir_k3_hs", 216, 1, 1, _ex(6), IRF_CFG],
            ],
            # stage 6
            [["ir_pool_hs", 1984, 1, 1, e6]],  ### 12 13
        ],
    },
}
MODEL_ARCH.register_dict(MODEL_ARCH_DMASKING_NET)

