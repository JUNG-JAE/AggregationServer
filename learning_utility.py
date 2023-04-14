# ------------ system library ------------ #
import sys
import numpy as np
from collections import OrderedDict

# ----------- Learning library ----------- #
import torch

# ------------ custom library ------------ #
from conf.global_settings import LOG_DIR, DATA_TYPE


def get_network(args):
    torch.manual_seed(777)

    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif args.net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34()
    elif args.net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50()
    elif args.net == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101()
    elif args.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu:  # use_gpu
        torch.cuda.manual_seed_all(777)
        net = net.cuda()

    return net


def aggregation(args, *models):
    aggregation_model = get_network(args)
    aggregation_model_dict = OrderedDict()

    for index, model in enumerate(models):
        for layer in model.state_dict().keys():
            if index == 0:
                aggregation_model_dict[layer] = 1 / len(models) * model.state_dict()[layer]
            else:
                aggregation_model_dict[layer] += 1 / len(models) * model.state_dict()[layer]

    aggregation_model.load_state_dict(aggregation_model_dict)

    return aggregation_model


def load_model(model, args, global_round, shard_id):
    model.load_state_dict(torch.load(f"./{LOG_DIR}/{DATA_TYPE}/{args.net}/global_model/G{global_round}/{shard_id}.pt"), strict=True)

    return model


def save_model(model, args, global_round):
    torch.save(model.state_dict(), f"{LOG_DIR}/{DATA_TYPE}/{args.net}/global_model/G{global_round}/aggregation.pt")
