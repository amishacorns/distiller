import torch
import pickle
import distiller.quantization.q_utils as utils

BITS = 6
SCALE = 30
CASCADE_FACTOR = 7
ARR_HEIGHT = 16  # height of the systolic array
ACT_LIMIT = int(1e6)  # limit the number of activations for development


def is_outlier(act):
    q_min, q_max = utils.get_quantized_range(BITS)
    q_act = utils.linear_quantize(act, SCALE, 0)
    representable = q_min < q_act < q_max

    return not representable


def overwrite_downstream(downstream):
    for act in downstream[:CASCADE_FACTOR]:
        if overwrite_act(act):
            return True
    return False


def overwrite_act(act):
    q_min, q_max = utils.get_quantized_range(BITS)
    q_act = utils.linear_quantize(act, SCALE, 0)
    # I believe this is the method Ritchie discusses in the arxiv
    small = q_min/4 < q_act < q_max/4

    return small

file = open('test_tensor.p', 'rb')
acts = pickle.load(file)

# TODO: Check that this streaming pattern makes sense, probably not realistic
# Activations in the same output channel to stream in output channel first
# (B, C, H, W) -> (B, H, W, C)
acts = acts.permute(0, 2, 3, 1)
numel = acts.numel()
n_pad = numel % ARR_HEIGHT if ACT_LIMIT == 0 else ACT_LIMIT % ARR_HEIGHT
acts_flat = torch.flatten(acts) if ACT_LIMIT == 0 else torch.flatten(acts)[:ACT_LIMIT]
acts_stream = torch.cat([acts_flat, torch.zeros(n_pad)])
acts_stream = acts_stream.reshape(-1, ARR_HEIGHT)

outliers = 0
covered = 0  # number of outliers covered
for stream in acts_stream:
    for act_i, act in enumerate(stream):
        if is_outlier(act):
            outliers = outliers + 1
            down = stream[act_i+1:]
            if overwrite_downstream(down):
                covered = covered + 1
                break

coverage = covered / outliers if outliers > 0 else None
outlier_ratio = outliers / numel
print("Covered: {}, Outliers: {}, Coverage: {}, Outlier Ratio: {}".format(covered, outliers, coverage, outlier_ratio))
