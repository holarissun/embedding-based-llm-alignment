import numpy as np

def load_embd_data(task, res_gen_model, embed_model, split):
    raw_embd = np.load(f"../embd/embd_{res_gen_model}_{embed_model}_{task}_{split}.npy")
    raw_reward = np.load(f"../embd/reward_{res_gen_model}_{embed_model}_{task}_{split}.npy")
    return raw_embd, raw_reward
