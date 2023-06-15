import numpy as np
import os
data_folder = 'Dataset/ednet'
embed_file = 'embedding_200_original.npz'
embed_data = np.load(os.path.join(data_folder, embed_file))
_, _, pre_pro_embed = embed_data['pro_repre'], embed_data['skill_repre'], embed_data['pro_final_repre']
print(pre_pro_embed.shape)
np.savez(os.path.join(data_folder,'embed_pretrain.npz'),q_embed=pre_pro_embed[1:,:])
