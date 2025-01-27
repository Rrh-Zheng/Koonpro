from data.dataset_gen import takens_embed, read_data
from torch.utils.data import DataLoader
import torch

def data_provider(args, flag):
    time_shifts = args.time_shifts
    window = args.window
    data = read_data(args, flag)
    data_embed = takens_embed(data, time_shifts=time_shifts, window=window)

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = data_embed.shape[0]
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
    data_loader = DataLoader(
        data_embed,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=0,
        drop_last=drop_last)
    context_set = torch.from_numpy(data_embed).float()
    return context_set, data, data_loader

