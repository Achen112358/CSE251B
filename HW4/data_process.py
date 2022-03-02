from torch.utils.data import IterableDataset, DataLoader
import json

class MultiFileIterDataset(IterableDataset):
    def __init__(self, path):
        super().__init__()
        self.path = Path(path)

    def __iter__(self):
        for f in self.path.iterdir():
            if not f.is_file():
                continue
            with open(f, 'r') as content:
                for record in content:
                    yield self.format(record)

    def format(self, record):
        raise NotImplementedError


class SimpleRead(MultiFileIterDataset):
    def format(self, record):
        list_data = json.loads(record)
        return list_data

def collate_fn(batch):
    """
    处理batch输入的格式
    """
    feature_idx = list(range(2, 241))
    y = [[e[1] for e in all_e] for all_e in batch]
    X = [[ [z] + [e[idx] for idx in feature_idx] for z, e in zip( [0.0] + yy[:-1],  all_e)] for all_e, yy in zip(batch, y)]
    return (X, y)

f = open("train_data.json")
for i, (images, captions, img_ids) in enumerate(val_loader):
    images = images.to(device)
    vector = encoder(images)
    for v, c, id in zip(vector, captions, img_ids):
        v = list(v.cpu().detach().numpy())
        c = list(c)
        ret = json.dumps([v, c, id])
        f.write(ret + '\n')
f.close()