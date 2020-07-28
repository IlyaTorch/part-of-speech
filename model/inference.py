import torch
import re


def predict_tags(model, data, WORD, TAG, DEVICE='cpu'):
    model.eval()

    with torch.no_grad():
        if type(data) == str:
            words = re.sub('[,\.?!]',' ', data).split()
        else:
            words, _ = data
        example = torch.LongTensor([WORD.vocab.stoi[elem] for elem in words]).unsqueeze(1).to(DEVICE)

        output = model(example).argmax(dim=-1).cpu().numpy()
        tags = [TAG.vocab.itos[int(elem)] for elem in output]

        res = [f'{token:15s}{tag}' for token, tag in zip(words, tags)]
        return res
