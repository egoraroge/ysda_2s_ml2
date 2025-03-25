# Описание структуры присылаемого архива - для самопроверки
#  - submit_main.py
#  - vocab.tsv
#  - checkpoint

# ВАЖНО: если в любой функции есть параметры - не меняйте их порядок и не переименовывайте,
#   если требуется добавить ещё параметры, то добавляйте в конец и обязательно с установленными default-ами

# 0. Все необходимые import-ы
from torch.utils.data import Dataset, DataLoader
from torch import nn
from typing import Dict, List, Optional, Tuple
from torchvision import models
from torchvision import transforms as tr
import os
import pandas as pd
import numpy as np
import torch
import cv2
import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm, trange
import re

# 1. Подготовка данных

tok_to_ind = {}
ind_to_tok = {}
## Как прочитать словарь, переданный вами внутри архива - используйте эту функцию в своём датасете
def get_vocab(unzip_root: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    global tok_to_ind
    global ind_to_tok
    """
        unzip_root ~ в тестовой среде будет произведена операция `unzip archive.zip` с переданным архивом и в эту функцию будет передан путь до `realpath .`
    """
    vocab_path = os.path.join(unzip_root, "vocab.tsv")
    tok_to_ind = {}
    ind_to_tok = {}
    
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            token, idx_str = line.strip().split("\t")
            idx = int(idx_str)
            tok_to_ind[token] = idx
            ind_to_tok[idx] = token
    
    return tok_to_ind, ind_to_tok

tok_to_ind, ind_to_tok = get_vocab('./')


## Ваш датасет
class ImageCaptioningDataset(Dataset):
    """
        imgs_path ~ путь к папке с изображениями
        captions_path ~ путь к .tsv файлу с заголовками изображений
    """
    def __init__(self, imgs_path, captions_path, train=True):
        super(ImageCaptioningDataset).__init__()
        # Читаем и записываем из файлов в память класса, чтобы быстро обращаться внутри датасета
        # Если не хватает памяти на хранение всех изображений, то подгружайте прямо во время __getitem__, но это замедлит обучение
        # Проведите всю предобработку, которую можно провести без потери вариативности датасета, здесь

        def tokenize(text):
            text = text.lower()
            text = re.sub(r'\W', ' ', text)
            tokens = re.split(r'\s+', text.strip(' \t\n'))
            tokens = ['<BOS>'] + tokens + ['<EOS>']
            return tokens
        
        def to_ids(text):
            return [tok_to_ind[token] if token in tok_to_ind else tok_to_ind['<UNK>'] for token in tokenize(text)]
    
        self.train = train
        self.captions_df = pd.read_csv(captions_path, sep='\t')
        
        for ind, row in tqdm(self.captions_df.iterrows()):
            for i in range(5):
                row[f'caption #{i}'] = to_ids(row[f'caption #{i}'])
        
        self.captions_df['img'] = [cv2.imread(os.path.join(imgs_path, img_id)) for img_id in self.captions_df['img_id']]
        

    def __getitem__(self, index):
        # Получаем предобработанное изображение (не забудьте отличие при train=True или train=False)
        
        row = self.captions_df.iloc[index]
        # Берём все заголовки или только один случайный (случайность должна происходить при каждом вызове __getitem__, 
        #  чтобы во время обучения вы в разных эпохах могли видеть разные заголовки для одного изображения)
        channel_mean = np.array([0.485, 0.456, 0.406])
        channel_std = np.array([0.229, 0.224, 0.225])
        
        image_prepare = tr.Compose([
            tr.ToPILImage(),
            # Любые преобразования, которые вы захотите:
            #   https://pytorch.org/vision/stable/transforms.html
            
            tr.Resize(size=256), # подгоним все изображения под один размер
            tr.RandomPerspective(distortion_scale=0.2, p=1.0),
            tr.RandomRotation(degrees=(-15, 15)),
            tr.RandomHorizontalFlip(),
            tr.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1)),
            tr.ColorJitter(brightness=.3, contrast=.2, saturation=.2),
            tr.RandomCrop((224, 224)),
            tr.ToTensor(),
            tr.Normalize(mean=channel_mean, std=channel_std),
        ])
        image_prepare_val = tr.Compose([
            tr.ToPILImage(),
            tr.Resize(size=224), # подгоним все изображения под один размер
            tr.CenterCrop((224, 224)),
            tr.ToTensor(),
            tr.Normalize(mean=channel_mean, std=channel_std),
        ])
        
        if self.train:
            image = image_prepare(row['img'])
        else:
            image = image_prepare_val(row['img'])
        caption = [row[f'caption #{i}'] for i in range(5)] 
        return image, caption
    
    def __len__(self):
        return self.captions_df.shape[0]

## Ваш даталоадер
def collate_fn(batch):
    # Функция получает на вход batch - представляет из себя List[el], где каждый el - один вызов __getitem__
    #  вашего датасета
    # На выход вы выдаёте то, что будет выдавать Dataloader на каждом next() из генератора - вы хотите иметь на выходе
    #  несколько тензоров
    
    # Моё предложение по тому как должен выглядеть батч на выходе:
    #   img_batch: [batch_size, num_channels, height, width] --> сложенные в батч изображения
    #   captions_batch: [batch_size, num_captions_per_image, max_seq_len or local_max_seq_len] --> сложенные в
    #       батч заголовки при помощи padding-а
    global tok_to_ind
    
    img_batch, captions_batch = zip(*batch)
    img_batch = torch.stack(img_batch, dim=0)
    print(captions_batch[0])
    max_caption_len = max([max([len(caption) for caption in captions]) for captions in captions_batch])
    padded_captions = []
    for i, captions in enumerate(captions_batch):
        for j, caption in enumerate(captions):
            captions_batch[i][j] += [tok_to_ind['<PAD>']] * (max_caption_len - len(caption))
        
    captions_batch = torch.tensor(captions_batch, dtype=torch.int64)
    return img_batch, captions_batch

def get_val_dataloader(dataset, batch_size=64):
    num_workers = 0
    return DataLoader(dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    

# 2. Построение модели

## Аргументы для общего класса
init_kwargs = {'vocab_size': 3478,
        'glove_weights': torch.zeros(3478, 300),
        'img_feature_dim': 256,
        'hidden_size': 512,
        'num_layers': 2,
        'num_captions': 5}

## Общий класс модели
from collections import OrderedDict
from einops import rearrange
from torch.utils.data import Dataset, DataLoader
from torch import nn
from typing import Dict, List, Optional, Tuple
from torchvision import models

class img_fe_class(nn.Module):
    def __init__(self, pretrained=True, freeze_layers=True):
        super(img_fe_class, self).__init__()
        resnet = models.resnet18()
        if freeze_layers:
            for param in resnet.parameters():
                param.requires_grad = False
        self.additional_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(resnet.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
            
            
    def forward(self, imgs):
        x = self.resnet(imgs)
        x = self.additional_layers(x)
        return x

class text_fe_class(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers, num_captions, glove_weights, img_feature_size):
        super(text_fe_class, self).__init__()

        #tok_to_ind, _ = get_vocab("")
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=300, padding_idx=3)
        #self.embed.weight = nn.Parameter(
        #    torch.from_numpy(glove_weights).to(dtype=self.embed.weight.dtype),
        #    requires_grad=True,
        #)

        self.lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5,
        )
        self.img_linear = nn.Linear(img_feature_size, hidden_size)
        
    def forward(self, texts, img_features):
        batch_size, num_captions, seq_len = texts.shape
        texts = texts.reshape(-1, seq_len)  # (batch_size * num_captions, seq_len)
        
        text_emb = self.embed(texts)
        img_features = self.img_linear(img_features)
        
        img_features = img_features.unsqueeze(1).repeat(1, num_captions, 1)
        img_features = img_features.reshape(-1, img_features.shape[-1])  # (batch_size * num_captions, hidden_size)
        
        h_0 = (img_features.unsqueeze(0).repeat(self.lstm.num_layers, 1, 1), torch.zeros_like(img_features).unsqueeze(0).repeat(self.lstm.num_layers, 1, 1))
        lstm_out, _ = self.lstm(text_emb, h_0)
        
        # (batch_size, num_captions, seq_len, hidden_size)
        lstm_out = lstm_out.reshape(batch_size, num_captions, seq_len, -1)
        
        return lstm_out


class image_captioning_model(nn.Module):
    def __init__(self, vocab_size, glove_weights, img_feature_dim, hidden_size, num_layers, num_captions):
        super(image_captioning_model, self).__init__()
        self.image_feature_extractor = img_fe_class(pretrained=True, freeze_layers=True)
        self.text_feature_extractor = text_fe_class(vocab_size=vocab_size, emb_size=300, hidden_size=hidden_size, num_layers=num_layers, num_captions=num_captions, glove_weights=glove_weights, img_feature_size=256)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, img_batch, texts_batch):
        img_features = self.image_feature_extractor(img_batch)
        text_features = self.text_feature_extractor(texts_batch, img_features)
        return self.fc(text_features)
# 3. Обучение модели

## Сборка вашей модели с нужными параметрами и подгрукой весов из чекпоинта
import torch.optim as optim

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_model(unzip_root: str):
    """
        unzip_root ~ в тестовой среде будет произведена операция `unzip archive.zip` с переданным архивом и в эту функцию будет передан путь до `realpath .`
    """
    #glove_path = os.path.join(unzip_root, "glove.840B.300d.txt")
    #glove_weights, mask_found = load_glove_weights(glove_path, tok_to_ind, "<PAD>")
    

    model_params = {
        'vocab_size': 3478,
        'glove_weights': None,
        'img_feature_dim': 256,
        'hidden_size': 512,
        'num_layers': 2,
        'num_captions': 5
    }
    
    optimizer_params = {
        'lr': 0.001,
        'weight_decay': 1e-5
    }

    
    def create_model_and_optimizer(model_class, model_params, optimizer_params, device=device):
        model = model_class(**model_params)
        model = model.to(device)
        
        optimizer = optim.Adam(model.parameters(), **optimizer_params)
        return model, optimizer
    
    model, optimizer = create_model_and_optimizer(image_captioning_model, model_params, optimizer_params)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    checkpoint = torch.load(
        os.path.join(unzip_root, "model.pt"),
        map_location=device,
        weights_only=False,
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    losses = checkpoint['losses']
    return model

# 4. Оценка результатов

## Генерация предсказания по картинке
from typing import Optional
def generate(
    model,
    image,
    max_seq_len: Optional[int],
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
):
    """
    По картинке image генерируете текст моделью model либо пока не сгенерируете '<EOS>' токен, либо пока не сгенерируете max_seq_len токенов
        top_k -> после получения предсказания оставляете первые top_k слов и сэмплируете случайно с перенормированными вероятностями из оставшихся слов
        top_p -> после получения предсказания оставляете первые сколько-то слов, так, чтобы суммарная вероятность оставшихся слов была не больше top_p,
            после чего сэмплируете с перенормированными вероятностями из оставшихся слов
        иначе -> сэмплируете случайное слово с предсказанными вероятностями
    """
    assert top_p is None or top_k is None, "Don't use top_p and top_k at the same time"
    
    model.eval()
    
    channel_mean = np.array([0.485, 0.456, 0.406])
    channel_std = np.array([0.229, 0.224, 0.225])
    
    image_prepare_val = tr.Compose([
            tr.ToPILImage(),
            tr.Resize(size=224), # подгоним все изображения под один размер
            tr.CenterCrop((224, 224)),
            tr.ToTensor(),
            tr.Normalize(mean=channel_mean, std=channel_std),
        ])
    
    with torch.no_grad():
        img = image_prepare_val(image)[None,:,:,:].to(device)
        text = [tok_to_ind['<BOS>']]
        generated_size = 0
        while text[generated_size] != tok_to_ind['<EOS>'] and generated_size + 1 < max_seq_len:
            caption = torch.tensor(text, dtype=torch.int64) [None, None, :].repeat(1, 5, 1).to(device)
            pred = model(img, caption)
            probs = pred.detach().cpu().exp().numpy()[0, 0, -1, :]

            if top_k is not None:
                inds = np.argsort(-probs)[:top_k]
                pred_ind = np.random.choice(inds, p = probs[inds] / probs[inds].sum())

            elif top_p is not None:
                inds = np.argsort(-probs)
                while probs[inds].sum() > top_p and len(inds) > 1:
                    inds = inds[:-1]
                pred_ind = np.random.choice(inds, p = probs[inds] / probs[inds].sum())
            else:
                pred_ind = np.random.choice(np.arange(probs.shape[0]), p = probs/ probs.sum())
            text.append(pred_ind)
            generated_size += 1

            
        if text[generated_size] == tok_to_ind['<EOS>']:
            result_tokens = [ind_to_tok[ind] for ind in text][1:-1]
            result_text = ' '.join(result_tokens)
            
        return result_tokens, result_text
