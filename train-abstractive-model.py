import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import dataloader
from dataset import AbstrativeDataset
from model.kogpt2 import AbstractiveKoGPT2

if __name__ == '__main__':
    checkpoint_path ="./checkpoint"
    save_ckpt_path = f"{checkpoint_path}/kogpt2-abstractive.pth"

    n_epoch = 5         # Num of Epoch
    batch_size = 2      # 배치 사이즈
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_step = 100 # 학습 저장 주기
    learning_rate = 5e-5  # Learning Rate

    dataset= AbstrativeDataset(device=device)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AbstractiveKoGPT2()
    model.to(device)

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_losses = []
    losses =[]

    if os.path.isfile(save_ckpt_path):
        checkpoint = torch.load(save_ckpt_path, map_location=device)
        pre_epoch = checkpoint['epoch']
        pre_loss = checkpoint['loss']
        total_losses = checkpoint['losses']

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"load pretrain from: {save_ckpt_path}, epoch={pre_epoch}, loss={pre_loss}")

    for epoch in range(n_epoch):
        count = 0
        with tqdm(total=len(train_loader), desc=f"Train({epoch})") as pbar:
            for i, data in enumerate(train_loader):
                optimizer.zero_grad()

                outputs = model(data, labels=data)
                _, logits = outputs[:2]

                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = data[..., 1:].contiguous()

                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

                if (count > 0 and count % save_step == 0) or (len(data) < batch_size):
                    torch.save({
                        'epoch': epoch,
                        'train_no': count,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        'losses': losses
                    }, save_ckpt_path)
                count += 1
                pbar.update(1)
                pbar.set_postfix_str(f"Loss: {loss.item():.3f} ({np.mean(losses):.3f})")

        total_losses.append(np.mean(losses))

    # data
    data = {
        "loss": total_losses
    }
    df = pd.DataFrame(data)
    display(df)

    # graph
    plt.figure(figsize=[12, 4])
    plt.plot(losses, label="loss")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()