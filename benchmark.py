import numpy as np
import torch
from main import load_model, backprop, load_dataset, convert_to_windows, pot_eval, hit_att, ndcg
import pandas as pd
from src.utils import *
import time

train_loader, test_loader, labels = load_dataset(args.dataset)

model, optimizer, scheduler, epoch, accuracy_list = load_model(labels.shape[1])

trainD, testD = next(iter(train_loader)), next(iter(test_loader))
trainO, testO = trainD, testD
trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)

torch.zero_grad = True
model.eval()

print(f'{color.HEADER}test model latency on {args.dataset}{color.ENDC}')
times = []
for rep in range(10):
    start_time = time.perf_counter()
    loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)

    ### evaluation
    df = pd.DataFrame()
    lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
    thresholds = []
    preds = []  
    for i in range(loss.shape[1]):
        lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
        result, pred = pot_eval(lt, l, ls)
        preds.append(pred)
        thresholds.append(result['threshold'])
        df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)  # 更新為 pd.concat，append 已棄用

    # Calculate final loss and labels
    lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
    labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
    result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
    result.update(hit_att(loss, labels))
    result.update(ndcg(loss, labels))

    end_time = time.perf_counter()
    latency = end_time - start_time
    times.append(latency)

avg_latency = np.mean(times)
min_latency = np.min(times)
max_latency = np.max(times)
print(f"\n=== Latency Statistics ===")
print(f"Average Latency: {avg_latency:.4f} seconds")
print(f"Minimum Latency: {min_latency:.4f} seconds")
print(f"Maximum Latency: {max_latency:.4f} seconds")

