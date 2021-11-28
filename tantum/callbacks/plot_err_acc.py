import numpy as np
import os
from tantum import enums
from tantum.callbacks import Callback
import matplotlib.pyplot as plt
import IPython.display as display
import time


class PlotErrAcc(Callback):
    def __init__(self, attention_show=False) -> None:
        self.attention_show = attention_show

    def plot_without_attention(self, tr_err, ts_err, tr_acc, ts_acc):
        plt.clf()
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        axs[0].plot(tr_err, label='tr_err')
        axs[0].plot(ts_err, label='ts_err')
        axs[0].legend()
        axs[1].plot(tr_acc, label='tr_acc')
        axs[1].plot(ts_acc, label='ts_acc')
        axs[1].legend()
        axs[2].axis('off')
        axs[3].axis('off')
        display.clear_output(wait=True)
        display.display(plt.gcf())
        time.sleep(0.01)

    def plot_with_attention(self, tr_err, ts_err, tr_acc, ts_acc, img, att_out, no_images=6):
        plt.clf()
        fig, axs = plt.subplots(1+no_images, 4, figsize=(20, (no_images+1)*5))
        axs[0, 0].plot(tr_err, label='tr_err')
        axs[0, 0].plot(ts_err, label='ts_err')
        axs[0, 0].legend()
        axs[0, 1].plot(tr_acc, label='tr_acc')
        axs[0, 1].plot(ts_acc, label='ts_acc')
        axs[0, 1].legend()
        axs[0, 2].axis('off')
        axs[0, 3].axis('off')
        for img_no in range(6):
            im = img[img_no].cpu().detach().numpy()
            axs[img_no+1, 0].imshow(im)
            for i in range(3):
                att_out_img = att_out[img_no, i+1].cpu().detach().numpy()
                axs[img_no+1, i+1].imshow(att_out_img)
        display.clear_output(wait=True)
        display.display(plt.gcf())
        time.sleep(0.01)
    
    def on_epoch_end(self, model, **kwargs):
        print("Plot err att")
        tr_err = model.history_mertrics['train']['err']
        ts_err = model.history_mertrics['valid']['err']

        tr_acc = model.history_mertrics['train']['acc']
        ts_arr = model.history_mertrics['valid']['acc']

        if not self.attention_show:
            self.plot_without_attention(tr_err, ts_err, tr_acc, ts_arr)
        elif self.attention_show:
            self.plot_with_attention(tr_err, ts_err, tr_acc, ts_arr, model.input, model.attention)