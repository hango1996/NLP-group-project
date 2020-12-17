"""
Author      : Parichaya Chatterji (parichay@ucdavis.edu)
Description : XLNet for ECS 289G NLP Group Project
            : Referemce [https://www.kaggle.com/c/nlp-getting-started]
Date        : 16 Dec 2020
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import transformers
import nltk
import re
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
plt.style.use('seaborn')
from transformers import TFXLNetModel, XLNetTokenizer

class XLNet:
    """XLNet Class"""
    def __init__(self):
        print(tf.__version__)
        print(tf.config.list_physical_devices())

    def clean_text(self, text):
        clean = text
        reg = re.compile('\&amp')
        clean = clean.apply(lambda r: re.sub(reg, string=r, repl='&'))
        reg = re.compile('\\n')
        clean = clean.apply(lambda r: re.sub(reg, string=r, repl=' '))
        reg = re.compile('@[a-zA-Z0-9\_]+')
        clean = clean.apply(lambda r: re.sub(reg, string=r, repl='@'))
        reg = re.compile('https?\S+(?=\s|$)')
        clean = clean.apply(lambda r: re.sub(reg, string=r, repl='www'))
        return clean

    def create_xlnet(self, mname):
        word_inputs = tf.keras.Input(shape=(120,), name='word_inputs', dtype='int32')
        xlnet = TFXLNetModel.from_pretrained(mname)
        xlnet_encodings = xlnet(word_inputs)[0]
        doc_encoding = tf.squeeze(xlnet_encodings[:, -1:, :], axis=1)
        doc_encoding = tf.keras.layers.Dropout(.1)(doc_encoding)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='outputs')(doc_encoding)
        model = tf.keras.Model(inputs=[word_inputs], outputs=[outputs])
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=2e-5), loss='binary_crossentropy',
                      metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        return model

    def get_inputs(self, tweets, tokenizer, max_len=120):
        inps = [tokenizer.encode_plus(t, max_length=max_len, pad_to_max_length=True, add_special_tokens=True) for t in
                tweets]
        inp_tok = np.array([a['input_ids'] for a in inps])
        ids = np.array([a['attention_mask'] for a in inps])
        segments = np.array([a['token_type_ids'] for a in inps])
        return inp_tok, ids, segments

    def warmup(self, epoch, lr):
        return max(lr + 1e-6, 2e-5)

    def plot_metrics(self, pred, true_labels):
        acc = accuracy_score(true_labels, np.array(pred.flatten() >= .5, dtype='int'))
        fpr, tpr, thresholds = roc_curve(true_labels, pred)
        auc = roc_auc_score(true_labels, pred)
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.plot(fpr, tpr, color='red')
        ax.plot([0, 1], [0, 1], color='black', linestyle='--')
        ax.set_title(f"AUC: {auc}\nACC: {acc}");
        return fig



    def execute(self):
        print("XLNet")
        train_df = pd.read_csv("../data/tutorial/train.csv")
        test_df = pd.read_csv("../data/tutorial/test.csv")

        nltk.download('stopwords')

        print("::train_df:")
        print(train_df)

        print("::test_df:")
        print(test_df)

        print("::train_df[\"text\"]:")
        print(train_df["text"])

        print("::self.clean_text(train_df[\"text\"]):")
        print(self.clean_text(train_df["text"]))

        print("::self.clean_text(test_df[\"text\"]):")
        print(self.clean_text(test_df["text"]))

        train_df['clean'] = self.clean_text(train_df["text"])
        test_df['clean'] = self.clean_text(test_df["text"])

        print("::train_df:")
        print(train_df)

        print("::test_df:")
        print(test_df)

        xlnet_model = 'xlnet-large-cased'
        xlnet_model = 'xlnet-large-cased'
        xlnet_tokenizer = XLNetTokenizer.from_pretrained(xlnet_model)
        xlnet = self.create_xlnet(xlnet_model)

        print("::xlnet.summary(): [In [22]]")
        print(xlnet.summary())

        # Original context: "Clean and split the data"
        # Potentially important for score tabulation. Please revisit.
        tweets = train_df['clean']
        labels = train_df['target']
        X_train, X_test, y_train, y_test = train_test_split(tweets, labels, test_size=0.15, random_state=196)

        print("::X_test: [In [24]]")
        print(X_test)

        # Create the input data (tensors)
        inp_tok, ids, segments = self.get_inputs(X_train, xlnet_tokenizer)

        print("::inp_tok: [In [27]]")
        print(inp_tok)

        print("::ids: [In [28]]")
        print(ids)

        print("::segments: [In [29]]")
        print(segments)

        # Training

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4, min_delta=0.02,
                                             restore_best_weights=True),
            tf.keras.callbacks.LearningRateScheduler(self.warmup, verbose=0),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=1e-6, patience=2, verbose=0,
                                                 mode='auto', min_delta=0.001, cooldown=0, min_lr=1e-6)
        ]

        hist = xlnet.fit(x=inp_tok, y=y_train, epochs=15, batch_size=16, validation_split=.15, callbacks=callbacks)




if __name__ == '__main__':
    print("Okay, here we go.")
    XLNet().execute()

