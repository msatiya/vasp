import os
from datetime import datetime  

from utils import *
import tensorflow as tf
import tensorflow_addons as tfa
tf.config.experimental.set_memory_growth
import gc

gc.collect()

set_seed(42)

print(tf.config.list_physical_devices())

ds_name = 'ml-20m'
# ml-20m: len(self.pr_val) = 13668
# divisors: {1,2,3,4,6,12,17,34,51,67,68,102,134,201,204,268,402,804,1139,2278,3417,4556,6834,13668}
chunk_val = 1139
# ml-20m: len(self.pr_test) = 13667
# divisors: {1,79,173,13667}
chunk_test = 173

# ds_name = 'netflix'
# # netflix: len(self.pr_val) = 46344
# # divisors: {1,2,3,4,6,8,12,24,1931,3862,5793,7724,11586,15448,23172,46344}
# chunk_val= 1931
# # netflix: len(self.pr_test) = 46343
# # divisors: {1,11,121,383,4213,46343}
# chunk_test= 383

# ds_name = 'steam-200k'
# # steam: len(self.pr) = 165
# # divisors: {1,3,5,11,15,33,55,165}
# chunk_val = 165
# # steam: len(self.pr) = 163
# # divisors: {1,163}
# chunktest = 163

data_path = os.path.join('/home/ubuntu/vasp/Datasets/', ds_name, 'preprocessed_vasp/')

str_date_time = datetime.fromtimestamp(datetime.now().timestamp()).strftime("%d-%m-%Y_%H-%M-%S")
model_name = str('VASP_'+ds_name+'_'+str_date_time)

class DiagonalToZero(tf.keras.constraints.Constraint):
    def __call__(self, w):
        """Set diagonal to zero"""
        q = tf.linalg.set_diag(w, tf.zeros(w.shape[0:-1]), name=None)
        return q

class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a basket."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), stddev=1.)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VASP(Model):
    class Model(tf.keras.Model):
        def __init__(self, num_words, latent=1024, hidden=1024, items_sampling=1.):
            """
            num_words             nr of items in dataset (size of tokenizer)
            latent                size of latent space
            hidden                size of hidden layers
            items_sampling        Large items datatsets can be very gpu memory consuming in EASE layer.
                                  This coefficient reduces number of ease parametrs by taking only
                                  fraction of items sorted by popularity as input for model.
                                  Note: This coef should be somewhere around coverage@100 achieved by full
                                  size model.
                                  For ML20M this coef should be between 0.4888 (coverage@100 for full model)
                                  and 1.0
                                  For Netflix this coef should be between 0.7055 (coverage@100 for full
                                  model) and 1.0
            """
            super(VASP.Model, self).__init__()

            self.sampled_items = int(num_words * items_sampling)

            assert self.sampled_items > 0
            assert self.sampled_items <= num_words

            self.s = self.sampled_items < num_words

            # ************* ENCODER ***********************
            self.encoder1 = tf.keras.layers.Dense(hidden)
            self.ln1 = tf.keras.layers.LayerNormalization()
            self.encoder2 = tf.keras.layers.Dense(hidden)
            self.ln2 = tf.keras.layers.LayerNormalization()
            self.encoder3 = tf.keras.layers.Dense(hidden)
            self.ln3 = tf.keras.layers.LayerNormalization()
            self.encoder4 = tf.keras.layers.Dense(hidden)
            self.ln4 = tf.keras.layers.LayerNormalization()
            self.encoder5 = tf.keras.layers.Dense(hidden)
            self.ln5 = tf.keras.layers.LayerNormalization()
            self.encoder6 = tf.keras.layers.Dense(hidden)
            self.ln6 = tf.keras.layers.LayerNormalization()
            self.encoder7 = tf.keras.layers.Dense(hidden)
            self.ln7 = tf.keras.layers.LayerNormalization()

            # ************* SAMPLING **********************
            self.dense_mean = tf.keras.layers.Dense(latent,
                                                    name="Mean")
            self.dense_log_var = tf.keras.layers.Dense(latent,
                                                       name="log_var")

            self.sampling = Sampling(name='Sampler')

            # ************* DECODER ***********************
            self.decoder1 = tf.keras.layers.Dense(hidden)
            self.dln1 = tf.keras.layers.LayerNormalization()
            self.decoder2 = tf.keras.layers.Dense(hidden)
            self.dln2 = tf.keras.layers.LayerNormalization()
            self.decoder3 = tf.keras.layers.Dense(hidden)
            self.dln3 = tf.keras.layers.LayerNormalization()
            self.decoder4 = tf.keras.layers.Dense(hidden)
            self.dln4 = tf.keras.layers.LayerNormalization()
            self.decoder5 = tf.keras.layers.Dense(hidden)
            self.dln5 = tf.keras.layers.LayerNormalization()

            self.decoder_resnet = tf.keras.layers.Dense(self.sampled_items,
                                                        activation='sigmoid',
                                                        name="DecoderR")
            self.decoder_latent = tf.keras.layers.Dense(self.sampled_items,
                                                        activation='sigmoid',
                                                        name="DecoderL")

            # ************* PARALLEL SHALLOW PATH *********

            self.ease = tf.keras.layers.Dense(
                self.sampled_items,
                activation='sigmoid',
                use_bias=False,
                kernel_constraint=DiagonalToZero(),  # critical to prevent learning simple identity
            )

        def call(self, x, training=None):
            sampling = self.s
            if sampling:
                sampled_x = x[:, :self.sampled_items]
                non_sampled = x[:, self.sampled_items:] * 0.
            else:
                sampled_x = x

            z_mean, z_log_var, z = self.encode(sampled_x)
            if training:
                d = self.decode(z)
                # Add KL divergence regularization loss.
                kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
                kl_loss = tf.reduce_mean(kl_loss)
                kl_loss *= -0.5
                self.add_loss(kl_loss)
                self.add_metric(kl_loss, name="kl_div")
            else:
                d = self.decode(z_mean)

            if sampling:
                d = tf.concat([d, non_sampled], axis=-1)

            ease = self.ease(sampled_x)

            if sampling:
                ease = tf.concat([ease, non_sampled], axis=-1)

            return d * ease

        def decode(self, x):
            e0 = x
            e1 = self.dln1(tf.keras.activations.swish(self.decoder1(e0)))
            e2 = self.dln2(tf.keras.activations.swish(self.decoder2(e1) + e1))
            e3 = self.dln3(tf.keras.activations.swish(self.decoder3(e2) + e1 + e2))
            e4 = self.dln4(tf.keras.activations.swish(self.decoder4(e3) + e1 + e2 + e3))
            e5 = self.dln5(tf.keras.activations.swish(self.decoder5(e4) + e1 + e2 + e3 + e4))

            dr = self.decoder_resnet(e5)
            dl = self.decoder_latent(x)

            return dr * dl

        def encode(self, x):
            e0 = x
            e1 = self.ln1(tf.keras.activations.swish(self.encoder1(e0)))
            e2 = self.ln2(tf.keras.activations.swish(self.encoder2(e1) + e1))
            e3 = self.ln3(tf.keras.activations.swish(self.encoder3(e2) + e1 + e2))
            e4 = self.ln4(tf.keras.activations.swish(self.encoder4(e3) + e1 + e2 + e3))
            e5 = self.ln5(tf.keras.activations.swish(self.encoder5(e4) + e1 + e2 + e3 + e4))
            e6 = self.ln6(tf.keras.activations.swish(self.encoder6(e5) + e1 + e2 + e3 + e4 + e5))
            e7 = self.ln7(tf.keras.activations.swish(self.encoder7(e6) + e1 + e2 + e3 + e4 + e5 + e6))

            z_mean = self.dense_mean(e7)
            z_log_var = self.dense_log_var(e7)
            z = self.sampling((z_mean, z_log_var))

            return z_mean, z_log_var, z

    def create_model(self, latent=2048, hidden=4096, ease_items_sampling=1., summary=False):
        self.model = VASP.Model(self.dataset.num_words, latent, hidden, ease_items_sampling)
        self.model(self.split.train_gen[0][0])
        if summary:
            self.model.summary()
        self.mc = MetricsCallback(self)

    def compile_model(self, lr=0.00002, fl_alpha=0.25, fl_gamma=2.0):
        """
        lr         learning rate of Nadam optimizer
        fl_alpha   alpha parameter of focal crossentropy
        fl_gamma   gamma parameter of focal crossentropy
        """
        self.model.compile(
            optimizer=tf.keras.optimizers.Nadam(lr),
            loss=lambda x, y: tfa.losses.sigmoid_focal_crossentropy(x, y, alpha=fl_alpha, gamma=fl_gamma),
            metrics=['mse', cosine_loss]
        )

    def train_model(self, epochs=150):
        log_dir = "tb_logs/" + model_name
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        self.model.fit(
            self.split.train_gen,
            validation_data=self.split.validation_gen,
            epochs=epochs,
            callbacks=[self.mc, tensorboard_callback]
        )

dataset = Data(d=data_path, pruning='u5')
dataset.splits = []
dataset.create_splits(1, 10000, shuffle=False, n_fold=False, generators=False, batch_size=1024, chunk=chunk_val)
dataset.split.train_users = pd.read_json(os.path.join(data_path, "train_users.json")).userid.apply(str).to_frame()
dataset.split.validation_users = pd.read_json(os.path.join(data_path, "val_users.json")).userid.apply(str).to_frame()
dataset.split.test_users = pd.read_json(os.path.join(data_path, "test_users.json")).userid.apply(str).to_frame()
dataset.split.generators()

m = VASP(dataset.split, name=model_name)
gc.collect()

m.create_model(latent=2048, hidden=4096, ease_items_sampling=0.33)
#m.model.summary()
print("=" * 80)
print("Train for 50 epochs with lr 0.00005")
m.compile_model(lr=0.00005, fl_alpha=0.25, fl_gamma=2.0)
m.train_model(50)
gc.collect()

print("=" * 80)
print("Than train for 20 epochs with lr 0.00001")
m.compile_model(lr=0.00001, fl_alpha=0.25, fl_gamma=2.0)
m.train_model(20)
gc.collect()

print("=" * 80)
print("Than train for 20 epochs with lr 0.000001")
m.compile_model(lr=0.000001, fl_alpha=0.25, fl_gamma=2.0)
m.train_model(20)

print(m.mc.get_history_df())

# 5-fold evaluation on the test set
test_r20s = []
test_r50s = []
test_n100s = []
test_custom_n100s = []

for fold in range(1,6):
    ev=Evaluator(m.split, method=str(fold)+'_20', chunk=chunk_test)
    ev.update(m.model)

    test_n100s.append(ev.get_ncdg(100))
    test_custom_n100s.append(ev.custom_ndcg_at_rank_k(100))
    test_r20s.append(ev.get_recall(20))
    test_r50s.append(ev.get_recall(50))

print("TEST SET (MEAN)")
print("5-fold mean NCDG@100", round(sum(test_n100s) / len(test_n100s),3))
print("5-fold mean customNCDG@100", round(sum(test_custom_n100s) / len(test_custom_n100s),3))
print("5-fold mean Recall@20", round(sum(test_r20s) / len(test_r20s),3))
print("5-fold mean Recall@50", round(sum(test_r50s) / len(test_r50s),3))