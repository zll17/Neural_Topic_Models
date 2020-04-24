# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import numpy as np

from tqdm import tqdm

from mxnet import nd, autograd, gluon, io

from core import Compute
from utils import to_numpy, stack_numpy
# from diff_sample import normal
import os


def mmd_loss(x, y, ctx_model, t=0.1, kernel='diffusion'):
    '''
    computes the mmd loss with information diffusion kernel
    :param x: batch_size x latent dimension
    :param y:
    :param t:
    :return:
    '''
    eps = 1e-6
    n,d = x.shape
    if kernel == 'tv':
        sum_xx = nd.zeros(1, ctx=ctx_model)
        for i in range(n):
            for j in range(i+1, n):
                sum_xx = sum_xx + nd.norm(x[i] - x[j], ord=1)
        sum_xx = sum_xx / (n * (n-1))

        sum_yy = nd.zeros(1, ctx=ctx_model)
        for i in range(y.shape[0]):
            for j in range(i+1, y.shape[0]):
                sum_yy = sum_yy + nd.norm(y[i] - y[j], ord=1)
        sum_yy = sum_yy / (y.shape[0] * (y.shape[0]-1))

        sum_xy = nd.zeros(1, ctx=ctx_model)
        for i in range(n):
            for j in range(y.shape[0]):
                sum_xy = sum_xy + nd.norm(x[i] - y[j], ord=1)
        sum_yy = sum_yy / (n * y.shape[0])
    else:
        qx = nd.sqrt(nd.clip(x, eps, 1))
        qy = nd.sqrt(nd.clip(y, eps, 1))
        xx = nd.dot(qx, qx, transpose_b=True)
        yy = nd.dot(qy, qy, transpose_b=True)
        xy = nd.dot(qx, qy, transpose_b=True)

        def diffusion_kernel(a, tmpt, dim):
            # return (4 * np.pi * tmpt)**(-dim / 2) * nd.exp(- nd.square(nd.arccos(a)) / tmpt)
            return nd.exp(- nd.square(nd.arccos(a)) / tmpt)

        off_diag = 1 - nd.eye(n, ctx=ctx_model)
        k_xx = diffusion_kernel(nd.clip(xx, 0, 1-eps), t, d-1)
        k_yy = diffusion_kernel(nd.clip(yy, 0, 1-eps), t, d-1)
        k_xy = diffusion_kernel(nd.clip(xy, 0, 1-eps), t, d-1)
        sum_xx = (k_xx * off_diag).sum() / (n * (n-1))
        sum_yy = (k_yy * off_diag).sum() / (n * (n-1))
        sum_xy = 2 * k_xy.sum() / (n * n)
    return sum_xx + sum_yy - sum_xy


class Unsupervised(Compute):
    '''
    Class to manage training, testing, and
    retrieving outputs.
    '''
    def __init__(self, data, Enc, Dec,  Dis_y, args):
        '''
        Constructor.

        Args
        ----
        Returns
        -------
        Compute object
        '''
        super(Unsupervised, self).__init__(data, Enc, Dec, Dis_y, args)

    def unlabeled_train_op_mmd_combine(self, update_enc=True):
        '''
        Trains the MMD model
        '''
        batch_size = self.args['batch_size']
        model_ctx = self.model_ctx
        eps = 1e-10

        # Retrieve data
        docs = self.data.get_documents(key='train')

        y_true = np.random.dirichlet(np.ones(self.ndim_y) * self.args['dirich_alpha'], size=batch_size)
        y_true = nd.array(y_true, ctx=model_ctx)

        with autograd.record():
            ### reconstruction phase ###
            y_onehot_u = self.Enc(docs)
            y_onehot_u_softmax = nd.softmax(y_onehot_u)
            if self.args['latent_noise'] > 0:
                y_noise = np.random.dirichlet(np.ones(self.ndim_y) * self.args['dirich_alpha'], size=batch_size)
                y_noise = nd.array(y_noise, ctx=model_ctx)
                y_onehot_u_softmax = (1 - self.args['latent_noise']) * y_onehot_u_softmax + self.args['latent_noise'] * y_noise
            x_reconstruction_u = self.Dec(y_onehot_u_softmax)

            logits = nd.log_softmax(x_reconstruction_u)
            loss_reconstruction = nd.mean(nd.sum(- docs * logits, axis=1))
            loss_total = loss_reconstruction * self.args['recon_alpha']

            ### mmd phase ###
            if self.args['adverse']:
                y_fake = self.Enc(docs)
                y_fake = nd.softmax(y_fake)
                loss_mmd = mmd_loss(y_true, y_fake, ctx_model=model_ctx, t=self.args['kernel_alpha'])
                loss_total = loss_total + loss_mmd

            if self.args['l2_alpha'] > 0:
                loss_total = loss_total + self.args['l2_alpha'] * nd.mean(nd.sum(nd.square(y_onehot_u), axis=1))

            loss_total.backward()

        self.optimizer_enc.step(1)
        self.optimizer_dec.step(1)  # self.m.args['batch_size']

        latent_max = nd.zeros(self.args['ndim_y'], ctx=model_ctx)
        for max_ind in nd.argmax(y_onehot_u, axis=1):
            latent_max[max_ind] += 1.0
        latent_max /= batch_size
        latent_entropy = nd.mean(nd.sum(- y_onehot_u_softmax * nd.log(y_onehot_u_softmax + eps), axis=1))
        latent_v = nd.mean(y_onehot_u_softmax, axis=0)
        dirich_entropy = nd.mean(nd.sum(- y_true * nd.log(y_true + eps), axis=1))

        if self.args['adverse']:
            loss_mmd_return = loss_mmd.asscalar()
        else:
            loss_mmd_return = 0.0
        return nd.mean(loss_reconstruction).asscalar(), loss_mmd_return, latent_max.asnumpy(), latent_entropy.asscalar(), latent_v.asnumpy(), dirich_entropy.asscalar()


    def retrain_enc(self, l2_alpha=0.1):
        docs = self.data.get_documents(key='train')
        with autograd.record():
            ### reconstruction phase ###
            y_onehot_u = self.Enc(docs)
            y_onehot_u_softmax = nd.softmax(y_onehot_u)
            x_reconstruction_u = self.Dec(y_onehot_u_softmax)

            logits = nd.log_softmax(x_reconstruction_u)
            loss_reconstruction = nd.mean(nd.sum(- docs * logits, axis=1))
            loss_reconstruction = loss_reconstruction + l2_alpha * nd.mean(nd.norm(y_onehot_u, ord=1, axis=1))
            loss_reconstruction.backward()

        self.optimizer_enc.step(1)
        return loss_reconstruction.asscalar()


    def unlabeled_train_op_adv_combine_add(self, update_enc=True):
        '''
        Trains the GAN model
        '''
        batch_size = self.args['batch_size']
        model_ctx = self.model_ctx
        eps = 1e-10
        ##########################
        ### unsupervised phase ###
        ##########################
        # Retrieve data
        docs = self.data.get_documents(key='train')

        class_true = nd.zeros(batch_size, dtype='int32', ctx=model_ctx)
        class_fake = nd.ones(batch_size, dtype='int32', ctx=model_ctx)
        loss_reconstruction = nd.zeros((1,), ctx=model_ctx)

        ### adversarial phase ###
        discriminator_z_confidence_true = nd.zeros(shape=(1,), ctx=model_ctx)
        discriminator_z_confidence_fake = nd.zeros(shape=(1,), ctx=model_ctx)
        discriminator_y_confidence_true = nd.zeros(shape=(1,), ctx=model_ctx)
        discriminator_y_confidence_fake = nd.zeros(shape=(1,), ctx=model_ctx)
        loss_discriminator = nd.zeros(shape=(1,), ctx=model_ctx)
        dirich_entropy = nd.zeros(shape=(1,), ctx=model_ctx)

        ### generator phase ###
        loss_generator = nd.zeros(shape=(1,), ctx=model_ctx)

        ### reconstruction phase ###
        with autograd.record():
            y_u = self.Enc(docs)
            y_onehot_u_softmax = nd.softmax(y_u)
            x_reconstruction_u = self.Dec(y_onehot_u_softmax)

            logits = nd.log_softmax(x_reconstruction_u)
            loss_reconstruction = nd.sum(- docs * logits, axis=1)
            loss_total = loss_reconstruction * self.args['recon_alpha']

            if self.args['adverse']: #and np.random.rand()<0.8:
                y_true = np.random.dirichlet(np.ones(self.ndim_y) * self.args['dirich_alpha'], size=batch_size)
                y_true = nd.array(y_true, ctx=model_ctx)
                dy_true = self.Dis_y(y_true)
                dy_fake = self.Dis_y(y_onehot_u_softmax)
                discriminator_y_confidence_true = nd.mean(nd.softmax(dy_true)[:, 0])
                discriminator_y_confidence_fake = nd.mean(nd.softmax(dy_fake)[:, 1])
                softmaxCEL = gluon.loss.SoftmaxCrossEntropyLoss()
                loss_discriminator = softmaxCEL(dy_true, class_true) + \
                                       softmaxCEL(dy_fake, class_fake)
                loss_generator = softmaxCEL(dy_fake, class_true)
                loss_total = loss_total + loss_discriminator + loss_generator
                dirich_entropy = nd.mean(nd.sum(- y_true * nd.log(y_true + eps), axis=1))

        loss_total.backward()

        self.optimizer_enc.step(batch_size)
        self.optimizer_dec.step(batch_size)
        self.optimizer_dis_y.step(batch_size)

        latent_max = nd.zeros(self.args['ndim_y'], ctx=model_ctx)
        for max_ind in nd.argmax(y_onehot_u_softmax, axis=1):
            latent_max[max_ind] += 1.0
        latent_max /= batch_size
        latent_entropy = nd.mean(nd.sum(- y_onehot_u_softmax * nd.log(y_onehot_u_softmax + eps), axis=1))
        latent_v = nd.mean(y_onehot_u_softmax, axis=0)

        return nd.mean(loss_discriminator).asscalar(), nd.mean(loss_generator).asscalar(), nd.mean(loss_reconstruction).asscalar(), \
               nd.mean(discriminator_z_confidence_true).asscalar(), nd.mean(discriminator_z_confidence_fake).asscalar(), \
               nd.mean(discriminator_y_confidence_true).asscalar(), nd.mean(discriminator_y_confidence_fake).asscalar(), \
               latent_max.asnumpy(), latent_entropy.asscalar(), latent_v.asnumpy(), dirich_entropy.asscalar()


    def test_synthetic_op(self):
        batch_size = self.args['batch_size']
        dataset = 'train'
        num_samps = self.data.data[dataset].shape[0]
        batches = int(np.ceil(num_samps / batch_size))
        batch_iter = range(batches)
        enc_out = nd.zeros(shape=(batches * batch_size, self.ndim_y))
        for batch in batch_iter:
            # 1. Retrieve data
            if self.args['data_source'] == 'Ian':
                docs = self.data.get_documents(key=dataset)
            # 2. Compute loss
            y_onehot_u = self.Enc(docs)
            y_onehot_softmax = nd.softmax(y_onehot_u)
            enc_out[batch*batch_size:(batch+1)*batch_size, :] = y_onehot_softmax

        return enc_out

    def test_op(self, num_samples=None, num_epochs=None, reset=True, dataset='test'):
        '''
        Evaluates the model using num_samples.

        Args
        ----
        num_samples: integer, default None
          The number of samples to evaluate on. This is converted to
          evaluating on (num_samples // batch_size) minibatches.
        num_epochs: integer, default None
          The number of epochs to evaluate on. This used if num_samples
          is not specified. If neither is specified, defaults to 1 epoch.
        reset: bool, default True
          Whether to reset the test data index to 0 before iterating
          through and evaluating on minibatches.
        dataset: string, default 'test':
          Which dataset to evaluate on: 'valid' or 'test'.

        Returns
        -------
        Loss_u: float
          The loss on the unlabeled data.
        Loss_l: float
          The loss on the labeled data.
        Eval_u: list of floats
          A list of evaluation metrics on the unlabeled data.
        Eval_l: list of floats
          A list of evaluation metrics on the labeled data.
        '''
        batch_size = self.args['batch_size']
        model_ctx = self.model_ctx

        if num_samples is None and num_epochs is None:
            # assume full dataset evaluation
            num_epochs = 1

        if reset:
            # Reset Data to Index Zero
            if self.data.data[dataset] is not None:
                self.data.force_reset_data(dataset)
            if self.data.data[dataset + '_with_labels'] is not None:
                self.data.force_reset_data(dataset+'_with_labels')

        # Unlabeled Data
        u_loss = 'NA'
        u_eval = []
        if self.data.data[dataset] is not None:
            u_loss = 0
            if num_samples is None:
                num_samps = self.data.data[dataset].shape[0] * num_epochs
            else:
                num_samps = num_samples
            batches = int(np.ceil(num_samps / self.args['batch_size']))
            batch_iter = range(batches)
            if batches > 1: batch_iter = tqdm(batch_iter, desc='unlabeled')
            for batch in batch_iter:
                # 1. Retrieve data
                docs = self.data.get_documents(key=dataset)

                # 2. Compute loss
                y_u = self.Enc(docs)
                y_onehot_u_softmax = nd.softmax(y_u)
                x_reconstruction_u = self.Dec(y_onehot_u_softmax)

                logits = nd.log_softmax(x_reconstruction_u)
                loss_recon_unlabel = nd.sum(- docs * logits, axis=1)

                # 3. Convert to numpy
                u_loss += nd.mean(loss_recon_unlabel).asscalar()
            u_loss /= batches

        # Labeled Data
        l_loss = 0.0
        l_acc = 0.0
        if self.data.data[dataset+'_with_labels'] is not None:
            l_loss = 0
            if num_samples is None:
                num_samps = self.data.data[dataset+'_with_labels'].shape[0] * num_epochs
            else:
                num_samps = num_samples
            batches = int(np.ceil(num_samps / self.args['batch_size']))
            batch_iter = range(batches)
            if batches > 1: batch_iter = tqdm(batch_iter, desc='labeled')
            softmaxCEL = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)
            for batch in batch_iter:
                # 1. Retrieve data
                labeled_docs, labels = self.data.get_documents(key=dataset+'_with_labels', split_on=self.data.data_dim)
                # 2. Compute loss
                y_u = self.Enc(docs)
                y_onehot_u_softmax = nd.softmax(y_u)
                class_pred = nd.argmax(y_onehot_u_softmax, axis=1)
                l_a = labels[list(range(labels.shape[0])), class_pred]
                l_acc += nd.mean(l_a).asscalar()
                labels = labels / nd.sum(labels, axis=1, keepdims=True)
                l_l = softmaxCEL(y_onehot_u_softmax, labels)

                # 3. Convert to numpy
                l_loss += nd.mean(l_l).asscalar()
            l_loss /= batches
            l_acc /= batches

        return u_loss, l_loss, l_acc


    def save_latent(self, saveto):
        before_softmax = True
        try:
            if type(self.data.data['train']) is np.ndarray:
                dataset_train = gluon.data.dataset.ArrayDataset(self.data.data['train'])
                train_data = gluon.data.DataLoader(dataset_train, self.args['batch_size'], shuffle=False, last_batch='discard')

                dataset_val = gluon.data.dataset.ArrayDataset(self.data.data['valid'])
                val_data = gluon.data.DataLoader(dataset_val, self.args['batch_size'], shuffle=False, last_batch='discard')

                dataset_test = gluon.data.dataset.ArrayDataset(self.data.data['test'])
                test_data = gluon.data.DataLoader(dataset_test, self.args['batch_size'], shuffle=False, last_batch='discard')
            else:
                train_data = io.NDArrayIter(data={'data': self.data.data['train']}, batch_size=self.args['batch_size'],
                                            shuffle=False, last_batch_handle='discard')
                val_data = io.NDArrayIter(data={'data': self.data.data['valid']}, batch_size=self.args['batch_size'],
                                            shuffle=False, last_batch_handle='discard')
                test_data = io.NDArrayIter(data={'data': self.data.data['test']}, batch_size=self.args['batch_size'],
                                            shuffle=False, last_batch_handle='discard')
        except:
            print("Loading error during save_latent. Probably caused by not having validation or test set!")
            return

        train_output = np.zeros((self.data.data['train'].shape[0], self.ndim_y))
        # train_label_output = np.zeros(self.data.data['train'].shape[0])
        # for i, (data, label) in enumerate(train_data):
        for i, data in enumerate(train_data):
            if type(data) is io.DataBatch:
                data = data.data[0].as_in_context(self.model_ctx)
            else:
                data = data.as_in_context(self.model_ctx)
            if before_softmax:
                output = self.Enc(data)
            else:
                output = nd.softmax(self.Enc(data))
            train_output[i*self.args['batch_size']:(i+1)*self.args['batch_size']] = output.asnumpy()
            # train_label_output[i*self.args['batch_size']:(i+1)*self.args['batch_size']] = label.asnumpy()
        train_output = np.delete(train_output, np.s_[(i+1)*self.args['batch_size']:], 0)
        # train_label_output = np.delete(train_label_output, np.s_[(i+1)*self.args['batch_size']:])
        np.save(os.path.join(saveto, self.args['domain']+'train_latent.npy'), train_output)
        # np.save(os.path.join(saveto, self.args['domain']+'train_latent_label.npy'), train_label_output)

        val_output = np.zeros((self.data.data['valid'].shape[0], self.ndim_y))
        # train_label_output = np.zeros(self.data.data['train'].shape[0])
        # for i, (data, label) in enumerate(train_data):
        for i, data in enumerate(val_data):
            if type(data) is io.DataBatch:
                data = data.data[0].as_in_context(self.model_ctx)
            else:
                data = data.as_in_context(self.model_ctx)
            if before_softmax:
                output = self.Enc(data)
            else:
                output = nd.softmax(self.Enc(data))
            val_output[i*self.args['batch_size']:(i+1)*self.args['batch_size']] = output.asnumpy()
            # train_label_output[i*self.args['batch_size']:(i+1)*self.args['batch_size']] = label.asnumpy()
        val_output = np.delete(val_output, np.s_[(i+1)*self.args['batch_size']:], 0)
        # train_label_output = np.delete(train_label_output, np.s_[(i+1)*self.args['batch_size']:])
        np.save(os.path.join(saveto, self.args['domain']+'val_latent.npy'), val_output)
        # np.save(os.path.join(saveto, self.args['domain']+'train_latent_label.npy'), train_label_output)

        test_output = np.zeros((self.data.data['test'].shape[0], self.ndim_y))
        # test_label_output = np.zeros(self.data.data['test'].shape[0])
        # for i, (data, label) in enumerate(test_data):
        for i, data in enumerate(test_data):
            if type(data) is io.DataBatch:
                data = data.data[0].as_in_context(self.model_ctx)
            else:
                data = data.as_in_context(self.model_ctx)
            if before_softmax:
                output = self.Enc(data)
            else:
                output = nd.softmax(self.Enc(data))
            test_output[i*self.args['batch_size']:(i+1)*self.args['batch_size']] = output.asnumpy()
            # test_label_output[i*self.args['batch_size']:(i+1)*self.args['batch_size']] = label.asnumpy()
        test_output = np.delete(test_output, np.s_[(i+1)*self.args['batch_size']:], 0)
        # test_label_output = np.delete(test_label_output, np.s_[(i+1)*self.args['batch_size']:])
        np.save(os.path.join(saveto, self.args['domain']+'test_latent.npy'), test_output)
        # np.save(os.path.join(saveto, self.args['domain']+'test_latent_label.npy'), test_label_output)