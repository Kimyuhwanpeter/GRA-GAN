# -*- coding: utf-8 -*-
from absl import flags, app
#from Gender_age_model_3 import *
from Gender_age_model_3_ver3 import *
from random import random
from keras_radam.training import RAdamOptimizer

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import datetime
import sys
import os

flags.DEFINE_bool("train", True, "True or False")

flags.DEFINE_integer("img_size", 256, "After crop (train)")

flags.DEFINE_integer("load_size", 266, "Before crop (train)")

flags.DEFINE_string("A_txt_path", "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/race_age_gender_generation/UTK_AAF_16_63/first_fold/AAF-M_UTK-F_16_39_40_63/train/male_16_39_train.txt", "A text path")

flags.DEFINE_string("A_img_path", "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/AAF/crop/All/first_fold/train/aug/male_16_39/", "A image path")

flags.DEFINE_integer("A_int", 16, "minimum A age")

flags.DEFINE_string("A_buf_age_txt", "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/race_age_gender_generation/Morph_AFAD_16_63/first_fold/AFAD-M_Morph-F_16_39_40_63/train/male_16_39_ageBuf.txt", "Other(about A) image's distribution")

flags.DEFINE_string("A_buf_age_img", "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/AAF/crop/All/first_fold/train/aug/male_16_39/", "Other(about A) image's distribution")

flags.DEFINE_string("B_txt_path", "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/race_age_gender_generation/UTK_AAF_16_63/first_fold/AAF-M_UTK-F_16_39_40_63/train/female_40_63_train.txt", "B text path")

flags.DEFINE_string("B_img_path", "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/UTK/All/first_fold/train/aug/female_40_63/", "B image path")

flags.DEFINE_integer("B_int", 40, "minimum B age")

flags.DEFINE_string("B_buf_age_txt", "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/race_age_gender_generation/Morph_AFAD_16_63/first_fold/AFAD-M_Morph-F_16_39_40_63/train/female_40_63_ageBuf.txt", "Other(about B) image's distribution")

flags.DEFINE_string("B_buf_age_img", "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/UTK/All/first_fold/train/aug/female_40_63/", "Other(about B) image's distribution")

flags.DEFINE_integer("epochs", 200, "Total epochs")

flags.DEFINE_integer("ep_decay", 100, "Epochs decay")

flags.DEFINE_integer("batch_size", 1, "Training batch size")

flags.DEFINE_float("lr", 0.0002, "Training learning rate")

flags.DEFINE_integer("n_classes", 24, "Number of classes")

flags.DEFINE_float("L1_lambda", 10.0, "Age loss factor")

flags.DEFINE_float("L1_lambda2", 5.0, "Gender loss factor")

flags.DEFINE_string("save_checkpoint", "C:/Users/Yuhwan/Pictures/sample", "Saving checkpoint path")

flags.DEFINE_string("sample_img_path", "C:/Users/Yuhwan/Pictures/sample", "Saving training sample images")

flags.DEFINE_string("graphs", "C:/Users/Yuhwan/Downloads/", "Training graphs path")

flags.DEFINE_bool("pre_checkpoint", False, "Want to restore?")

flags.DEFINE_string("pre_checkpoint_path", "C:/Users/Yuhwan/Downloads/0", "Saved checkpoint path")

##################################################################################################################
flags.DEFINE_string("A_test_txt", "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/race_age_gender_generation/Morph_AFAD_16_63/first_fold/AFAD-F_UTK-M_16_39_40_63/test/female_16_39_test.txt", "")

flags.DEFINE_string("A_test_img", "D:/[1]DB/[1]second_paper_DB/AFAD_16_69_DB/backup/fix_AFAD/", "")

flags.DEFINE_string("A_test_output", "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/race_age_gender_generation/Morph_AFAD_16_63/first_fold/AFAD-F_UTK-M_16_39_40_63/test_images/AFAD-M_40_63", "")

flags.DEFINE_string("B_test_txt", "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/race_age_gender_generation/Morph_AFAD_16_63/first_fold/AFAD-F_UTK-M_16_39_40_63/test/male_40_63_test.txt", "")

flags.DEFINE_string("B_test_img", "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/UTK/All/first_fold/test/aug/male_40_63/", "")

flags.DEFINE_string("B_test_output", "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/race_age_gender_generation/Morph_AFAD_16_63/first_fold/AFAD-F_UTK-M_16_39_40_63/test_images/UTK-F_16_39", "")

flags.DEFINE_string("dir", "B2A", "")
##################################################################################################################

FLAGS = flags.FLAGS
FLAGS(sys.argv)
# 이것은 B2A가 잘나왔던 Gender_age_model_3의 generator모델과 A2B가 잘나오는 Gender_age_model_3_ver2의 generator의 모델을
# 각각 사용한 방법이다!
class LinearDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    # if `step` < `step_decay`: use fixed learning rate
    # else: linearly decay the learning rate to zero

    def __init__(self, initial_learning_rate, total_steps, step_decay):
        super(LinearDecay, self).__init__()
        self._initial_learning_rate = initial_learning_rate
        self._steps = total_steps
        self._step_decay = step_decay
        self.current_learning_rate = tf.Variable(initial_value=initial_learning_rate, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        self.current_learning_rate.assign(tf.cond(
            step >= self._step_decay,
            true_fn=lambda: self._initial_learning_rate * (1 - 1 / (self._steps - self._step_decay) * (step - self._step_decay)),
            false_fn=lambda: self._initial_learning_rate
        ))
        return self.current_learning_rate
#len_dataset = np.loadtxt(FLAGS.A_txt_path, dtype=np.int32, skiprows=0, usecols=1)
#len_dataset = len(len_dataset)
#G_lr_scheduler = LinearDecay(FLAGS.lr, FLAGS.epochs * len_dataset, FLAGS.ep_decay * len_dataset)
#D_lr_scheduler = LinearDecay(FLAGS.lr, FLAGS.epochs * len_dataset, FLAGS.ep_decay * len_dataset)
#g_optim = tf.keras.optimizers.Adam(G_lr_scheduler, beta_1=0.5)
#d_optim = tf.keras.optimizers.Adam(D_lr_scheduler, beta_1=0.5)

g_optim = RAdamOptimizer(FLAGS.lr)
d_optim = RAdamOptimizer(FLAGS.lr)

def test_func(img):

    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size]) / 127.5 - 1.

    return img

def train_data(A_zip, B_zip):

    A_img = tf.io.read_file(A_zip[0])
    A_img = tf.image.decode_jpeg(A_img, 3)
    A_img = tf.image.resize(A_img, [FLAGS.load_size, FLAGS.load_size])
    A_img = tf.image.random_crop(A_img, [FLAGS.img_size, FLAGS.img_size, 3], seed=1234) / 127.5 - 1.

    B_img = tf.io.read_file(B_zip[0])
    B_img = tf.image.decode_jpeg(B_img, 3)
    B_img = tf.image.resize(B_img, [FLAGS.load_size, FLAGS.load_size])
    B_img = tf.image.random_crop(B_img, [FLAGS.img_size, FLAGS.img_size, 3], seed=1234) / 127.5 - 1.

    if random() > 0.5:
        A_img = tf.image.flip_left_right(A_img)
        B_img = tf.image.flip_left_right(B_img)

    A_lab = int(A_zip[1]) - FLAGS.A_int
    B_lab = int(B_zip[1]) - FLAGS.B_int

    return A_img, A_lab, B_img, B_lab, A_zip[0], B_zip[0]

def age_data(list):

    img = tf.io.read_file(list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.load_size, FLAGS.load_size])
    img = tf.image.random_crop(img, [FLAGS.img_size, FLAGS.img_size, 3]) / 127.5 - 1.

    if random() > 0.5:

        img = tf.image.flip_left_right(img)

    return img

def buf_data(A, B):

    if FLAGS.A_int == 16:
        A_age_list = np.arange(FLAGS.A_int, 40, dtype=np.int32)
    else:
        A_age_list = np.arange(FLAGS.A_int, 64, dtype=np.int32)

    if FLAGS.B_int == 16:
        B_age_list = np.arange(FLAGS.B_int, 40, dtype=np.int32)
    else:
        B_age_list = np.arange(FLAGS.B_int, 64, dtype=np.int32)

    A_img = []
    B_img = []
    for j in range(FLAGS.n_classes):
        temp_A = []
        temp_B = []
        for i in range(len(A)):
            if int(A[i][1]) == A_age_list[j]:
                temp_A.append(A[i][0])
        A_img.append(temp_A)
        for i in range(len(B)):
            if int(B[i][1]) == B_age_list[j]:
                temp_B.append(B[i][0])
        B_img.append(temp_B)

    return A_img, B_img

def abs_criterion(input, target):
    return tf.reduce_mean(tf.abs(input - target))

def mae_criterion(input, target):
    return tf.reduce_mean((input - target)**2)

class Var(tf.Module):
  @tf.function
  def __call__(self, x):
    if not hasattr(self, "v"):  # Or set self.v to None in __init__
      self.v = tf.Variable(tf.ones([], dtype=tf.float32) * 1, dtype=tf.float32, trainable=True)
      #self.v = tf.Variable(tf.keras.initializers.Constant(0.0)(shape=[]), dtype=tf.float32, trainable=True)
    return self.v

@tf.function
def train_loss(A2B_model, B2A_model, A_model, B_model, A_images, B_images, A_target_age, B_target_age, 
               Cycle_A_lambda, Cycle_B_lambda, age_a2b_id_lambda, age_b2a_id_lambda):
    
    with tf.GradientTape(persistent=True) as g_tape, tf.GradientTape(persistent=True) as d_tape:

        fake_B = A2B_model(A_images, training=True)
        fake_A = B2A_model(B_images, training=True)

        #id_fake_B = B2A_model((A_target_age) * -A_images, training=True)
        #id_fake_A = A2B_model((B_target_age) * -B_images, training=True)

        id_fake_B = B2A_model(A_images, training=True)
        id_fake_A = A2B_model(B_images, training=True)

        fake_A_ = B2A_model(fake_B, training=True)
        fake_B_ = A2B_model(fake_A, training=True)

        real_A_logits = A_model(A_images, training=True)
        real_B_logits = B_model(B_images, training=True)
        fake_A_logits = A_model(fake_A, training=True)
        fake_B_logits = B_model(fake_B, training=True)

        Cycle_A_loss = 5.0 * abs_criterion(A_images, fake_A_)
        Cycle_B_loss = 5.0 * abs_criterion(B_images, fake_B_)
        Cycle_loss = Cycle_A_loss + Cycle_B_loss
        ##################
        # Age loss 
        A2B_G_id_loss = 4.5 * abs_criterion(A_target_age, id_fake_B)    # 이걸로 학습 다시 돌리기!!
        B2A_G_id_loss = 4.5 * abs_criterion(B_target_age, id_fake_A)
        G_Id_loss = A2B_G_id_loss + B2A_G_id_loss
        ##################

        ##################
        # Gender loss
        #A2B_G_gender_loss = tf.where(B_tar_name == A_img_name, 2.5 * abs_criterion(A_images, fake_A), 2.5 * abs_criterion(fake_B, id_fake_A))
        #B2A_G_gender_loss = tf.where(A_tar_name == B_img_name, 2.5 * abs_criterion(B_images, fake_B), 2.5 * abs_criterion(fake_A, id_fake_B))
        #G_Gender_loss = A2B_G_gender_loss + B2A_G_gender_loss
        ##################

        A2B_G_GAN_loss = mae_criterion(fake_B_logits, tf.ones_like(fake_B_logits))
        B2A_G_GAN_loss = mae_criterion(fake_A_logits, tf.ones_like(fake_A_logits))
        G_GAN_loss = A2B_G_GAN_loss + B2A_G_GAN_loss

        Age_gender_loss = Cycle_loss + G_Id_loss
        g_loss = G_GAN_loss + Age_gender_loss
        #####################################################################################

        A2B_D_GAN_loss = (mae_criterion(fake_A_logits, tf.zeros_like(fake_A_logits)) \
                        + mae_criterion(real_A_logits, tf.ones_like(real_A_logits))) / 2
        B2A_D_GAN_loss = (mae_criterion(fake_B_logits, tf.zeros_like(fake_B_logits)) \
                        + mae_criterion(real_B_logits, tf.ones_like(real_B_logits))) / 2
        D_GAN_loss = A2B_D_GAN_loss + B2A_D_GAN_loss

        d_loss = D_GAN_loss
        #####################################################################################

    g_grads = g_tape.gradient(g_loss, A2B_model.trainable_variables + B2A_model.trainable_variables)
    #Age_gender_grads = g_tape.gradient(Age_gender_loss, A.trainable_variables + B.trainable_variables)
    d_grads = d_tape.gradient(d_loss, A_model.trainable_variables + B_model.trainable_variables)
    
    #g_optim.apply_gradients(zip(Age_gender_grads, A.trainable_variables + B.trainable_variables))
    g_optim.apply_gradients(zip(g_grads, A2B_model.trainable_variables + B2A_model.trainable_variables))
    d_optim.apply_gradients(zip(d_grads, A_model.trainable_variables + B_model.trainable_variables))

    return g_loss, d_loss, Cycle_A_lambda, Cycle_B_lambda, age_a2b_id_lambda, age_b2a_id_lambda

def main(argv=None):
    
    A_model = ConvDiscriminator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    B_model = ConvDiscriminator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    A2B_model = A2B_att_based_generator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    B2A_model = B2A_att_based_generator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    A2B_model.summary()
    B2A_model.summary()
    A_model.summary()

    Cycle_A_lambda, Cycle_B_lambda = 1., 1.
    age_a2b_id_lambda, age_b2a_id_lambda = 1., 1.

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(A2B_model=A2B_model,
                                   B2A_model=B2A_model,
                                   A_model=A_model,
                                   B_model=B_model,
                                   g_optim=g_optim,
                                   d_optim=d_optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored the latest checkpoint")

    if FLAGS.train:
        count = 0
        
        A_img_data = np.loadtxt(FLAGS.A_txt_path, dtype="<U100", skiprows=0, usecols=0)
        A_img_data = [FLAGS.A_img_path + img for img in A_img_data]
        A_lab_data = np.loadtxt(FLAGS.A_txt_path, dtype=np.int32, skiprows=0, usecols=1)
        A = list(zip(A_img_data, A_lab_data))

        B_img_data = np.loadtxt(FLAGS.B_txt_path, dtype="<U100", skiprows=0, usecols=0)
        B_img_data = [FLAGS.B_img_path + img for img in B_img_data]
        B_lab_data = np.loadtxt(FLAGS.B_txt_path, dtype=np.int32, skiprows=0, usecols=1)
        B = list(zip(B_img_data, B_lab_data))

        #A_age_buf_img = np.loadtxt(FLAGS.A_buf_age_txt, dtype="<U100", skiprows=0, usecols=0)
        #A_age_buf_img = [FLAGS.A_buf_age_img + img for img in A_age_buf_img]
        #A_age_buf_lab = np.loadtxt(FLAGS.A_buf_age_txt, dtype=np.int32, skiprows=0, usecols=1)
        #A_ = list(zip(A_age_buf_img, A_age_buf_lab))

        #B_age_buf_img = np.loadtxt(FLAGS.B_buf_age_txt, dtype="<U100", skiprows=0, usecols=0)
        #B_age_buf_img = [FLAGS.B_buf_age_img + img for img in B_age_buf_img]
        #B_age_buf_lab = np.loadtxt(FLAGS.B_buf_age_txt, dtype=np.int32, skiprows=0, usecols=1)
        #B_ = list(zip(B_age_buf_img, B_age_buf_lab))

        #############################
        # Define the graphs
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = FLAGS.graphs + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        #############################

        A, B = np.array(A), np.array(B)
        A_age_lists, B_age_lists = buf_data(A, B)

        for epoch in range(FLAGS.epochs):
            
            np.random.shuffle(A)
            np.random.shuffle(B)

            data = tf.data.Dataset.from_tensor_slices((A, B))
            data = data.shuffle(len(A))
            data = data.map(train_data)
            data = data.batch(FLAGS.batch_size)
            data = data.prefetch(tf.data.experimental.AUTOTUNE)

            train_iter = iter(data)
            train_idx = len(A) // FLAGS.batch_size
            for step in range(train_idx):

                A_images, A_labels, B_images, B_labels, A_file_name, B_file_name = next(train_iter)
                list_A = np.array(A_age_lists[A_labels[0].numpy()])
                list_B = np.array(B_age_lists[B_labels[0].numpy()])
                #######################################################################
                A_age_data = tf.data.Dataset.from_tensor_slices(list_A)
                A_age_data = A_age_data.shuffle(len(list_A))
                A_age_data = A_age_data.map(age_data)
                A_age_data = A_age_data.batch(FLAGS.batch_size)
                A_age_data = A_age_data.prefetch(tf.data.experimental.AUTOTUNE)

                B_age_data = tf.data.Dataset.from_tensor_slices(list_B)
                B_age_data = B_age_data.shuffle(len(list_B))
                B_age_data = B_age_data.map(age_data)
                B_age_data = B_age_data.batch(FLAGS.batch_size)
                B_age_data = B_age_data.prefetch(tf.data.experimental.AUTOTUNE)
                #######################################################################
                iter_A_age_list = iter(A_age_data)
                iter_B_age_list = iter(B_age_data)

                A_target_age = next(iter_B_age_list)
                B_target_age = next(iter_A_age_list)

                g_loss, d_loss, Cycle_A_lambda, Cycle_B_lambda, age_a2b_id_lambda, age_b2a_id_lambda = train_loss(A2B_model, B2A_model, A_model, B_model, A_images, B_images, A_target_age, B_target_age, 
                                                                                                        Cycle_A_lambda, Cycle_B_lambda, age_a2b_id_lambda, age_b2a_id_lambda)
                print("======================================================================================")
                print("Epoch: {} [{}/{}] G loss = {}, D loss = {}".format(epoch, step, train_idx, g_loss, d_loss))
                #print("Cycle_A_lambda : {}".format(Cycle_A_lambda))
                #print("Cycle_B_lambda : {}".format(Cycle_B_lambda))
                #print("age_a2b_id_lambda : {}".format(age_a2b_id_lambda))
                #print("age_b2a_id_lambda : {}".format(age_b2a_id_lambda))
                print("======================================================================================")

                with train_summary_writer.as_default():
                    tf.summary.scalar('Generator loss', g_loss, step=count)
                    tf.summary.scalar('Discriminator loss', d_loss, step=count)

                #if count % 100 == 0:
                #    fake_B = A2B_model(A_images, training=False)
                #    fake_A = B2A_model(B_images, training=False)

                #    A_name = str(tf.compat.as_str_any(A_file_name[0].numpy())).split('/')[-1].split('.')[0]
                #    B_name = str(tf.compat.as_str_any(B_file_name[0].numpy())).split('/')[-1].split('.')[0]
                    
                #    plt.imsave(FLAGS.sample_img_path + "/"+ A_name + "_fake_B_{}.jpg".format(count), fake_B[0].numpy() * 0.5 + 0.5)
                #    plt.imsave(FLAGS.sample_img_path + "/"+ B_name +"_fake_A_{}.jpg".format(count), fake_A[0].numpy() * 0.5 + 0.5)
                #    #plt.imsave(FLAGS.sample_img_path + "/fake_BB_{}.jpg".format(count), fake_BB[0].numpy() * 0.5 + 0.5)
                #    #plt.imsave(FLAGS.sample_img_path + "/fake_AA_{}.jpg".format(count), fake_AA[0].numpy() * 0.5 + 0.5)
                #    plt.imsave(FLAGS.sample_img_path + "/"+ A_name + "_real_A_{}.jpg".format(count), A_images[0].numpy() * 0.5 + 0.5)
                #    plt.imsave(FLAGS.sample_img_path + "/"+ B_name + "_real_B_{}.jpg".format(count), B_images[0].numpy() * 0.5 + 0.5)

                #if count % 1000 == 0:
                #    number = int(count/1000)
                #    model_dir = "%s/%s" % (FLAGS.save_checkpoint, number)
                #    if not os.path.isdir(model_dir):
                #        print("Make {} files to save checkpoint files".format(number))
                #        os.makedirs(model_dir)
                #    ckpt = tf.train.Checkpoint(A2B_model=A2B_model,
                #                               B2A_model=B2A_model,
                #                               A_model=A_model,
                #                               B_model=B_model,
                #                               g_optim=g_optim,
                #                               d_optim=d_optim)
                #    ckpt_dir = model_dir + "/New_generation_{}.ckpt".format(count)
                #    ckpt.save(ckpt_dir)
                count += 1
    else:
        if FLAGS.dir == "A2B":
            A_img = np.loadtxt(FLAGS.A_test_txt, dtype="<U200", skiprows=0, usecols=0)
            A_name = A_img
            A_img = [FLAGS.A_test_img + img for img in A_img]

            gener = tf.data.Dataset.from_tensor_slices(A_img)
            gener = gener.map(test_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            gener = gener.batch(1)
            gener = gener.prefetch(tf.data.experimental.AUTOTUNE)

            it = iter(gener)
            for i in range(len(A_img)):
                images = next(it)

                fake_B = A2B_model(images, False)

                plt.imsave(FLAGS.A_test_output + "/" + A_name[i], fake_B[0].numpy() * 0.5 + 0.5)

                if i % 1000 == 0:
                    print("{} images are completed!!".format(i + 1))

        if FLAGS.dir == "B2A":
            B_img = np.loadtxt(FLAGS.B_test_txt, dtype="<U200", skiprows=0, usecols=0)
            B_name = B_img
            B_img = [FLAGS.B_test_img + img for img in B_img]

            gener = tf.data.Dataset.from_tensor_slices(B_img)
            gener = gener.map(test_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            gener = gener.batch(1)
            gener = gener.prefetch(tf.data.experimental.AUTOTUNE)

            it = iter(gener)
            for i in range(len(B_img)):
                images = next(it)

                fake_A = B2A_model(images, False)

                plt.imsave(FLAGS.B_test_output + "/" + B_name[i], fake_A[0].numpy() * 0.5 + 0.5)

                if i % 1000 == 0:
                    print("{} images are completed!!".format(i + 1))


if __name__ == "__main__":
    app.run(main)