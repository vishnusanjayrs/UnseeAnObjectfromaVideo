import cv2
import tensorflow as tf
import numpy as np
from inpaint_model import InpaintCAModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import temporal

class TemporalLoss(nn.Module):
    def __init__(self):
        super(TemporalLoss, self).__init__()
    def forward(self, generated, corrupted, weight_mask):
        print('getting shapes',generated.shape, corrupted.shape, weight_mask.shape)
        cri = torch.nn.L1Loss(reduce=False)
        l1 = cri(generated, corrupted)
        lc = l1 * weight_mask
        print(l1.shape)
        # lc = l1
        lc = lc.mean(dim=[0, 1, 2])
        return lc

class temporal_inpaint(nn.Module):
    def __init__(self):
        super(temporal_inpaint, self).__init__()
    def temporal_smooth(self,ref,prev_inp, result, name, mask, mask_prev, threshould, per_iter_step = 1500):
        output_dir = 'output/own_video/'
        # print(ref.shape, result_rgb.shape)
        # temporal_loss = tf.losses.absolute_difference(np.expand_dims(ref, axis=0), np.expand_dims(result_rgb,axis=0))
        result_rgb = result[0][:, :, [2, 1, 0]] / 255
        # plt.imshow(result_rgb[:, :, [2, 1, 0]])
        # plt.show()
        print('mask shape', mask.shape)
        result_rgb = torch.Tensor(result_rgb)
        ref = torch.Tensor(ref / 255)
        # prev_inp = torch.Tensor(prev_inp / 255)
        # ref = torch.mean(ref,prev_inp)
        optimizer = torch.optim.Adam([result_rgb.requires_grad_()])
        criterion = TemporalLoss()
        mask_prev=mask_prev[0]
        union = mask[0]|mask_prev
        print(np.unique(mask))
        print(np.unique(mask_prev))
        print(np.unique(union))
        union = union/255.0
        for i in range(per_iter_step):
            optimizer.zero_grad()
            loss = criterion(result_rgb, ref, torch.Tensor(union))
            print(loss)
            if(loss.item()<threshould):
                break
            loss.backward()
            optimizer.step()
            print(i, 'th contextual loss: ', loss.item())
        # print(result_rgb.shape)
        # plt.imshow(result_rgb.detach().numpy()[:, :, [2, 1, 0]])
        # plt.show()
        output_name = output_dir + 'rf_'+str(name)+'.png'
        plt.imsave(output_name, result_rgb.detach().numpy()[:, :, [2, 1, 0]])


sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)

model = InpaintCAModel()
input_image_ph = tf.placeholder(
    tf.float32, shape=(1, 256, 256*2, 3))
output = model.build_server_graph(input_image_ph)
output = (output + 1.) * 127.5
output = tf.reverse(output, [-1])
output = tf.saturate_cast(output, tf.uint8)
vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
assign_ops = []
for var in vars_list:
    vname = var.name
    from_name = vname
    var_value = tf.contrib.framework.load_variable('model_logs/', from_name)
    assign_ops.append(tf.assign(var, var_value))
sess.run(assign_ops)
print('Model loaded.')

tempo_model = temporal_inpaint()

mask_prev = cv2.imread('input/256_big_mask/hallway_103_mask.png')

for i in range(104, 330):
    print('=====in ',i)
    input_img = 'input/hallway_' + str(i) + '_input.png'
    mask_img = 'input/hallway_' + str(i) + '_mask.png'
    ref_img = 'output/own_video/rf_'+str(i-1)+'.png'
    prev_input_img = 'input/own_video/hallway_' + str(i-1) + '_input.png'
    image = cv2.imread(input_img)
    mask = cv2.imread(mask_img)

    ref = cv2.imread(ref_img)
    print('img size',image.shape)
    assert image.shape == mask.shape

    prev_input = cv2.imread(prev_input_img)

    # ref = np.mean(ref ,prev_input,keepdims=True)

    print('Shape of image before the weird griding: {}'.format(image.shape))
    h, w, _ = image.shape
    grid = 4
    image = image[:h // grid * grid, :w // grid * grid, :]
    mask = mask[:h // grid * grid, :w // grid * grid, :]
    print('Shape of image: {}'.format(image.shape))

    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)

    # load pretrained model
    result = sess.run(output, feed_dict={input_image_ph: input_image})
    tempo_model.temporal_smooth(ref,prev_input,result,i,mask, mask_prev, 0.0002)
    mask_prev=mask.copy()


