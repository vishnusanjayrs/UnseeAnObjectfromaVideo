import cv2
import numpy as np
print('first')
import tensorflow as tf


from inpaint_model import InpaintCAModel




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
output_dir = 'output/own_video/'

for i in range(1, 330):
    input_img = 'input/hallway_' + str(i) + '_input.png'
    mask_img = 'input/hallway_' + str(i) + '_mask.png'
    image = cv2.imread(input_img)
    mask = cv2.imread(mask_img)

    assert image.shape == mask.shape

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
    output_name = output_dir + 'output_' + str(i) + '.png'
    cv2.imwrite(output_name, result[0][:, :, ::-1])
