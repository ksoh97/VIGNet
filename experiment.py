# Import APIs
import tensorflow as tf
import numpy as np
import network
import utils

from sklearn.metrics import accuracy_score

def experiment(subject, fold):
    # Hyper-parameter setting
    num_epochs = 100
    batch_size = 10
    learning_rate = 1e-5

    # Placeholding
    # TODO: Check placeholding
    num_channel = 62
    X_train = tf.placeholder(tf.float32, [None, num_channel, 3000, 1]) # (minibatch size, # of electrode channel (based on dataset), # of timepoints, 1)
    Y_train = tf.placeholder(tf.float32, [None])

    X_valid = tf.placeholder(tf.float32, [None, num_channel, 3000, 1])
    Y_valid = tf.placeholder(tf.float32, [None])

    X_test = tf.placeholder(tf.float32, [None, num_channel, 3000, 1])
    Y_test = tf.placeholder(tf.float32, [None])

    # Call Network
    training_loss, train_prediction = network.drowsynet(eeg=X_train, label=Y_train)
    _, valid_prediction = network.drowsynet(eeg=X_valid, label=Y_valid, reuse=True)
    _, test_prediction = network.drowsynet(eeg=X_test, label=Y_test, reuse=True)

    # Call tunable parameters
    theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="drowsynet") # theta means weight

    # TODO-HW: Learning rate decay
    # # Exponentially decayed learning rate
    # global_step = tf.Variable(0, trainable=False)
    # learning_rate = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_step,
    #                                            decay_steps=10000, decay_rate=0.96, staircase=True)

    # Adam optimizer
    # TODO-HW: Adam, RMSprop, AdaDelta, Adaboost / momentum, Adagrad etc.
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss=training_loss, var_list=theta)

    # Load dataset
    train_eeg, train_label, valid_eeg, valid_label, test_eeg, test_label = \
        utils.load_datasetabc(subject=subject, fold=fold)


    # train_eeg, train_label, valid_eeg, valid_label, _, _ = \
    #     utils.load_dataset(subject=subject, fold=fold)
    # _, _, _, _, test_eeg, test_label = \
    #     utils.load_dataset(subject=subject+4, fold=fold)

    # Start training
    # For GPU server
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # saver = tf.train.Saver(keep_checkpoint_every_n_hours=24, max_to_keep=1000)
    print("Start Training, Subject ID: %02d, %01d-th Fold" % (subject, fold))

    total_batch = int(train_eeg.shape[0]/batch_size)

    for epoch in range(num_epochs):
        print("Operating Epoch %02d..." %epoch)

        # Randomize the training dataset for each epoch
        # DNN MUST generalize the training data, NOT memorize.
        rand_idx = np.random.permutation(train_eeg.shape[0])
        train_eeg = train_eeg[rand_idx, :, :]
        train_label = train_label[rand_idx]

        # Feed dictionaries
        for batch in range(total_batch):
            batch_X = train_eeg[batch*batch_size:(batch+1)*batch_size, :, :]
            batch_Y = train_label[batch*batch_size:(batch+1)*batch_size]

            tr_loss, tr_pred, _ = sess.run(fetches=[training_loss, train_prediction, optimizer],
                                           feed_dict={X_train: batch_X, Y_train: batch_Y})
            # print(np.squeeze(batch_Y) == np.squeeze(tr_pred))

            if batch % 100 == 0:
                print("Intermediate Loss: %.3f" % tr_loss)

        # Calculate validation accuracy
        # batch_X, batch_Y = valid_eeg, valid_label
        correct = 0
        for b in range(valid_eeg.shape[0]):
            # print(b)
            batch_X, batch_Y = valid_eeg[b:b+1, :, :, :], valid_label[b:b+1]
            val_pred = sess.run(fetches=[valid_prediction], feed_dict={X_valid: batch_X, Y_valid: batch_Y})
            val_pred = np.asarray(val_pred).astype(int)
            # print(np.squeeze(val_pred), np.squeeze(batch_Y))
            if np.squeeze(val_pred) == np.squeeze(batch_Y):
                correct += 1

        # val_pred = sess.run(fetches=[valid_prediction], feed_dict={X_valid: batch_X, Y_valid: batch_Y})
        # val_pred = np.asarray(val_pred).astype(int)
        # print("%03d-th Epoch, Validation Acc: %.3f" %(epoch, accuracy_score(y_true=np.squeeze(batch_Y)
        #                                                                     , y_pred=np.squeeze(val_pred))))
        print(correct / valid_eeg.shape[0])

        # Calculate test accuracy
        # batch_X, batch_Y = test_eeg, test_label
        correct = 0
        for b in range(test_eeg.shape[0]):
            batch_X, batch_Y = test_eeg[b:b+1, :, :, :], test_label[b:b+1]
            tst_pred = sess.run(fetches=[test_prediction], feed_dict={X_test: batch_X, Y_test: batch_Y})
            tst_pred = np.asarray(tst_pred).astype(int)
            if np.squeeze(tst_pred) == np.squeeze(batch_Y):
                correct += 1
            # print(np.squeeze(batch_Y).astype(np.int), np.squeeze(tst_pred).astype(np.int))
        print(test_label)
        print(correct / test_eeg.shape[0])


        # tst_pred = sess.run(fetches=[test_prediction], feed_dict={X_test: batch_X, Y_test: batch_Y})
        # tst_pred = np.asarray(tst_pred).astype(int)
        # print("%03d-th Epoch, Test Acc: %.3f\n" % (epoch, accuracy_score(y_true=np.squeeze(batch_Y)
        #                                                                  , y_pred=np.squeeze(tst_pred))))

    tf.reset_default_graph()
    return