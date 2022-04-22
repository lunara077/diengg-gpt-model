import uuid
import pickle

import valohai
import sys, os, json, shutil, re, urllib.request, time
from src.accumulate import AccumulatingOptimizer
from tensorflow.core.protobuf import rewriter_config_pb2
from src import model, sample, encoder
import tensorflow as tf
from src.load_dataset import load_dataset, Sampler

tf.compat.v1.disable_eager_execution()

# Feature selection
# Restler output file may contain duplicate requests.
# This method eliminates duplicate ones and also extract the desired features. We just need request_type , request_uri and request body
def process_restler_output(restler_raw_file='test_cases_produced.csv',
                           restler_processed_file="RESTler_unique_output.txt"):
    import csv, itertools
    file = open(restler_raw_file)
    csvreader = csv.reader(file)
    header = next(csvreader)
    print(header)
    rows = []
    for row in csvreader:
        rows.append(row)
    file.close()

    for row in rows:
        del row[5]
        del row[4]
        del row[0]

    rows = [list(tupl) for tupl in {tuple(item) for item in rows}]
    textfile = open(restler_processed_file, "w")
    firstLine = True
    for row in rows:
        if row[2]:
            element = "HTTP " + row[0] + " " + row[1] + " " + row[2]
        else:
            element = "HTTP " + row[0] + " " + row[1]
        if (firstLine):
            textfile.write(element)
            firstLine = False
        else:
            textfile.write("\n" + element)
    textfile.close()
    return restler_processed_file


# It is needed to download the GPT-2 pre-trained model. there are four types:
# 1. 124M (default)   2. 355M   3. 774M   4. 1558M
# The last two models are large and cannot finetuned in the google Colab
def get_model(model_type='124M'):
    # create the directory if not exist
    path = os.path.join('models', model_type)
    if not os.path.exists(path):
        os.makedirs(path)
        url = "https://openaipublic.blob.core.windows.net/gpt-2/models/" + model_type + "/" + "checkpoint"
        urllib.request.urlretrieve(url, os.path.join(path, "checkpoint"))
        url = "https://openaipublic.blob.core.windows.net/gpt-2/models/" + model_type + "/" + "encoder.json"
        urllib.request.urlretrieve(url, os.path.join(path, "encoder.json"))
        url = "https://openaipublic.blob.core.windows.net/gpt-2/models/" + model_type + "/" + "hparams.json"
        urllib.request.urlretrieve(url, os.path.join(path, "hparams.json"))
        url = "https://openaipublic.blob.core.windows.net/gpt-2/models/" + model_type + "/" + "model.ckpt.data-00000-of-00001"
        urllib.request.urlretrieve(url, os.path.join(path, "model.ckpt.data-00000-of-00001"))
        url = "https://openaipublic.blob.core.windows.net/gpt-2/models/" + model_type + "/" + "model.ckpt.index"
        urllib.request.urlretrieve(url, os.path.join(path, "model.ckpt.index"))
        url = "https://openaipublic.blob.core.windows.net/gpt-2/models/" + model_type + "/" + "model.ckpt.meta"
        urllib.request.urlretrieve(url, os.path.join(path, "model.ckpt.meta"))
        url = "https://openaipublic.blob.core.windows.net/gpt-2/models/" + model_type + "/" + "vocab.bpe"
        urllib.request.urlretrieve(url, os.path.join(path, "vocab.bpe"))


# return Tensorflow session. A Session places the graph ops onto Devices, such as CPUs or GPUs, and provides methods to execute them.
def tensorflow_session():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF
    return tf.compat.v1.Session(config=config)


# with fine tuning we can train the model on our specific dataset
def fine_tuning(session,
                restler_raw_file='data/test_cases_produced.csv',
                steps=-1,
                model_type='124M',
                batch_size=1,
                learning_rate=0.0001,
                accumulate_gradients=5,
                run_name='run1',
                sample_length=1023,
                only_train_transformer_layers=False,
                optimizer='adam'):
    dataset = process_restler_output(restler_raw_file)

    checkpoint_path = os.path.join('checkpoint', run_name)

    try:
        os.makedirs(checkpoint_path)
    except:
        pass
    files = [f for f in os.listdir(checkpoint_path)]
    for file in ['hparams.json', 'encoder.json', 'vocab.bpe']:
        try:
            shutil.copyfile(os.path.join('models', model_type, file),
                            os.path.join(checkpoint_path, file))
        except FileNotFoundError as fnf_error:
            print("You need to download the GPT-2 model first via get_model()")
            raise (fnf_error)

    enc = encoder.get_encoder(checkpoint_path)
    hparams = model.default_hparams()
    with open(os.path.join(checkpoint_path, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if sample_length > hparams.n_ctx:
        raise ValueError(
            "Can't get samples longer than window size: %s" % hparams.n_ctx)

    context = tf.compat.v1.placeholder(tf.int32, [batch_size, None])
    gpus = []

    output = model.model(hparams=hparams, X=context, gpus=gpus, reuse=False)
    loss = tf.reduce_mean(
        input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=context[:, 1:], logits=output['logits'][:, :-1]))

    all_vars = [v for v in tf.compat.v1.trainable_variables() if 'model' in v.name]
    # For models larger than 124M, it is better to set only_train_transformer_layers= true
    train_vars = [v for v in all_vars if '/h' in v.name] if only_train_transformer_layers else all_vars

    if optimizer == 'adam':
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)

    # for models larger than 124M, it is better to set accumulate_gradients = 1
    # accumulate_gradients = 1 means no accumulated grads
    if accumulate_gradients > 1:

        # It calculates the loss and gradients after each mini-batch, but instead of updating the model parameters, it waits and accumulates the gradients over consecutive batches.
        opt = AccumulatingOptimizer(
            opt=opt,
            var_list=train_vars)
        opt_reset = opt.reset()
        opt_compute = opt.compute_gradients(loss)
        opt_apply = opt.apply_gradients()
        summary_loss = tf.compat.v1.summary.scalar('loss', opt_apply)
    else:
        opt_grads = tf.gradients(ys=loss, xs=train_vars)
        opt_grads = list(zip(opt_grads, train_vars))
        opt_apply = opt.apply_gradients(opt_grads)
        summary_loss = tf.compat.v1.summary.scalar('loss', loss)

    summary_log = tf.compat.v1.summary.FileWriter(checkpoint_path)

    saver = tf.compat.v1.train.Saver(
        var_list=all_vars,
        max_to_keep=1)
    # execute the operation in the tensors
    session.run(tf.compat.v1.global_variables_initializer())
    ckpt = tf.train.latest_checkpoint(os.path.join('models', model_type))

    # Loading checkpoint
    saver.restore(session, ckpt)

    print('Loading dataset...')
    chunks = load_dataset(enc, dataset, 50000)
    data_sampler = Sampler(chunks)
    print('dataset has', data_sampler.total_size, 'tokens')
    print('Training...')
    counter = 1
    counter_path = os.path.join(checkpoint_path, 'counter')
    counter_base = counter

    def sample_batch():
        return [data_sampler.sample(1024) for _ in range(batch_size)]

    avg_loss = (0.0, 0.0)
    start_time = time.time()

    if steps:
        steps = int(steps)

    # saving the trained model checkpoints
    def save():
        try:
            os.makedirs(checkpoint_path)
        except:
            pass
        print('Saving', os.path.join(checkpoint_path, 'model-{}').format(counter - 1))
        saver.save(
            session,
            os.path.join(checkpoint_path, 'model'),
            global_step=counter - 1)
        with open(counter_path, 'w') as fp:
            fp.write(str(counter - 1) + '\n')
        
        suffix = uuid.uuid4()
        output_path = valohai.outputs().path(f'model-{suffix}.h5')
        with open(output_path, 'wb') as f:
            pickle.dump(session, f)
        model.save(output_path)
        
    while True:
        if steps > 0 and counter == (counter_base + steps):
            save()
            return

        if accumulate_gradients > 1:
            # execute the operation in the tensors
            session.run(opt_reset)
            for _ in range(accumulate_gradients):
                session.run(
                    opt_compute, feed_dict={context: sample_batch()})
            (v_loss, v_summary) = session.run((opt_apply, summary_loss))
        else:
            (_, v_loss, v_summary) = session.run(
                (opt_apply, loss, summary_loss),
                feed_dict={context: sample_batch()})

        summary_log.add_summary(v_summary, counter)

        if counter % 20 == 0:
            avg_loss = (avg_loss[0] * 0.99 + v_loss,
                        avg_loss[1] * 0.99 + 1.0)

            print(
                '[{counter} | {time:2.2f}] loss={loss:2.2f} avg={avg:2.2f}'
                    .format(
                    counter=counter,
                    time=time.time() - start_time,
                    loss=v_loss,
                    avg=avg_loss[0] / avg_loss[1]))

        counter += 1


def main():
    # valohai.prepare enables us to update the valohai.yaml configuration file with
    # the Valohai command-line client by running `valohai yaml step train_model.py`

    valohai.prepare(
        step='train-model',
        image='tensorflow/tensorflow:1.15.0',
        default_inputs={
            'dataset': '"data/test_cases_produced.csv"',
        }
    )

    data_file = valohai.inputs('dataset').path()

    get_model(model_type="124M")

    session = tensorflow_session()
    # with fine tuning we can train the model on our specific dataset
    fine_tuning(session,
                restler_raw_file=data_file,
                model_type='124M',
                steps=100,
                run_name='run1')
    


if __name__ == '__main__':
    main()
