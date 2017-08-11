<!-- 直接、C#のコードでグラフを作成しようとしたが、どうやら先人たちはPython上でグラフを書き、学習モデルを作成後、それを他の言語上で読み込んでいる。

 モデルを読み込む際に、Android（Java）やC++上でも同様であるが、チェックポイント(シリアル化された変数)のエクスポートデータを直接読込むことはできない.

 そのため、プロトコルバッファ(シリアル化されたグラフ)にチェックポイントをマージしなければ使うことができない。つまり、変数の状態をもつプロトコルバッファを作成する。

 グラフとテンソルデータの両方を出力するためには、VariablesをConstantに変換後、再度グラフを作成してProtocolBuffersファイルとして出力する必要がある。

　方法としては、2つあり、手動で変換する方法と、自動で変換する方法がある。今回は両方法を示す。 -->

Although it is the same on Android (Java) and C ++ when reading the model, it is not possible to directly read the checkpoint (serialized variable) export data.  

So, you can not use it without merging checkpoints in the protocol buffer (serialized graph). So, create a protocol buffer has the state of the variable.  

In order to output both graphs and tensor data, after converting Variables to Constant, it is necessary to create graph again and output it as a ProtocolBuffers file.  

There are two methods, there are a manual conversion method and an automatic conversion method. This time shows both methods.


# Version
macOS Sierra
TensorFlow on python3: 1.0.0
TensorFlow on c#(c): 1.1.0-rc0
Python 3.6.0
Visul Studio for Mac ver 7.0.1

# Create a CNN model.
First, Make source code for CNN model.
Name is model.py.

Download input_data.py from [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/input_data.py
 "https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/input_data.py
") and place it on the same directory of model.py.


```py:model.py
# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals
import input_data
import tensorflow as tf
import shutil
import os

# Output directory of model
export_dir = './models'

if os.path.exists(export_dir):
    shutil.rmtree(export_dir)
    os.mkdir(export_dir);
else:
    os.mkdir(export_dir);

'''
Weight Initialization

To create this model, we're going to need to create a lot of weights and biases.
One should generally initialize weights with a small amount of noise for symmetry breaking,
and to prevent 0 gradients. Since we're using ReLU neurons, it is also good practice to initialize
them with a slightly positive initial bias to avoid "dead neurons." Instead of doing this repeatedly
while we build the model, let's create two handy functions to do it for us.
'''
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Convolution layer
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Pooling layer
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Create a graph
g = tf.Graph()
with g.as_default():
    x = tf.placeholder("float", shape=[None, 784])
    y_ = tf.placeholder("float", shape=[None, 10])

    # First layer
    # [5, 5, 1, 32] is the first 5, 5 is the patch size, 1 is the number of input channels, 32 is the number of output channels
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # Second layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # fully connected layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Add softmax as in the first layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # Learning and evaluation of models
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Create a session and initialize variables
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Start learning
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(
                    {x: batch[0], y_: batch[1], keep_prob: 1.0}, sess)
                print("step %d, training accuracy %g" % (i, train_accuracy))
            train_step.run(
                {x: batch[0], y_: batch[1], keep_prob: 0.5}, sess)

        # Score display
        print("test accuracy %g" % accuracy.eval(
            {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}, sess))
```

Reference1 :  https://github.com/miyosuda/TensorFlowAndroidMNIST/blob/master/trainer-script/expert.py  
Reference2 : [TensorFlow 畳み込みニューラルネットワークで手書き認識率99.2%の分類器を構築](http://qiita.com/haminiku/items/36982ae65a770565458d?1447141109024=1 "TensorFlow 畳み込みニューラルネットワークで手書き認識率99.2%の分類器を構築")


## How to manually convert
<!-- 学習後に、Variableの値をevalで取り出して、Constantにする。
　流れとしてはViriables -> ndarray -> Constantと変換する。その後、Constantでグラフを再構成して、プロトコルバッファとして書き出す。名前は、C#上でモデルを読込むときに対応させるためのもの。 -->
After learning, That retrieve the value of Variable to use eval() and make it Constant.

As a flow convert to Viriables -> ndarray -> Constant. After that, Constant reconstructs the graph and writes it as a protocol buffer. The name is to correspond to when loading the model on C #.

```py:
with tf.Session() as sess:
        ...
        # Convert the contents of Viriables to ndarray
        _W_conv1 = W_conv1.eval(sess)
        _b_conv1 = b_conv1.eval(sess)
        _W_conv2 = W_conv2.eval(sess)
        _b_conv2 = b_conv2.eval(sess)
        _W_fc1 = W_fc1.eval(sess)
        _b_fc1 = b_fc1.eval(sess)
        _W_fc2 = W_fc2.eval(sess)
        _b_fc2 = b_fc2.eval(sess)


# After converting ndarray to a constant, reconstruct the new graph.
g_2 = tf.Graph()
with g_2.as_default():
    # The input node is "input". This is specified when loading .pb.
    x_2 = tf.placeholder("float", shape=[None, 784], name="input")

    W_conv1_2 = tf.constant(_W_conv1, name="constant_W_conv1")
    b_conv1_2 = tf.constant(_b_conv1, name="constant_b_conv1")
    x_image_2 = tf.reshape(x_2, [-1, 28, 28, 1])
    h_conv1_2 = tf.nn.relu(conv2d(x_image_2, W_conv1_2) + b_conv1_2)
    h_pool1_2 = max_pool_2x2(h_conv1_2)

    W_conv2_2 = tf.constant(_W_conv2, name="constant_W_conv2")
    b_conv2_2 = tf.constant(_b_conv2, name="constant_b_conv2")
    h_conv2_2 = tf.nn.relu(conv2d(h_pool1_2, W_conv2_2) + b_conv2_2)
    h_pool2_2 = max_pool_2x2(h_conv2_2)

    W_fc1_2 = tf.constant(_W_fc1, name="constant_W_fc1")
    b_fc1_2 = tf.constant(_b_fc1, name="constant_b_fc1")
    h_pool2_flat_2 = tf.reshape(h_pool2_2, [-1, 7 * 7 * 64])
    h_fc1_2 = tf.nn.relu(tf.matmul(h_pool2_flat_2, W_fc1_2) + b_fc1_2)

    W_fc2_2 = tf.constant(_W_fc2, name="constant_W_fc2")
    b_fc2_2 = tf.constant(_b_fc2, name="constant_b_fc2")

    # Since it only outputs data after learning, it is not necessary "Dropout layer".
    # The output node is "output". Specify this when loading .pb in the same way as the input node.
    y_conv_2 = tf.nn.softmax(tf.matmul(h_fc1_2, W_fc2_2) + b_fc2_2, name="output")

    with tf.Session() as sess_2:
        init_2 = tf.global_variables_initializer();
        sess_2.run(init_2)

        # Output the graph as a ProtocolBuffers file.
        graph_def = g_2.as_graph_def()
        tf.train.write_graph(graph_def, export_dir, 'Manual_model.pb', as_text=False)

        # Test the model after training.
        y__2 = tf.placeholder("float", [None, 10])
        correct_prediction_2 = tf.equal(tf.argmax(y_conv_2, 1), tf.argmax(y__2, 1))
        accuracy_2 = tf.reduce_mean(tf.cast(correct_prediction_2, "float"))

        # Score display.
        print("check accuracy %g" % accuracy_2.eval(
            {x_2: mnist.test.images, y__2: mnist.test.labels}, sess_2))

```

Reference1 :
https://github.com/miyosuda/TensorFlowAndroidMNIST/blob/master/trainer-script/expert.py  
Reference2 : [TesorFlow: Pythonで学習したデータをAndroidで実行](http://qiita.com/miyosuda/items/e53ad2efeed0ff040606 "TesorFlow: Pythonで学習したデータをAndroidで実行")


## How to convert automatically

<!-- 上記の記事のように_freeze_graph.py_は使おうとしたが、エラーが何度も出て、r12の新しいモデル形式への未対応やpython3のとき引数が増えたりするので面倒になり、使わないことにした。 -->

First, I tried to use "freeze_graph.py" as in [this article](http://qiita.com/YusukeSuzuki@github/items/476e599d84eb3d6d184d#3-%E3%83%81%E3%82%A7%E3%83%83%E3%82%AF%E3%83%9D%E3%82%A4%E3%83%B3%E3%83%88%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB%E3%82%92protocolbuffers%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB%E3%81%AB%E5%A4%89%E6%8F%9B%E3%81%99%E3%82%8B "TensorFlowで学習してモデルファイルを小さくしてコマンドラインアプリを作るシンプルな流れ").

However, it was troublesome as errors got out many times, not compatible with r12 's new model format, and arguments increased when it was python 3.

Therefore, I decided not to use it.


<!-- そのため、今回は convert_variables_to_constants()のみを使った。
モデルの学習後に、convert_variables_to_constants()でvariableからconstantへ変換後、プロトコルバッファとして書き出す。
　ただし、手動で変換していたときは、グラフの再構成するときに各ノードに名前をつけていたが、今回は学習を行ったグラフを変換するため、各ノードに名前をつけておく必要がある。この名前がC#上で読込むときに対応する。 -->

So, I used convert_variables_to_constants().

After model learning, convert from variable to constant with convert_variables_to_constants() and write it as a protocol buffer.

However, when manually converting, each node is given a name when reconstructing the graph, but this time it is necessary to name each node in order to convert the learned graph is there. Corresponds when this name is read on C #.

```py:
with g.as_default() as gr_def:
    x = tf.placeholder("float", shape=[None, 784], name="input")
    y_ = tf.placeholder("float", shape=[None, 10], name="labels")

    ...

    keep_prob = tf.placeholder("float", name="dropout")

    ...

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name="output")

    with tf.Session() as sess:

        ...

        # Make a graph with converting variables to constants.
        # Specify the name of the output node.
        converted_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['output'])
        # Output as a protocol buffer.
        tf.train.write_graph(converted_graph, export_dir, 'Auto_model.pb',  as_text=False)
```
Reference : [学習済みグラフをプロトコルバッファ形式で保存する](http://tyfkda.github.io/blog/2016/09/14/tensorflow-protobuf.html)


# model loading and inference

<!-- 手動での変換と、convert_variables_to_constants()を用いた変換では、モデルの実行方法が少し異なる。
　手動での変換では推論では使わないDropoutは入れてないが、convert_variables_to_constants()で変換したときDropoutのplaceholderはそのままなので、実行時に値を入れる必要がある。

以下に手動で変換したときに書き出したモデルのManual_model.pbの読込み及び推測のコードを示す。 -->

The manual conversion and the conversion using convert_variables_to_constants (), the model execution method differs slightly.

Manual conversion does not include "Dropout layer", which is not used in inference, but when transforming with convert_variables_to_constants (), the placeholder of Dropout remains unchanged, so you need to enter a value at run time.

The code for loading and being inference Manual_model.pb of the model outputed when manually converted below is shown below.


```c#:Sample.cs
using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using TensorFlow;
using System.IO;
using System.Collections.Generic;
using Learn.Mnist;
using System.Linq;

namespace SampleTest
{
	class MainClass
	{

		// Convert the image in filename to a Tensor suitable as input to the Inception model.
		static TFTensor CreateTensorFromImageFile(string file)
		{
			var contents = File.ReadAllBytes(file);

			// DecodeJpeg uses a scalar String-valued tensor as input.
			var tensor = TFTensor.CreateString(contents);

			TFGraph graph;
			TFOutput input, output;

			// Construct a graph to normalize the image
			ConstructGraphToNormalizeImage(out graph, out input, out output);

			// Execute that graph to normalize this one image
			using (var session = new TFSession(graph))
			{
				var normalized = session.Run(
						 inputs: new[] { input },
						 inputValues: new[] { tensor },
						 outputs: new[] { output });

				return normalized[0];
			}
		}

    // The inception model takes as input the image described by a Tensor in a very
		// specific normalized format (a particular image size, shape of the input tensor,
		// normalized pixel values etc.).
		//
		// This function constructs a graph of TensorFlow operations which takes as
		// input a JPEG-encoded string and returns a tensor suitable as input to the
		// inception model.
		static void ConstructGraphToNormalizeImage(out TFGraph graph, out TFOutput input, out TFOutput output)
		{
			// - The model was trained after with images scaled to 28x28 pixels.
			// - Image is monochrome, only one color is represented.
      // Since the pixel value convert the range from 0~255 to 0~1,
      // Mean = 255, Scale = 255
      // using [the conversion value = (pixel value - Mean) / Scale].

			const int W = 28;
			const int H = 28;
			const float Mean = 255;
			const float Scale = 255;
			const int channels = 1;

			graph = new TFGraph();
			input = graph.Placeholder(TFDataType.String);

			output = graph.Div(
				x: graph.Sub(
					x: graph.ResizeBilinear(
						images: graph.ExpandDims(
							input: graph.Cast(
								graph.DecodeJpeg(contents: input, channels: channels), DstT: TFDataType.Float),
							dim: graph.Const(0, "make_batch")),
						size: graph.Const(new int[] { W, H }, "size")),
					y: graph.Const(Mean, "mean")),
				y: graph.Const(Scale, "scale"));
		}
		// Load models created with python.
		void MNSIT_read_model()
		{
			var graph = new TFGraph();

			//var model = File.ReadAllBytes("tensorflow_inception_graph.pb");

			// Load serialized GraphDef from file.
			var model = File.ReadAllBytes("Manual_model.pb");
			graph.Import(model, "");

			using (var session = new TFSession(graph))
			{
				var labels = File.ReadAllLines("labels.txt");

				var file = "temp.jpg";

				// Execute inference on an image file.
				// For multiple images, session.Run () can be called (simultaneously) in a loop.
				// Alternatively, the model accepts a batch of image data as input, so you can batch the image.
				var tensor = CreateTensorFromImageFile(file);

				var runner = session.GetRunner();
				// Specify a graph of the learning model.
				// Register the name of the input / output tensor in session.
        // When reading manually converted models, .AddInput (graph ["dropout"] [0], 0.5f) does not use.
				runner.AddInput(graph["input"][0], tensor).Fetch(graph["output"][0]);
				var output = runner.Run();

				// output[0].Value（）is the background containing the accuracy of the label of each image in the "batch". The batch size was 1.
				var result = output[0];
				var rshape = result.Shape;
				if (result.NumDims != 2 || rshape[0] != 1)
				{
					var shape = "";
					foreach (var d in rshape)
					{
						shape += $"{d} ";
					}
					shape = shape.Trim();
					Console.WriteLine($"Error: expected to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape [{shape}]");
					Environment.Exit(1);
				}

				var bestIdx = 0;
				float best = 0;
        // Find and display numbers with most probability
				var probabilities = ((float[][])result.GetValue(true))[0];
				for (int i = 0; i < probabilities.Length; i++)
				{
					Console.WriteLine(probabilities[i]);
					if (probabilities[i] > best)
					{
						bestIdx = i;
						best = probabilities[i];
					}
				}
				Console.WriteLine($"{file} best match: [{bestIdx}] {best * 100.0}% {labels[bestIdx]}");
			}
		}
		public static void Main(string[] args)
		{
			Console.WriteLine(Environment.CurrentDirectory);
			Console.WriteLine("TensorFlow version: " + TFCore.Version);


			var t = new MainClass();
			t.MNSIT_read_model();
		}
	}
}
```

<!-- 以下のlabels.txtとtemp.jpg、作成した学習モデルを実行ファイルと同層に配置する。 -->
Below labels.txt and temp.jpg and the created learning model in the same layer as the executable file.

```:labels.txt
0
1
2
3
4
5
6
7
8
9
```

temp.jpg ↓  
![temp.jpg](https://qiita-image-store.s3.amazonaws.com/0/70879/310bbb5d-0374-85d4-58a2-b782c75b70f9.jpeg)




<!-- convert_variables_to_constants()で変換したときは、以下のようにコードを変更する。 -->
When converted with convert_variables_to_constants (), change the code as follows.

```
var model = File.ReadAllBytes("Manual_model.pb");
↓
var model = File.ReadAllBytes("Auto_model.pb");
```

```
runner.AddInput(graph["input"][0], tensor).Fetch(graph["output"][0]);
↓
runner.AddInput(graph["input"][0], tensor).AddInput(graph["dropout"][0], 0.5f).Fetch(graph["output"][0]);
```


<!-- 一応実行時の結果の画像を示す。 -->
It shows the image of the result at the time of execution.
<!-- Manual_model.pbの実行結果 -->
Execution result of Manual_model.pb  
<img width="571" alt="スクリーンショット 2017-06-07 14.49.23.png" src="https://qiita-image-store.s3.amazonaws.com/0/70879/63179e23-d582-d639-c2c6-4664077a5d2d.png">

<!-- Auto_model.pbの実行結果 -->
Execution result of Auto_model.pb  
<img width="571" alt="スクリーンショット 2017-06-07 14.48.37.png" src="https://qiita-image-store.s3.amazonaws.com/0/70879/514fc8f3-98e0-7ade-1957-88d9e9bc4bf4.png">



<!-- # 学習

C#上で学習をさせたくて、一からモデルを構築しようとしたが、最適化関数が一つしか見つからず、リファレンスもなくなってたので使い方が分からず詰んだ。また、python上でモデルをつくって読み込み後、グラフを再構築すればよいとも考えたが、チェックポイントファイルから最適化器のデータを取り除いてあるので、pythonので書いてもC#上では使えない。
今後の課題としては、最適化関数の使い方を知ってC#上で学習をさせたい。 -->
