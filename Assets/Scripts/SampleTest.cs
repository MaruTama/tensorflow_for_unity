using System.Collections;
using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using TensorFlow;
using System.IO;
using System.Collections.Generic;
using Learn.Mnist;
using System.Linq;
using UnityEngine;

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

		//開始モデルは、非常に特定の正規化されたフォーマット（特定の画像サイズ、入力テンソルの形状、正規化されたピクセル値など）
		//でテンソルによって記述された画像を入力として取ります。
		//このファンクションは、入力としてJPEGでエンコードされた文字列を取り込み、
		//入力モデルとしての入力として適したテンソルを戻すTensorFlow操作のグラフを作成します。
		static void ConstructGraphToNormalizeImage(out TFGraph graph, out TFOutput input, out TFOutput output)
		{
			// - モデルは28x28ピクセルにスケーリングされた画像で訓練されました。
			// - モノクロなので表される色は1色のみ。（値 - 平均）/ スケールを使用してfloatに変換して使用する。
            // 画素値を0-255 から 0-1 の範囲にするので、変換値 = (Mean - 画素値) / Scale の式から,
            // Mean = 255, Scale = 255 となる。

            const int W = 28;
            const int H = 28;
            const float Mean = 255;
            const float Scale = 255;
            const int channels = 1;

            graph = new TFGraph();
            input = graph.Placeholder(TFDataType.String);

            output = graph.Div(
                x: graph.Sub(
                  x: graph.Const(Mean, "mean"),
                    y: graph.ResizeBilinear(
                        images: graph.ExpandDims(
                            input: graph.Cast(graph.DecodeJpeg(contents: input, channels: channels), DstT: TFDataType.Float),
                            dim: graph.Const(0, "make_batch")),
                        size: graph.Const(new int[] { W, H }, "size"))),
                y: graph.Const(Scale, "scale"));
		}
		// pythonで作成したモデルの読込を行う
		public void MNSIT_read_model()
		{
			var graph = new TFGraph();

			//var model = File.ReadAllBytes("tensorflow_inception_graph.pb");

			// シリアル化されたGraphDefをファイルからロードします。
			var model = File.ReadAllBytes(Application.dataPath + "/models/Auto_model.pb");
			graph.Import(model, "");

			using (var session = new TFSession(graph))
			{
				var labels = File.ReadAllLines(Application.dataPath + "/models/labels.txt");

				var file = Application.dataPath + "/saveImage.jpg";

				//画像ファイルに対して推論を実行する
				//複数のイメージの場合、session.Run（）はループで（同時に）呼び出すことができます。 
				//あるいは、モデルが画像データのバッチを入力として受け入れるので、画像をバッチ処理することができる。
				var tensor = CreateTensorFromImageFile(file);

				var runner = session.GetRunner();
				// 学習モデルのグラフを指定する。
				// 入出力テンソルの名前をsessionに登録する
				// 手動で変換したモデルの読込のときは、.AddInput(graph["dropout"][0], 0.5f)はいらない。
				runner.AddInput(graph["input"][0], tensor).AddInput(graph["dropout"][0], 0.5f).Fetch(graph["output"][0]);
				var output = runner.Run();

				// output[0].Value（）は、「バッチ」内の各画像のラベルの確率を含むベクトルです。 バッチサイズは1であった。
				//最も可能性の高いラベルインデックスを見つけます。
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
					Debug.Log($"Error: expected to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape [{shape}]");
					Environment.Exit(1);
				}

				var bestIdx = 0;
				float best = 0;
				// 尤も確率が高いものを調べて表示する
				var probabilities = ((float[][])result.GetValue(true))[0];
				for (int i = 0; i < probabilities.Length; i++)
				{
					if (probabilities[i] > best)
					{
						bestIdx = i;
						best = probabilities[i];
					}
				}
				Debug.Log($"{file} best match: [{bestIdx}] {best * 100.0}% {labels[bestIdx]}");
			}
		}
	}
}