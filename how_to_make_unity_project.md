# Introduction of unity
<!-- 文法的にC#6でないと動かないと所があるので、ここからから最新版(Unity 2017.1以上)を落としておく。 -->
Since there is a code unless it moves grammatically in C # 6, please download the latest version (Unity 2017.1 or higher) from [here](https://store.unity.com/download?ref=personal).

# Creating and preparing projects
<!-- プロジェクトを作成後は、Assetsの下にDllsというフォルダを作り、TensorFlowSharp\TensorFlowSharp\bin\Debug の下のTensorFlowSharp.dllとSystem.ValueTuple.dllをコピーしておく。 -->
After creating the project, create a folder called Dlls under Assets and copy TensorFlowSharp.dll and System.ValueTuple.dll under TensorFlowSharp/TensorFlowSharp/bin/Debug.

<img width="1001" alt="スクリーンショット 2017-06-09 00.33.19.png" src="https://qiita-image-store.s3.amazonaws.com/0/70879/81b4b508-6005-b57e-16f2-de2954454c70.png">


Change Scripting Runtime Version to Experiental (.NET 4.6 Equvalent)
<img width="1440" alt="スクリーンショット 2017-06-09 00.34.51.png" src="https://qiita-image-store.s3.amazonaws.com/0/70879/837ad798-ce4b-86ed-ca4f-ffb8eb570cba.png"><img width="1440" alt="スクリーンショット 2017-06-09 00.35.32.png" src="https://qiita-image-store.s3.amazonaws.com/0/70879/a06b58ac-2f52-176d-74d8-195dc9ede094.png">

<!-- 変更後、unityの再起動する。 -->
After change, restart unification.

# Create a pad for entering numbers

Create paint.cs.
<img width="1440" alt="スクリーンショット 2017-06-09 00.41.00.png" src="https://qiita-image-store.s3.amazonaws.com/0/70879/838e5367-dc45-79e3-13f2-b7a8d78d687c.png">
<img width="1440" alt="スクリーンショット 2017-06-09 00.42.56.png" src="https://qiita-image-store.s3.amazonaws.com/0/70879/90700a55-3bae-0324-6966-be12bfd1b807.png">

```csharp:paint.cs
// Reference : http://nn-hokuson.hatenablog.com/entry/2016/12/08/200133
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class paint : MonoBehaviour {

	Texture2D drawTexture;
	Color[] buffer;

	// Use this for initialization
	void Start () {
		Texture2D mainTexture = (Texture2D)GetComponent<Renderer>().material.mainTexture;
		Color[] pixels = mainTexture.GetPixels();

		buffer = new Color[pixels.Length];
		pixels.CopyTo(buffer, 0);

		drawTexture = new Texture2D(mainTexture.width, mainTexture.height, TextureFormat.RGBA32, false);
		drawTexture.filterMode = FilterMode.Point;
	}
	// Change the stroke width of the brush.
	public void Draw(Vector2 p)
	{
		//buffer.SetValue(Color.black, (int)p.x + 256 * (int)p.y);

		// boldface.
		for (int x = 0; x < 256; x++)
		{
			for (int y = 0; y < 256; y++)
			{
				if ((p - new Vector2(x, y)).magnitude < 5)
				{
					buffer.SetValue(Color.black, x + 256 * y);
				}
			}
		}
	}

	// Every frame, check all pixels on the texture, fill black if the distance from the coordinates on which the mouse is on is less than 8.
	void Update () {
		if (Input.GetMouseButton(0))
		{
			Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
			RaycastHit hit;
			if (Physics.Raycast(ray, out hit, 100.0f))
			{
				Draw(hit.textureCoord * 256);
			}

			drawTexture.SetPixels(buffer);
			drawTexture.Apply();
			GetComponent<Renderer>().material.mainTexture = drawTexture;
		}
	}
  // Save texture as jpg.
	public void SaveTexture()
	{
		byte[] data = drawTexture.EncodeToJPG();
		File.WriteAllBytes(Application.dataPath + "/saveImage.jpg", data);
	}
}

```


<!-- 入力パッドとなるplaneを追加する。 -->
Add a plane to be an input pad.
<img width="1440" alt="スクリーンショット 2017-06-09 00.50.24.png" src="https://qiita-image-store.s3.amazonaws.com/0/70879/59ca95a9-ad74-fa6c-4b66-8f2f11cb9323.png">
<img width="1440" alt="スクリーンショット 2017-06-09 00.50.43.png" src="https://qiita-image-store.s3.amazonaws.com/0/70879/f14054d5-a2c8-eb78-a991-1d8702e38e05.png">

<!-- planeの位置を以下のように変更する。 -->
Change the position of plane as follows.
<img width="326" alt="スクリーンショット 2017-06-09 00.52.54.png" src="https://qiita-image-store.s3.amazonaws.com/0/70879/f781067e-f197-b144-0cf3-0f040ad18b62.png">

<!-- Main cameraを操作する。
x:1, y:5, z:-9 がちょうどいい -->
Operate Main camera.
X: 1, y: 5, z: -9 is just right.
<img width="1440" alt="スクリーンショット 2017-06-09 00.56.28.png" src="https://qiita-image-store.s3.amazonaws.com/0/70879/68045149-4368-494c-167d-44ea64a79deb.png">

<!-- ↓テクスチャ画像 -->
↓Texture image
***
![texture.jpg](https://qiita-image-store.s3.amazonaws.com/0/70879/b3ae40fb-8fd0-f02f-26b9-7646f90bdd82.jpeg)
***

<!-- Assetsの直下にtexture.jpgを配置する。 -->
Place texture.jpg directly under Assets.
<img width="1440" alt="スクリーンショット 2017-06-09 00.59.39.png" src="https://qiita-image-store.s3.amazonaws.com/0/70879/0c36a6c7-2409-0963-7d56-24d2cff4e8ae.png">

<!-- texture.jpgをplaneにドラッグアンドドロップして、アタッチする。 -->
Drag and drop the texture.jpg onto the plane and attach it.
<img width="1440" alt="スクリーンショット 2017-06-09 01.01.12.png" src="https://qiita-image-store.s3.amazonaws.com/0/70879/201e335b-39c5-0cc6-66ee-8662d2bd594d.png">

<!-- Albedoをクリックすると、texture.jpgがアタッチされていることが確認できる -->
By clicking Albedo you can see that texture.jpg is attached.
<img width="1440" alt="スクリーンショット 2017-06-09 01.02.00.png" src="https://qiita-image-store.s3.amazonaws.com/0/70879/d0dc6efb-6957-fdf3-84de-85ae3f997d2f.png">

<!-- Shader をStandard から　Unit->Textureに変更する -->
Change Shader from Standard to Unit -> Texture.
<img width="1440" alt="スクリーンショット 2017-06-09 01.07.07.png" src="https://qiita-image-store.s3.amazonaws.com/0/70879/c199115a-de84-75fc-e179-a8b5b4fdb625.png">


<!-- 次にtexture.jpgを選択して、Read/Write Enabledにチェックを入れる。 -->
Next select texture.jpg and check [Read / Write Enabled].
<img width="1440" alt="スクリーンショット 2017-06-09 01.04.37.png" src="https://qiita-image-store.s3.amazonaws.com/0/70879/7871455d-612e-1b98-2dc5-62dc59fac72d.png">

<!-- planeにpaint.csをアタッチする -->
Attach paint.cs to the plane.
<img width="1440" alt="スクリーンショット 2017-06-09 1.10.16.png" src="https://qiita-image-store.s3.amazonaws.com/0/70879/4e643a35-c71b-e12f-4474-29dd5d37204d.png">

<!-- すると、planeに書き込めるようになる -->
Then you can write into plane
<img width="1440" alt="スクリーンショット 2017-06-09 01.13.07.png" src="https://qiita-image-store.s3.amazonaws.com/0/70879/a5110732-04fc-4d01-cbbf-8795d4ffc2be.png">

# Create a button to save and recognize the input numeric image

<!-- 次はボタンを追加する。 -->
Next, add a button.
<img width="1440" alt="スクリーンショット 2017-06-09 01.13.57.png" src="https://qiita-image-store.s3.amazonaws.com/0/70879/347ed7fc-229a-ff38-0112-0bf3f3060146.png">


<!-- ボタンのtextをRrecognitionにする -->
Set the text of the button to Rrecognition.
<img width="1440" alt="スクリーンショット 2017-06-09 01.15.40.png" src="https://qiita-image-store.s3.amazonaws.com/0/70879/7c9e7a97-9a0c-e073-f672-f7164fedb7a6.png">


<!-- ボタンのonclickの設定を行う。 -->
Set onclick of the button.
<!-- Inspectorの on Click()の[+]を押して、リスナーのメソッドを追加する。paint.SaveTexture()を追加する。 -->
Press On Click () [+] on Inspector to add the listener method. Add paint.SaveTexture().
<img width="1440" alt="スクリーンショット 2017-06-09 01.16.15.png" src="https://qiita-image-store.s3.amazonaws.com/0/70879/a96f53dd-691a-cd4a-cabc-21f7db5c968c.png">

<!-- paint.csはplaneにアタッチしてあるので、planeを選択する。 -->
Since paint.cs is attached to plane, select plane.
<img width="1440" alt="スクリーンショット 2017-06-09 01.16.27.png" src="https://qiita-image-store.s3.amazonaws.com/0/70879/93d7455e-7366-41d9-3d35-83c25333e1e3.png">

<!-- メソッドは paint -> SaveTexture() を選択する。 -->
The method selects paint -> SaveTexture().
<img width="1440" alt="スクリーンショット 2017-06-09 01.16.44.png" src="https://qiita-image-store.s3.amazonaws.com/0/70879/6d90e2c1-0219-9492-144a-88376c5922d7.png">

<!-- そして、planeに何かを書き込み後、ボタンを押すとAssetsの直下に画像が保存される。 -->
Then, after writing something to plane, press the button and the image is saved just under Assets.

# Prepare for recognition of numbers.
<!-- paint.csにコードを加筆する。 -->
Add code to paint.cs.

```csharp:paint.cs
public void SaveTexture()
{
	・・・
	Debug.Log("Environment.CurrentDirectory");
	Debug.Log("TensorFlow version: " + TFCore.Version);

	var t = new SampleTest.MainClass();
	t.MNSIT_read_model();
}
```

<!-- このような構成にして、スクリプトや学習モデルを配置する。学習モデルは、[前回作成](http://qiita.com/Tama_maru/items/6e50edfd8f8dea184d18#_reference-6803ca58c52b7c814a95 "C#でTensorFlowのCNNを動かす。")したものを使用する。 -->

With such a configuration, we arrange scripts and learning models.
Use the model created [here.](./how_to_make_models.md)

Assets
　├── Scripts
　│　　   ├── DataConverter.cs
　│ 　　  ├── Datasets
　│  　　 │  　　 ├── Helper.cs
　│   　　│  　　 └── MNIST.cs
　│   　　├── SampleTest.cs
　│  　　 └── paint.cs
　└── models
  　　　　  ├── Auto_model.pb
  　　　　  └── labels.txt

<!-- TensorFlowSharpのLearnの中のDataConverter.csとDatasets/Helper.cs, Datasets/MNIST.csをScriptsの下に配置する。 -->

Position DataConverter.cs and Datasets/Helper.cs, Datasets/MNIST.cs in Learn of TensorFlowSharp under Scripts.


```csharp:Sample.cs
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
		// Load models created with python
		public void MNSIT_read_model()
		{
			var graph = new TFGraph();

			//var model = File.ReadAllBytes("tensorflow_inception_graph.pb");

			// Load serialized GraphDef from file.
			var model = File.ReadAllBytes(Application.dataPath + "/models/Auto_model.pb");
			graph.Import(model, "");

			using (var session = new TFSession(graph))
			{
				var labels = File.ReadAllLines(Application.dataPath + "/models/labels.txt");

				var file = Application.dataPath + "/saveImage.jpg";

				// Execute inference on an image file.
				// For multiple images, session.Run () can be called (simultaneously) in a loop.
				// Alternatively, the model accepts a batch of image data as input, so you can batch the image.
				var tensor = CreateTensorFromImageFile(file);

				var runner = session.GetRunner();
				// Specify a graph of the learning model.
				// Register the name of the input / output tensor in session.
        // When reading manually converted models, .AddInput (graph ["dropout"] [0], 0.5f) does not use.
				runner.AddInput(graph["input"][0], tensor).AddInput(graph["dropout"][0], 0.5f).Fetch(graph["output"][0]);
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
					Debug.Log($"Error: expected to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape [{shape}]");
					Environment.Exit(1);
				}

				var bestIdx = 0;
				float best = 0;
				// Find and display numbers with most probability
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
```

# Execute

<!-- △の90°傾けたボタンを押し、入力パッドに数字を入力後、Rrecognitionボタンを押すと、結果が表示される。 -->
After pushing the button tilted by 90 degrees of Δ, entering the number on the input pad, press the Rrecognition button and the result is displayed.

<!-- 実行画面 -->
Execution screen
<img width="1440" alt="スクリーンショット 2017-06-11 00.34.16.png" src="https://qiita-image-store.s3.amazonaws.com/0/70879/1841c8c5-c50d-57c0-1b69-7bad76593fe7.png">
<img width="1440" alt="スクリーンショット 2017-06-11 00.35.30.png" src="https://qiita-image-store.s3.amazonaws.com/0/70879/7d3ad9c6-5a12-0201-addf-70a0d0030ad9.png">

reference : 
[Unityでテクスチャにお絵描きしよう](
http://nn-hokuson.hatenablog.com/entry/2016/12/08/200133 "Unityでテクスチャにお絵描きしよう")
