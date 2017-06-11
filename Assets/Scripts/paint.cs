// 参考
// http://nn-hokuson.hatenablog.com/entry/2016/12/08/200133
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using TensorFlow;

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
	// ブラシの太さを変える
	public void Draw(Vector2 p)
	{
		//buffer.SetValue(Color.black, (int)p.x + 256 * (int)p.y);

		//太字
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

	// 毎フレーム、テクスチャ上のすべてのピクセルをチェックして、マウスが乗っている座標からの距離が8以下なら黒く塗りつぶします。
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

	public void SaveTexture()
	{
		byte[] data = drawTexture.EncodeToJPG();

		File.WriteAllBytes(Application.dataPath + "/saveImage.jpg", data);

		Debug.Log("Environment.CurrentDirectory");
		Debug.Log("TensorFlow version: " + TFCore.Version);

		var t = new SampleTest.MainClass();
		t.MNSIT_read_model();
	}
}
