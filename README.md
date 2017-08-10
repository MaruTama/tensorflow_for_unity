# Reference Notes
[C#でTensorFlowを動かす。](http://qiita.com/Tama_maru/items/9ce0e7f88ae4f00cb18f "C#でTensorFlowを動かす。")  
[C#でTensorFlowのCNNを動かす。](http://qiita.com/Tama_maru/items/6e50edfd8f8dea184d18 "C#でTensorFlowのCNNを動かす。")  
[Unity上でTensorFlowのCNNを動かす。](http://qiita.com/Tama_maru/items/25346d8cf3a142dd6aaa "Unity上でTensorFlowのCNNを動かす。")  

# Quick start
1. To set up the development environment and build TensorFlowSharp, please refer to the following.
2. [Create a CNN model](./how_to_make_models.md)
3. Create a project with unity (Unspecified)

# How to make development environment

## Clone( or Download) TensorFlowSharp
https://github.com/migueldeicaza/TensorFlowSharp  

Link of dll file  
https://github.com/migueldeicaza/TensorFlowSharp#working-on-tensorflowsharp

## ubuntu 16.04
<!-- ### Version
monoDevelop 7.0.1 (build 24) -->
### Install flatpak
Reference: [Install MonoDevelop preview via FlatPak](http://www.monodevelop.com/download/linux/ "Install MonoDevelop preview via FlatPak")
```:txt
$ sudo add-apt-repository ppa:alexlarsson/flatpak
$ sudo apt update
$ sudo apt install flatpak
```

### Install mono Dev
```:txt
$ flatpak install --user --from https://download.mono-project.com/repo/monodevelop.flatpakref
# Check if it can be started
$ flatpak run com.xamarin.MonoDevelop
```

### Add library to system
Reference1 : [LD_LIBRARY_PATH を設定しても反映されないことがある](https://kokufu.blogspot.jp/2016/01/ldlibrarypath.html "LD_LIBRARY_PATH を設定しても反映されないことがある")  
Reference2 : [Interop with Native Libraries](http://www.mono-project.com/docs/advanced/pinvoke/ "Interop with Native Libraries")
```:txt
$ cd Downloads
$ wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.1.0.tar.gz
$ cd /usr/local
$ sudo tar xzvf ~/Downloads/libtensorflow-cpu-linux-x86_64-1.1.0.tar.gz
#Configuration a path
$ cd /etc/ld.so.conf.d
$ sudo vim lib_path.conf
#Save the string below(A directory with Libtensorflow.so)
/usr/local/lib
#Rebuild /etc/ld.so.cache
$ sudo ldconfig
$ ldconfig -p > ~/out.txt
#Check if the library has been added
$ find ~/ -type f -name "out.txt" | xargs grep 'libtensorflow.so'
```

### Build the solution
Open the solution (.sin) right under the clone (or download) directory with MonoDevelop.  
![Screenshot from 2017-03-17 23-33-57.png](https://qiita-image-store.s3.amazonaws.com/0/70879/eb95faf2-839b-f075-d082-ff96760cb067.png)
Next, rebuild the solution with Build -> Rebuild All.  
![Screenshot from 2017-03-17 23-34-52.png](https://qiita-image-store.s3.amazonaws.com/0/70879/28353ddb-4aae-c0d1-bb24-958c4c0777d6.png)
I move SampleTest. Select SampleTest and press Build Project.  
![Screenshot from 2017-03-18 00-55-02.png](https://qiita-image-store.s3.amazonaws.com/0/70879/fd556ade-8d0a-ee75-8898-79f0316710af.png)

Start the executable file on the console.  

### Install mono-complete
```:txt
$ sudo apt-get install mono-complete
$ cd ~/workspace/c_sharp/TensorFlowSharp/SampleTest/bin/Debug
$ mono SampleTest.exe
```

**[Additional notes]**  
Reference:[Nuget and “Unable to load the service index”](https://stackoverflow.com/questions/44688192/nuget-and-unable-to-load-the-service-index "Nuget and “Unable to load the service index”")

I got an error as below. (on 7.29 2017)  
```:txt
[nuget.org] Unable to load the service index for source https://api.nuget.org/v3/index.json. An error occurred while sending the request
Error: SecureChannelFailure (Object reference not set an instance of an object) Object reference not set to an instance of an object
```

How to deal  
run command below from terminal.  
```:txt
export MONO_TLS_PROVIDER=legacy
```

So the build will pass.  

## Mac
### Version
macOS Sierra 10.12.6  
Visual Studio 7.0.1 (build 24)  
unity 2017.1.0f3  
### Install VisualStudio
download and install the installer from the below link  
https://www.visualstudio.com/vs/visual-studio-mac/

### Add library to system
```:txt
$ cd ~/Downloads
$ curl -O https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-x86_64-1.1.0.tar.gz
$ cd /usr/local
$ sudo tar zxvf ~/Downloads/libtensorflow-cpu-darwin-x86_64-1.1.0.tar.gz
$ cd lib
$ sudo mv libtensorflow.so libtensorflow.dylib
#Check if the library has been added
$ find /usr/local/ -type f -name "libtensorflow.dylib"
```

### Build the solution
Open the solution (.sin) right under the clone (or download) directory with VisualStudio.  
Next, rebuild the solution with Build -> Rebuild All.  
<img width="1440" alt="スクリーンショット 2017-07-31 00.40.19.png" src="https://qiita-image-store.s3.amazonaws.com/0/70879/a8f54292-9b27-379e-c8c0-6ef0013106ee.png">




## Windows

### Version
Microsoft Visual Studio Community 2017  
Version 15.2 (26430.16) Releasec  
Microsoft .NET Framework  
Version 4.6.01586  

### Add library to system
download dll file from the below link  
https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-windows-x86_64-1.2.0-rc0.zip

After expansion, rename from tensorflow.dll to libtensorflow.dll and move in C:\Windows\System32.  
<img width="1440" alt="スクリーンショット 2017-05-15 23.50.34.png" src="https://qiita-image-store.s3.amazonaws.com/0/70879/848ef169-320f-827c-78e6-533e93c2e646.png">

### Install VisualStudio
download and install the installer from the below link.  
https://www.visualstudio.com/downloads/  
Follow the instructions to install the tool.  

### Build the solution
Open the solution (.sin) right under the clone (or download) directory with VisualStudio.  
Next, rebuild the solution with Build -> Rebuild All.
<img width="1440" alt="スクリーンショット 2017-07-31 00.52.47.png" src="https://qiita-image-store.s3.amazonaws.com/0/70879/7da5c67d-c5fb-d79d-c122-b53a581635c7.png">



<!--
## how to make models

##unity
ソリューションのリビルド
Macだと普通に最新のものをgithubから落としても動く


2つコピーしないと、エラーになる。
System.ValueTuple.dll
TensorFlowSharp.dll
Unexpected Error Could not load file or assembly 'System.ValueTuple, Version=4.0.1.0, Culture=neutral, PublicKeyToken=cc7b13ffcd2ddd51' or one of its dependencies. The system cannot find the file specified. -->
