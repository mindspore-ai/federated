## 联邦学习端侧编译指导

本章节介绍如何完成MindSpore联邦学习的端侧编译，当前联邦学习端侧仅提供linux上的编译指导，其他系统暂不支持。

### 环境要求

- 系统环境：Linux x86_64，推荐使用Ubuntu 18.04.02LTS
- C++编译依赖， 用于编译C++版本的flat-buffer
    - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0
    - [CMake](https://cmake.org/download/) >= 3.18.3

- Java API模块的编译依赖
    - [Gradle](https://gradle.org/releases/)  6.6.1
        - 配置环境变量：`export GRADLE_HOME=GRADLE路径`和`export GRADLE_USER_HOME=GRADLE路径`
        - 将bin目录添加到PATH中：`export PATH=${GRADLE_HOME}/bin:$PATH`
        - 建议采用[gradle-6.6.1-complete](https://gradle.org/next-steps/?version=6.6.1&format=all)版本，配置其他版本gradle将会采用gradle
          wrapper机制自动下载`gradle-6.6.1-complete`。
    - [Maven](https://archive.apache.org/dist/maven/maven-3/) >= 3.3.1
        - 配置环境变量：`export MAVEN_HOME=MAVEN路径`
        - 将bin目录添加到PATH中：`export PATH=${MAVEN_HOME}/bin:$PATH`
    - [OpenJDK](https://openjdk.java.net/install/) 1.8 到 1.15
        - 配置环境变量：`export JAVA_HOME=JDK路径`
        - 将bin目录添加到PATH中：`export PATH=${JAVA_HOME}/bin:$PATH`

### 编译选项

联邦学习device_client目录下的`cli_build.sh`脚本用于联邦学习端侧的编译。

#### `cli_build.sh`的参数使用说明

| 参数  | 参数说明         | 取值范围   | 默认值          |
|-----|--------------|--------|--------------|
| -p  | 依赖外部包的下载存放路径 | 字符串    | 工程目录下创建third |
| -c  | 是否复用之前下载的依赖包 | on、off | on           |

### 编译示例

首先，在进行编译之前，需从gitee代码仓下载源码。

```bash
git clone https://gitee.com/mindspore/federated.git ./
```

然后进入目录mindspore_federated/device_client，执行如下命令：

```bash
bash cli_build.sh
```

### 单元测试指导
#### gradle工具执行单元测试
- 将工程外部依赖包(编译时自动下载)路径下的mindspore-lite-$version-linux-x64/runtime/lib和$third/mindspore-lite-$version-linux-x64/runtime/third_party/libjpeg-turbo/libs
添加到LD_LIBRARY_PATH
  >export LD_LIBRARY_PATH=$third/mindspore-lite-$version-linux-x64/runtime/lib:$third/mindspore-lite-$version-linux-x64/runtime/third_party/libjpeg-turbo/libs:${LD_LIBRARY_PATH}
- 准备好单元测试数据，并设置环境变量MS_FL_UT_BASE_PATH为单元测试数据目录, 单元测试数据位于内部仓。
  >export MS_FL_UT_BASE_PATH=$UT_DATA
- 将$MS_FL_UT_BASE_PATH/test_data/jar目录下的quick_start_flclient.jar，flclient_models.jar 复制到libs目录
- 执行`gradle test` 完成单元测试执行

#### 命令行执行单元测试
- 将工程外部依赖包(编译时自动下载)路径下的mindspore-lite-$version-linux-x64/runtime/lib和$third/mindspore-lite-$version-linux-x64/runtime/third_party/libjpeg-turbo/libs
  添加到LD_LIBRARY_PATH
  >export LD_LIBRARY_PATH=$third/mindspore-lite-$version-linux-x64/runtime/lib:$third/mindspore-lite-$version-linux-x64/runtime/third_party/libjpeg-turbo/libs:${LD_LIBRARY_PATH}
- 准备好单元测试数据，并设置环境变量MS_FL_UT_BASE_PATH为单元测试数据目录, 单元测试数据位于内部仓。
  >export MS_FL_UT_BASE_PATH=$UT_DATA
- 将$MS_FL_UT_BASE_PATH/test_data/jar目录下的quick_start_flclient.jar，flclient_models.jar 复制到libs目录
- 执行`grade flUTJarX86` 生成单元测试包， 然后执行如下指令：
  > java -javaagent:build/libs/jarX86UT/jmockit-1.49.jar -cp build/libs/jarX86UT/mindspore-lite-java-flclient.jar org.junit.runner.JUnitCore com.mindspore.flclient.FLFrameUTRun
  > 
  > java -javaagent:build/libs/jarX86UT/jmockit-1.49.jar -cp build/libs/jarX86UT/mindspore-lite-java-flclient.jar org.junit.runner.JUnitCore com.mindspore.flclient.FLFrameUTInfer

