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

### 单元测试指导(待补充)

1. 准备好端侧依赖的so， 请参考<https://www.mindspore.cn/`federated/docs/zh-CN/r1.6/deploy_federated_client.html#id7>
   ，并更新settings.gradle的test stask中的LD_LIBRARY_PATH
2. 准备号端侧依赖的模型jar包quick_start_flclient.jar，并更新settings.gradle的test stask中的MS_FL_UT_BASE_PATH
3. 可以直接在idea中执行test进行ut测试
4. 命令行ut测试需要先生成flUTJarX86包，具体的执行指令类似如下：

> LD_LIBRARY_PATH=${lite_x86_lib_path} java -javaagent:build/libs/jarX86UT/jmockit-1.49.jar -cp build/libs/jarX86UT/mindspore-lite-java-flclient.jar org.junit.runner.JUnitCore com.mindspore.flclient.FLFrameUTRun

### 端侧开发指南(待补充完善)

### 原MindSpore master分支代码需要清理，避免端侧apk依赖多个联邦 /ST需要下线避免master代码清理后CI失败
