# 一、你的角色

你是一名资深 NVIDIA DeepStream、GStreamer、TensorRT、CUDA、Python 和计算机视觉工程师。

你现在接手的是一个已经完成基础环境搭建、已经能够运行的 NVIDIA DeepStream 视频分析工程。

不要重新创建项目。
不要重新搭建 Docker。
不要重新设计整个架构。

你的任务是：

直接读取并分析当前 VSCode 工程，在现有代码基础上完成本次优化、修改代码、运行测试、排查错误，直到当前测试视频可以正常运行并满足本提示词要求。

本次不是分阶段交付。

请一次性完成：

读取工程
→ 分析现状
→ 修改
→ 运行
→ 调试
→ 修复
→ 再运行
→ 验证结果

下一次与我的沟通默认是：

“我在实际测试中发现新的问题，需要继续优化。”

不要等我逐阶段确认。

==================================================
# 二、当前运行环境
==================================================

开发环境：

Windows 11
RTX 4090
VSCode
Codex
Docker Desktop
NVIDIA GPU Docker环境

当前工程已经完成 DeepStream Docker 环境搭建。

测试输入：

本地视频文件。

生产环境：

Ubuntu Linux
NVIDIA GPU

目标生产环境主要考虑：

2 × NVIDIA L20

目标视频并发：

128路

目标分析频率：

约5 FPS

因此：

性能非常重要。

绝对不要为了实现截图逻辑破坏 DeepStream Pipeline 的吞吐能力。

==================================================
# 三、本次核心目标
==================================================

本次只重点解决以下问题：

1. 确认当前人形检测是否真正使用：
   PeopleNet + NvDCF

2. 如果不是：
   修改为 PeopleNet + NvDCF。

3. 保证人员 Track ID 尽可能稳定。

4. 同一个 Track ID 生命周期内：
   持续寻找更清晰、更正面、更大的最佳人脸。

5. 视频前面即使出现模糊脸、侧脸，
   后面出现清晰正脸时，
   必须能够用后面的清晰正脸替换前面的差图。

6. 但是：
   图片质量绝对不能成为是否记录事件的条件。

7. 即使全过程没有高质量人脸：
   也必须保存证据。

8. 如果检测到了人脸：
   最终保存“最佳人脸所在帧对应的人物上半身”。

9. 如果完全没有检测到人脸：
   保存最佳 Person 证据。

10. 每一个有效 Person Track：
    必须至少留下1张证据图片。

核心原则：

ZERO MISS FIRST
QUALITY BEST SECOND

即：

零漏报优先
质量择优其次。

==================================================
# 四、首先检查现有工程，不要凭假设修改
==================================================

请首先自行搜索当前工程。

重点查找：

DeepStream Pipeline
PGIE配置
nvinfer配置
PeopleNet配置
nvtracker配置
NvDCF配置
osd_sink_pad_buffer_probe
NvDsObjectMeta解析
人脸检测代码
AdaFace代码
track状态代码
截图保存代码
数据库代码
config.yaml
config.py
Dockerfile
docker-compose.yml

先分析当前实际实现。

不要因为文件名叫：

peoplenet
nvdcf

就认为它真的生效。

必须从：

Pipeline创建代码
实际nvinfer配置
模型路径
tracker low-level library
tracker config
启动日志

确认真正运行的组件。

==================================================
# 五、人形检测必须统一为 PeopleNet
==================================================

当前默认 Person Detector 必须为：

NVIDIA PeopleNet

如果当前不是：

直接修改。

DeepStream Pipeline 原则：

source
↓
decoder
↓
nvstreammux
↓
PeopleNet / nvinfer
↓
NvDCF / nvtracker
↓
Face / AdaFace / Behavior
↓
OSD / Sink

不要继续使用：

OpenCV + YOLO

作为主 Person Detector。

如果当前工程已经保留 YOLO Detector：

不要强行删除。

允许保留：

person_detector.type

配置切换能力。

例如：

person_detector:
  type: peoplenet

未来允许：

yolov11

但是当前默认：

必须是 peoplenet。

==================================================
# 六、Tracker必须确认使用 NvDCF
==================================================

当前默认 Tracker：

NvDCF

不要默认使用：

ByteTrack
NvSORT
IOU Tracker
KLT

检查：

nvtracker

实际加载的：

low-level tracker library
tracker config yaml

确认：

NvDCF真正生效。

不要仅根据配置文件名称判断。

启动时必须打印：

[DETECTOR] PeopleNet
[TRACKER] NvDCF
[TRACKER_CONFIG] xxx.yml

方便人工确认。

==================================================
# 七、不要修改现有 Track ID 业务语义
==================================================

当前业务后续还有：

报警
人员事件
后台推送
轨迹记录

这些功能依赖 Track ID。

因此：

不要随意改变现有业务层 Track ID 定义。

DeepStream NvDCF：

object_id

作为基础 Track ID。

如果现有工程：

对 object_id 又做了一层业务映射，

必须先分析为什么存在。

不能直接删除。

本次重点：

提高 Track 稳定性，

而不是重新定义 Track ID。

==================================================
# 八、PeopleNet class_id必须从实际模型配置确认
==================================================

禁止直接硬编码：

0 = Person
1 = Bag
2 = Face

必须检查：

PeopleNet当前实际版本
labels
nvinfer config
模型输出

确认 class_id。

然后统一进入配置：

people_classes:
  person: xxx
  bag: xxx
  face: xxx

如果当前 PeopleNet 不负责真正的人脸检测：

不要因为存在 Face class 就替换当前专业人脸检测模块。

AdaFace前面的人脸检测仍然优先使用现有专业 Face Detector。

==================================================
# 九、最重要原则：零漏报优先
==================================================

本系统的第一业务原则：

只要 PeopleNet + NvDCF 产生一个有效 Person Track，

该 Track 最终必须至少产生一张证据图片。

图片质量：

只能决定：

“保存哪一张更好。”

绝对不能决定：

“这个人保存还是不保存。”

禁止：

质量低
→ candidate被全部过滤
→ Track结束
→ 什么都没有保存

这是严重业务错误。

==================================================
# 十、每个Track维护4类候选
==================================================

保持设计简单。

不要创建复杂状态机。

建议每个 Track 只维护：

TrackEvidenceState

内部核心状态：

1. person_fallback
2. best_person
3. face_fallback
4. best_face

含义：

--------------------------------
person_fallback
--------------------------------

第一次确认 Person Track 时：

立即保留一张最基本的Person证据。

作用：

零漏报最终兜底。

只要 Track 建立：

原则上它就应该存在。

--------------------------------
best_person
--------------------------------

Track生命周期内：

如果后续出现：

Person更大
更完整
更清晰
confidence更好

则不断替换。

--------------------------------
face_fallback
--------------------------------

第一次真正检测到Face时：

无论：
模糊
侧脸
低头
质量一般
AdaFace识别失败

都先保留。

这是：

“曾经检测到人脸”

的兜底证据。

--------------------------------
best_face
--------------------------------

Track生命周期中：

不断比较后续Face。

发现质量更高：

覆盖。

最终：

best_face应该尽可能代表：

该人员整个Track生命周期中最值得保存的人脸帧。

==================================================
# 十一、第一次出现Person不能等待高质量
==================================================

当发现：

新的有效 Person Track

不要：

等待3帧
等待5帧
等待人脸
等待更高confidence
等待bbox变大

而是：

立即建立：

person_fallback

注意：

“立即留底”

不等于：

立即写磁盘。

而是：

保存必要的内存候选数据。

后续出现更好图片：

继续替换best_person。

这样：

即使这个人突然离开画面，

最终仍然有：

person_fallback

可保存。

==================================================
# 十二、第一次检测到Face也不能因为质量差而丢弃
==================================================

第一次检测到Face：

立即建立：

face_fallback

不要因为：

face_det_score偏低
Laplacian低
人脸偏小
侧脸
低头
AdaFace unknown

就直接continue掉。

质量规则只能决定：

是否成为best_face。

不能决定：

是否留下face_fallback。

==================================================
# 十三、最佳人脸必须持续更新
==================================================

当前问题：

视频中后面明明出现非常清晰的正脸，

但程序保存了前面：

模糊
侧脸
后脑
运动状态

必须解决。

禁止：

检测到第一张脸
→ 保存
→ 后续不再更新

也禁止：

只比较最开始5帧。

正确逻辑：

整个Track生命周期内：

Face1
↓
评分

Face2
↓
评分

Face3
↓
评分

...

FaceN
↓
评分

只要：

new_quality_score > current_best_score

就：

更新best_face。

一直持续到：

Track结束 / 超时finalize。

==================================================
# 十四、Face质量评分保持简单
==================================================

不要新增：

FaceQNet
SER-FIQ
复杂Pose模型
大型质量评估网络

本次不要把程序搞复杂。

直接利用已有信息。

FaceQualityEvaluator：

计算：

1. 人脸检测置信度
2. 清晰度
3. 正脸程度
4. 人脸大小

得到：

quality_score

建议：

quality_score =
    det_weight * detection_score
  + sharpness_weight * sharpness_score
  + frontal_weight * frontal_score
  + size_weight * size_score

默认：

face_quality:
  det_weight: 0.30
  sharpness_weight: 0.30
  frontal_weight: 0.30
  size_weight: 0.10

所有子分数：

必须归一化到0~1。

禁止：

直接把：

Laplacian = 300

与：

confidence = 0.8

直接相加。

==================================================
# 十五、清晰度评分
==================================================

清晰度可以使用：

Laplacian Variance

但是：

只对Face Crop计算。

不要对整个1920×1080画面计算。

然后将结果：

归一化到0~1。

例如：

sharpness_score

用于比较同一个Track中的候选。

不要使用一个过于严格的sharpness阈值把人脸直接丢掉。

==================================================
# 十六、正脸评分
==================================================

如果当前 Face Detector 已经输出：

5点landmarks：

左眼
右眼
鼻子
左嘴角
右嘴角

利用这些关键点计算一个简单：

frontal_score

0~1。

主要判断：

双眼是否基本水平
鼻尖是否大致居中
左右结构是否相对对称

目的：

让：

清晰正脸

优先于：

清晰侧脸。

不要引入新的Head Pose模型。

==================================================
# 十七、人脸大小评分
==================================================

计算：

face bbox面积

相对于：

当前Person bbox
或者Frame

进行归一化。

目的：

在其他条件类似时：

更大的人脸优先。

但：

face size小

不能导致不记录。

只能降低quality_score。

==================================================
# 十八、质量等级
==================================================

除了quality_score：

增加：

quality_level

例如：

HIGH
MEDIUM
LOW

但是：

即使LOW：

也必须允许最终保存。

quality_level：

只用于：

日志
调试
后台分析

不能作为漏报条件。

==================================================
# 十九、最佳证据必须来自同一帧
==================================================

这是本次非常重要的要求。

如果：

best_face

最终来自：

Frame 238

那么最终人物证据：

必须也使用Frame 238。

即：

best_face.frame
+
Frame 238对应person_bbox

然后：

截取人物上半身。

禁止：

人脸来自Frame238
Person来自Frame150

最终拼成两套不一致证据。

==================================================
# 二十、最终保存的不是纯脸
==================================================

最终证据图片：

不要只保存Face Crop。

我要的是：

最佳人脸所在帧中的：

人物上半身。

推荐：

以person bbox为基础。

左右扩展：

padding_x

顶部扩展：

padding_top

高度：

截取person bbox上部约75%。

例如配置：

person_crop:
  padding_x_ratio: 0.20
  padding_top_ratio: 0.20
  upper_body_height_ratio: 0.75

所有参数：

进入config。

不要硬编码。

==================================================
# 二十一、必须避免以前“只截胸口/手”的问题
==================================================

以前出现过：

只保存胸口
只保存手
头部被裁掉

必须增加非常简单的安全保护。

上半身Crop：

必须保证：

包含face bbox。

如果计算出来的upper-body crop：

不能完整包含当前face bbox，

自动向：

上
左
右

扩展。

最终：

Face必须位于保存图中。

如果：

person bbox明显异常
crop太小
坐标非法

宁可：

保存当前完整Frame

也不要保存：

一只手
胸口
没有头的人。

==================================================
# 二十二、最终证据选择优先级
==================================================

Track finalize时：

严格按照：

优先级1：

best_face存在

→ 使用best_face对应Frame
→ 使用同帧person_bbox
→ 截取上半身
→ 保存

优先级2：

没有best_face
但存在face_fallback

→ 使用face_fallback对应Frame
→ 使用同帧person_bbox
→ 保存上半身

优先级3：

完全没有检测到Face
但存在best_person

→ 保存best_person

优先级4：

best_person不存在

→ 必须保存person_fallback

也就是说：

best_face
>
face_fallback
>
best_person
>
person_fallback

最后一级：

person_fallback

必须保证：

只要Person Track成立，

它就存在。

==================================================
# 二十三、Know / Unknown处理
==================================================

AdaFace继续使用现有实现。

不要重写已经可以正常工作的AdaFace模型加载和数据库逻辑。

如果：

worker_id识别成功

保存：

output/snapshot/face/know/

如果：

AdaFace没有识别成功

保存：

output/snapshot/face/unknow/

注意：

unknown不是“不保存”。

恰恰相反：

unknown也必须保存证据。

==================================================
# 二十四、低质量脸也必须进行必要的人脸识别
==================================================

不要设计成：

quality_score < 0.6
→ AdaFace完全不执行

否则：

低质量但其实可以认出来的人可能被漏掉。

可以：

高质量Face优先送AdaFace

但是：

Track生命周期内只要检测到有效Face，

必须至少有一次AdaFace识别机会。

可以控制频率：

避免每帧都识别。

但是：

不能因为质量一般就永远不识别。

==================================================
# 二十五、身份识别与最佳图片不要强绑定
==================================================

必须区分两个概念：

Identity：

这个人是谁。

Evidence：

哪一张图最好。

例如：

Frame 50：

AdaFace成功识别张三
similarity=0.72

Frame 100：

人脸更清晰
quality=0.91

但由于识别频率限制，
这一帧没有重新做AdaFace。

不能因此：

丢掉Frame100。

Track状态中：

identity_result

和：

best_face

应该独立维护。

最终可以：

identity = 张三
best evidence = Frame100

但：

如果后续更高质量人脸重新AdaFace后识别出不同的人，

必须按照现有连续确认/身份一致性逻辑处理，

不能简单覆盖造成误识。

==================================================
# 二十六、Top N图片支持
==================================================

默认：

只保存最佳1张。

但是支持：

snapshot:
  best_face_count: 1

允许配置：

1
2
3

最大：

3

如果设置：

best_face_count: 3

则：

保留Track生命周期Top3候选。

不要：

保存几十张
缓存几十帧
保存整个Track视频

Top N必须使用固定小容量结构。

==================================================
# 二十七、Person fallback评分
==================================================

Person也需要简单择优。

但是不要搞复杂人体质量模型。

建议：

person_score

综合：

person detection confidence
bbox面积
简单清晰度

例如：

person_quality:
  confidence_weight: 0.30
  area_weight: 0.40
  sharpness_weight: 0.30

如果获取detector confidence不可靠：

允许主要使用：

bbox面积
+
crop清晰度

目的：

只是在没有Face时：

尽量选一张比较好的Person图。

不是决定是否保存。

==================================================
# 二十八、不要频繁复制整帧
==================================================

生产目标：

128路
5FPS
2×L20

因此：

不能每个Person每帧：

np.copy(1920×1080完整Frame)

这样内存带宽和CPU压力会非常大。

设计原则：

Probe只做：

metadata读取
bbox读取
轻量判断

只有：

第一次person_fallback
第一次face_fallback
或者候选明显可能成为best

才进行必要的Crop复制。

优先复制：

人物ROI

而不是：

完整1080P Frame。

Candidate中：

尽量保存：

person crop
face crop
必要metadata

而不是长期保存完整Frame。

但是：

必须保证最终可以得到：

best face对应的同帧人物上半身。

==================================================
# 二十九、Probe主线程禁止做重活
==================================================

Probe主线程禁止：

数据库查询
HTTP调用
文件写盘
大量OpenCV处理
AdaFace同步推理
大尺寸图像质量计算
复杂Python循环
等待锁

Probe只负责：

NvDsMeta读取
track_id
bbox
confidence
轻量判断
提交必要任务

然后：

立即返回。

==================================================
# 三十、异步设计保持简单
==================================================

不要按照复杂互联网架构设计。

本项目不要增加：

Kafka
Redis
RabbitMQ
EventBus
Actor Framework
多层线程池

只需要简单的：

TrackEvidenceManager

FaceQualityEvaluator

SnapshotWriter

如果确实需要异步质量计算：

增加一个：

QualityWorker

即可。

不要再拆十几个类。

程序必须：

容易理解
容易Debug
容易修改。

==================================================
# 三十一、线程安全
==================================================

如果：

Probe线程
QualityWorker
SnapshotWriter

都会访问Track状态，

必须保证线程安全。

但是：

不要使用一个全局大锁阻塞所有摄像头。

可以：

每个Track轻量Lock

或者：

单Consumer状态更新

根据现有架构选择最简单可靠方案。

必须避免：

dictionary changed size during iteration

以及：

Track已经finalize
异步任务又回来修改Track

这类问题。

Candidate必须包含：

camera_id
track_id
generation/version

或者其他简单机制，

防止过期异步任务污染新Track。

==================================================
# 三十二、Track结束判断
==================================================

优先复用现有：

track lost
track timeout
stale cleanup

不要重新造第二套生命周期。

如果当前没有：

可以增加简单：

last_seen_timestamp

配置：

snapshot:
  track_timeout_sec: 10

超过时间：

finalize。

注意：

NvDCF内部的Track buffer

与：

业务证据Track timeout

不是同一个概念。

不要混淆。

==================================================
# 三十三、不要因为短暂检测丢失立即finalize
==================================================

NvDCF的意义之一：

就是处理短暂遮挡和检测器漏帧。

所以：

某一帧没有Person detection

不能马上：

finalize track。

必须依据：

NvDCF object_id生命周期
+
业务timeout

判断。

避免：

同一个人走动过程中产生大量：

track finalize
新track
重复证据。

==================================================
# 三十四、截图目录
==================================================

保持：

output/
└── snapshot/
    ├── person/
    └── face/
        ├── know/
        └── unknow/

如果现有目录已经存在：

优先兼容现有目录。

不要无必要改变后台依赖路径。

==================================================
# 三十五、文件名
==================================================

文件名至少包含：

timestamp
camera_id
track_id
identity
similarity
quality_score

例如：

20260811_101530_cam01_track123_know_10086_sim0.72_q0.91.jpg

unknown：

20260811_101532_cam01_track128_unknow_sim0.31_q0.65.jpg

图片：

禁止画框
禁止文字
禁止OSD结果

保存原始人物证据图。

==================================================
# 三十六、日志要求
==================================================

为了后续调试：

不要每帧刷大量日志。

只打印关键状态变化。

新Track：

[TRACK_CREATE]
camera=
track=
person_conf=

第一次Face：

[FACE_FALLBACK]
camera=
track=
face_conf=
quality=

发现更好Face：

[BEST_FACE_UPDATE]
camera=
track=
old_quality=
new_quality=
det=
sharpness=
frontal=
size=

识别：

[FACE_IDENTITY]
camera=
track=
worker_id=
similarity=

Track结束：

[TRACK_FINALIZE]
camera=
track=
source=best_face|face_fallback|best_person|person_fallback
identity=
similarity=
quality=
snapshot=

这样我下一次测试反馈时：

可以直接把日志给你分析。

==================================================
# 三十七、配置项
==================================================

优先合并进当前已有配置体系。

不要重复创建多个配置文件。

至少包含：

person_detector:
  type: peoplenet

tracker:
  type: nvdcf

inference:
  person_fps: 5
  face_fps: 2

face_quality:
  det_weight: 0.30
  sharpness_weight: 0.30
  frontal_weight: 0.30
  size_weight: 0.10

snapshot:
  enabled: true
  best_face_count: 1
  track_timeout_sec: 10
  async_write: true
  jpeg_quality: 92

person_crop:
  padding_x_ratio: 0.20
  padding_top_ratio: 0.20
  upper_body_height_ratio: 0.75

person_quality:
  confidence_weight: 0.30
  area_weight: 0.40
  sharpness_weight: 0.30

注意：

不要把所谓：

min_face_quality

设计成“不保存阈值”。

如果现有配置存在：

MIN_FACE_QUALITY

必须检查其用途。

允许它控制：

是否成为HIGH质量候选

禁止它控制：

整个Track是否保存证据。

==================================================
# 三十八、行为模型本次不要大改
==================================================

当前系统未来还要支持：

吸烟
吃东西
喝水
大件物品搬运
其他.pt模型

本次：

不要重构这些模块。

如果已经存在：

保持可以运行。

如果PeopleNet/NvDCF修改影响它们：

做必要兼容。

但不要本次顺便开发：

复杂BehaviorAnalyzer
遗留物判断
Person-Bag轨迹算法

这些后续单独处理。

本次先把：

Person
Track
Face
AdaFace
Evidence

稳定下来。

==================================================
# 三十九、AdaFace与现有数据库不能破坏
==================================================

现有：

AdaFace
PostgreSQL
pgvector
t_worker_face_vector

已经有可运行逻辑。

不要因为DeepStream改造：

重新更换embedding算法。

必须保持：

注册端AdaFace
=
识别端AdaFace

向量归一化逻辑保持一致。

如果发现：

模型
预处理
归一化

不一致：

必须在最终报告指出。

不要未经分析直接重新生成整个底库。

==================================================
# 四十、测试必须实际运行
==================================================

代码修改完成：

不要只告诉我：

“已经实现。”

必须使用：

当前Docker
当前DeepStream
当前测试视频

实际运行。

如果报错：

自行根据日志继续修复。

直到：

测试视频完整跑通。

==================================================
# 四十一、必须重点验证的场景
==================================================

至少检查以下场景：

场景1：

人员进入
→ 一直没露脸
→ 离开

结果：

必须保存1张Person证据。

不能漏。

-----------------------------

场景2：

人员进入
→ 出现非常模糊的脸
→ 离开

结果：

必须保存Face证据。

即使quality=LOW。

-----------------------------

场景3：

人员进入
→ 先侧脸
→ 再模糊脸
→ 后面出现清晰正脸
→ 离开

结果：

最终应该保存后面的清晰正脸对应上半身。

-----------------------------

场景4：

人员进入
→ AdaFace识别成功

结果：

know目录保存。

-----------------------------

场景5：

人员进入
→ 检测到脸
→ AdaFace无法识别

结果：

unknow目录保存。

绝对不能因为unknown而不保存。

-----------------------------

场景6：

人员短暂被遮挡
→ 又出现

如果NvDCF仍保持同一object_id：

业务层必须继续使用原Track。

不能因为某一帧没有检测到就提前finalize。

-----------------------------

场景7：

人脸清晰帧与之前的人形帧不同

最终：

必须使用最佳Face所在的同一Frame生成上半身证据。

-----------------------------

场景8：

person bbox异常导致上半身Crop可能丢头

必须检查：

Face是否包含在最终Crop。

如果无法保证：

扩大Crop。

仍然异常：

保存当前完整Frame兜底。

==================================================
# 四十二、测试输出统计
==================================================

为了判断是否真的存在漏报：

测试结束后增加简单统计。

例如：

========== Evidence Summary ==========

Person Tracks Created:       12

Tracks Finalized:            12

Best Face Evidence:           8
Face Fallback Evidence:       1
Best Person Evidence:         2
Person Fallback Evidence:     1

Know:                         7
Unknown:                      2

Snapshot Success:            12
Snapshot Failed:              0

======================================

核心校验：

Person Tracks Created
≈
Tracks Finalized
≈
Snapshot Success

如果：

创建12个Track

最终只有10张证据，

必须打印：

[ERROR][EVIDENCE_MISSING]

并给出：

camera_id
track_id
reason

不能静默漏报。

==================================================
# 四十三、增加Debug模式
==================================================

增加：

debug:
  evidence: true
  save_result_video: true

Debug模式：

输出带框测试视频：

output/debug/result.mp4

可以显示：

Person bbox
track_id
Face bbox
worker_id
best face quality

注意：

这是Debug视频。

真正的证据截图：

仍然禁止画框和文字。

生产：

debug.evidence=false

时：

关闭大量调试输出。

==================================================
# 四十四、性能统计
==================================================

测试结束至少输出：

视频FPS
实际处理FPS
PeopleNet推理FPS
Face任务数量
AdaFace任务数量
平均Probe耗时
P95 Probe耗时
Quality Queue最大长度
Snapshot Queue最大长度

如果可能：

同时输出GPU利用情况。

重点关注：

Probe平均耗时。

不要为了证据截图：

把DeepStream主pipeline阻塞。

==================================================
# 四十五、代码设计原则
==================================================

优先：

修改现有代码。

不要：

为了“架构漂亮”

把整个项目重写。

保持：

简单
稳定
容易Debug

本次建议最多新增：

TrackEvidenceManager
FaceQualityEvaluator
SnapshotWriter
QualityWorker（确实需要时）

如果现有工程已经有类似类：

直接复用和改造。

禁止重复造轮子。

==================================================
# 四十六、最终交付
==================================================

本次一次性完成。

不要让我确认阶段1以后再继续。

完成后给我一个简洁但完整的报告：

1. 原程序实际使用的Person Detector是什么
2. 原程序实际使用的Tracker是什么
3. 是否已经改成PeopleNet + NvDCF
4. PeopleNet实际class映射
5. 修改了哪些文件
6. 新增了哪些类
7. 新增/修改了哪些配置
8. 最佳Face算法怎么工作
9. 零漏报机制怎么工作
10. Person fallback怎么工作
11. AdaFace逻辑是否修改
12. 测试视频运行结果
13. 创建Track数量
14. 成功保存证据数量
15. know/unknown数量
16. 是否发现Evidence Missing
17. 平均/P95 Probe耗时
18. 输出图片目录
19. Debug视频路径
20. 当前仍存在的已知问题

==================================================
# 四十七、最终验收原则
==================================================

本次功能优先级严格按照：

第一优先级：
不能漏掉Person事件。

第二优先级：
有Face就尽量留下Face证据。

第三优先级：
从整个Track生命周期中选择最清晰、最正面的人脸。

第四优先级：
最终人物证据与最佳Face必须来自同一帧。

第五优先级：
尽可能提高AdaFace识别成功率。

第六优先级：
不能明显降低DeepStream整体吞吐性能。

宁可：

保存一张质量一般的图

也不能：

因为质量不够而完全没有证据。

宁可：

最后使用person_fallback

也不能：

Track结束后什么都不保存。

同时：

不要因为追求“绝对零漏报”
而每帧保存图片、频繁写磁盘或复制完整1080P Frame。

正确实现应该是：

第一次先留底
+
生命周期持续择优
+
Track结束一次性交卷
+
异步落盘

现在请直接开始读取当前工程并实施修改。

不要先给我方案。
不要只生成伪代码。
不要等待我确认。
直接修改实际工程、运行测试并修复问题，直到当前测试视频能够正常完成上述验收。