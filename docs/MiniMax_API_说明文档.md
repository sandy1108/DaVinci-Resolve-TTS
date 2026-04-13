# MiniMax API 使用说明文档

## 概述

本项目使用 MiniMax 的语音合成（TTS）API 和音色快速复刻（Voice Cloning）API 来实现 DaVinci Resolve 中的文字转语音功能。

## 使用的 API 接口

### 1. 同步语音合成 API (Text-to-Audio)

**接口地址：** `POST https://api.minimaxi.com/v1/t2a_v2`

**国际版地址：** `POST https://api.minimaxi.chat/v1/t2a_v2`

**备用地址：** `POST https://api-bj.minimaxi.com/v1/t2a_v2`

**功能：** 将文本转换为语音音频

#### 认证方式
- **Header:** `Authorization: Bearer {API_KEY}`
- **Content-Type:** `application/json`

#### 支持的模型
- `speech-2.6-hd` - 最新 HD 模型，极致音质与韵律表现
- `speech-2.6-turbo` - 最新 Turbo 模型，超低时延
- `speech-02-hd` - 出色的韵律和稳定性
- `speech-02-turbo` - 小语种能力增强
- `speech-01-hd` - 旧版 HD 模型
- `speech-01-turbo` - 旧版 Turbo 模型

#### 主要请求参数

##### 必需参数
- **model** (string): 模型版本，如 `speech-2.6-hd`
- **text** (string): 需要合成的文本，长度限制 < 10000 字符
  - 支持换行符分段
  - 支持停顿控制：`<#x#>` (x为停顿秒数，范围 0.01-99.99)

##### voice_setting (语音设置对象)
- **voice_id** (string, 必需): 音色编号
  - 系统音色示例：
    - 中文：`moss_audio_ce44fc67-7ce3-11f0-8de5-96e35d26fb85`
    - 英文：`English_Graceful_Lady`
    - 日文：`Japanese_Whisper_Belle`
- **speed** (number, 默认1.0): 语速，范围 [0.5, 2]
- **vol** (number, 默认1.0): 音量，范围 (0, 10]
- **pitch** (integer, 默认0): 语调，范围 [-12, 12]
- **emotion** (string, 可选): 情绪控制
  - 可选值：`happy`, `sad`, `angry`, `fearful`, `disgusted`, `surprised`, `calm`, `fluent`
  - `fluent` 仅支持 speech-2.6 系列
- **text_normalization** (boolean, 默认false): 文本规范化
- **latex_read** (boolean, 默认false): 朗读 LaTeX 公式

##### audio_setting (音频设置对象)
- **sample_rate** (integer, 默认32000): 采样率
  - 可选：8000, 16000, 22050, 24000, 32000, 44100
- **bitrate** (integer, 默认128000): 比特率（仅 mp3）
  - 可选：32000, 64000, 128000, 256000
- **format** (string, 默认mp3): 音频格式
  - 可选：`mp3`, `pcm`, `flac`, `wav`（wav 仅非流式）
- **channel** (integer, 默认1): 声道数，1=单声道，2=双声道
- **force_cbr** (boolean, 默认false): 恒定比特率（仅流式 mp3）

##### voice_modify (声音效果器，可选)
- **pitch** (integer): 音高调整，范围 [-100, 100]
  - 负值更低沉，正值更明亮
- **intensity** (integer): 强度调整，范围 [-100, 100]
  - 负值更刚劲，正值更轻柔
- **timbre** (integer): 音色调整，范围 [-100, 100]
  - 负值更浑厚，正值更清脆
- **sound_effects** (string): 音效
  - 可选：`spacious_echo`（空旷回音）, `auditorium_echo`（礼堂广播）, `lofi_telephone`（电话失真）, `robotic`（电音）

##### 其他参数
- **stream** (boolean, 默认false): 是否流式输出
- **subtitle_enable** (boolean, 默认false): 是否开启字幕（仅非流式）
- **output_format** (string, 默认hex): 输出格式
  - 可选：`hex`, `url`（url 有效期 24 小时）
- **language_boost** (string): 小语种增强
  - 可选：`Chinese`, `English`, `Japanese`, `Korean` 等，或 `auto`
- **aigc_watermark** (boolean, 默认false): 添加音频水印

#### 响应格式

```json
{
  "data": {
    "audio": "<hex编码的音频数据>",
    "status": 2,
    "subtitle_file": "字幕文件URL（如启用）"
  },
  "extra_info": {
    "audio_length": 9900,
    "audio_sample_rate": 32000,
    "audio_size": 160323,
    "bitrate": 128000,
    "word_count": 52,
    "usage_characters": 26,
    "audio_format": "mp3",
    "audio_channel": 1
  },
  "trace_id": "请求追踪ID",
  "base_resp": {
    "status_code": 0,
    "status_msg": "success"
  }
}
```

---

### 2. 音色快速复刻 API (Voice Cloning)

音色复刻需要三个步骤：

#### 步骤 1: 上传待复刻音频

**接口地址：** `POST https://api.minimaxi.com/v1/files/upload`

**功能：** 上传需要复刻的音频文件

**请求参数：**
- **file** (multipart/form-data): 音频文件
  - 格式：mp3, m4a, wav
  - 时长：10秒 - 5分钟
  - 大小：≤ 20MB
- **purpose** (string): 固定值 `voice_clone`

**响应：**
```json
{
  "file": {
    "file_id": "文件ID"
  },
  "base_resp": {
    "status_code": 0,
    "status_msg": "success"
  }
}
```

#### 步骤 2: 上传示例音频（可选）

**接口地址：** `POST https://api.minimaxi.com/v1/files/upload`

**功能：** 上传示例音频以增强复刻效果

**请求参数：**
- **file** (multipart/form-data): 示例音频文件
  - 格式：mp3, m4a, wav
  - 时长：< 8秒
  - 大小：≤ 20MB
- **purpose** (string): 固定值 `voice_clone`

#### 步骤 3: 执行音色复刻

**接口地址：** `POST https://api.minimaxi.com/v1/voice_clone`

**功能：** 基于上传的音频创建复刻音色

**请求参数：**

##### 必需参数
- **file_id** (string): 待复刻音频的 file_id（步骤1获得）
- **voice_id** (string): 自定义音色ID
  - 长度：8-256 字符
  - 首字符必须为英文字母
  - 允许：数字、字母、`-`、`_`
  - 末位不可为 `-` 或 `_`
  - 不可与已有 ID 重复

##### 可选参数
- **clone_prompt** (object): 示例音频信息
  - **prompt_audio** (string): 示例音频的 file_id（步骤2获得）
  - **prompt_text** (string): 示例音频对应的文本
- **text** (string): 试听文本（≤ 1000 字符）
- **model** (string): 试听使用的模型（提供 text 时必需）
  - 可选：`speech-2.6-hd`, `speech-2.6-turbo`, `speech-02-hd`, `speech-02-turbo`
- **need_noise_reduction** (boolean, 默认false): 是否降噪
- **need_volume_normalization** (boolean, 默认false): 是否音量归一化
- **language_boost** (string): 小语种增强
- **aigc_watermark** (boolean, 默认false): 添加水印

**响应：**
```json
{
  "input_sensitive": false,
  "input_sensitive_type": 0,
  "demo_audio": "试听音频URL（如提供text）",
  "base_resp": {
    "status_code": 0,
    "status_msg": "success"
  }
}
```

#### 重要说明
1. **费用计算：** 复刻费用在首次使用该音色进行语音合成时收取
2. **音色有效期：** 临时音色 168 小时（7天），需在此期间使用一次以永久保留
3. **无状态设计：** 接口不存储用户上传内容

---

## 项目中的实现

### MiniMaxProvider 类

项目中通过 `MiniMaxProvider` 类封装了 MiniMax API 的调用：

```python
class MiniMaxProvider:
    BASE_URL = "https://api.minimax.chat"
    BASE_URL_INTL = "https://api.minimaxi.chat"
    
    def __init__(self, api_key: str, group_id: str, is_intl: bool = False)
    
    def synthesize(self, text, model, voice_id, speed, vol, pitch, 
                   file_format, subtitle_enable, emotion, sound_effects)
    
    def upload_file_for_clone(self, file_path: str)
    
    def submit_clone_job(self, file_id, voice_id, need_nr, need_vn, text)
    
    def download_media(self, url: str)
```

### 主要方法说明

1. **synthesize()** - 调用 `/v1/t2a_v2` 接口进行语音合成
2. **upload_file_for_clone()** - 调用 `/v1/files/upload` 上传音频文件
3. **submit_clone_job()** - 调用 `/v1/voice_clone` 执行音色复刻
4. **download_media()** - 下载生成的音频或字幕文件

---

## 与官方文档的对比

### 当前实现已包含的参数

✅ 基础参数：model, text, voice_id, speed, vol, pitch
✅ 音频设置：sample_rate, bitrate, format, channel
✅ 情绪控制：emotion
✅ 音效：sound_effects (voice_modify)
✅ 字幕：subtitle_enable
✅ 音色复刻：完整流程支持

### 可以新增的参数（官方最新）

以下是官方文档中存在但项目未使用的参数：

#### 语音合成新增参数
1. **stream_options** - 流式输出选项
2. **pronunciation_dict** - 发音字典（多音字、英文发音）
3. **timber_weights** - 混合音色权重
4. **language_boost** - 小语种增强
5. **output_format** - 输出格式控制（url/hex）
6. **aigc_watermark** - AIGC 水印
7. **voice_setting.text_normalization** - 文本规范化
8. **voice_setting.latex_read** - LaTeX 公式朗读
9. **audio_setting.force_cbr** - 恒定比特率（流式）
10. **voice_modify.pitch/intensity/timbre** - 更精细的音色调整

#### 音色复刻新增参数
1. **clone_prompt** - 示例音频（增强复刻效果）
2. **language_boost** - 小语种增强

---

## 建议的优化方向

### 1. 高优先级
- 添加 `language_boost` 参数支持多语种
- 添加 `text_normalization` 提升数字朗读
- 添加 `output_format="url"` 选项（避免大文件传输）

### 2. 中优先级
- 支持 `pronunciation_dict` 解决多音字问题
- 添加 `timber_weights` 实现混合音色
- 完善 `voice_modify` 的三个维度调整

### 3. 低优先级
- 支持 `latex_read` 用于科技内容
- 添加 `aigc_watermark` 用于版权保护
- 支持流式输出相关参数

---

## 参考链接

- [MiniMax 开放平台](https://platform.minimaxi.com)
- [同步语音合成 HTTP API](https://platform.minimaxi.com/docs/api-reference/speech-t2a-http)
- [音色快速复刻 API](https://platform.minimaxi.com/docs/api-reference/voice-cloning-intro)
- [系统音色列表](https://platform.minimaxi.com/docs/faq/system-voice-id)
- [错误码查询](https://platform.minimaxi.com/docs/api-reference/errorcode)

---

**文档生成时间：** 2025-11-30

**MiniMax API 版本：** v1 (Speech 2.6 系列)
