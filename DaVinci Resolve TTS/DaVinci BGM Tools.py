# -*- coding: utf-8 -*-
import json
import os
import platform
import sys
import time
from urllib import error as urllib_error
from urllib import request as urllib_request

SCRIPT_NAME = "DaVinci BGM Tools"
SCRIPT_VERSION = " 0.2-WSGH"
WINDOW_WIDTH = 760
WINDOW_HEIGHT = 580
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
X_CENTER = (SCREEN_WIDTH - WINDOW_WIDTH) // 2
Y_CENTER = (SCREEN_HEIGHT - WINDOW_HEIGHT) // 2

SCRIPT_PATH = os.path.dirname(os.path.abspath(sys.argv[0]))
PREFS_DIR = os.path.join(SCRIPT_PATH, "prefs")
DEFAULT_SERVER_BASE_URL = "http://127.0.0.1:39678"
BGMTOOLS_PREFS_PATH = os.path.join(PREFS_DIR, "bgmtools_prefs.json")
PAYLOAD_JSON_PATH = os.path.join(PREFS_DIR, "bgmtools_last_payload.json")
PAYLOAD_TEXT_PATH = os.path.join(PREFS_DIR, "bgmtools_last_payload.txt")
DEFAULT_PREFS = {
    "schema_version": 1,
    "server_base_url": DEFAULT_SERVER_BASE_URL,
}


def ensure_parent_dir(file_path):
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def normalize_prefs(prefs):
    normalized = dict(DEFAULT_PREFS)
    if not isinstance(prefs, dict):
        return normalized

    schema_version = prefs.get("schema_version", DEFAULT_PREFS["schema_version"])
    try:
        normalized["schema_version"] = int(schema_version)
    except (TypeError, ValueError):
        normalized["schema_version"] = DEFAULT_PREFS["schema_version"]

    server_base_url = str(prefs.get("server_base_url", DEFAULT_SERVER_BASE_URL) or "").strip()
    if server_base_url:
        normalized["server_base_url"] = server_base_url
    return normalized


def load_prefs():
    if not os.path.exists(BGMTOOLS_PREFS_PATH):
        return dict(DEFAULT_PREFS)

    try:
        with open(BGMTOOLS_PREFS_PATH, "r", encoding="utf-8") as file:
            content = file.read().strip()
    except IOError as err:
        print(f"读取 BGM Tools 偏好设置失败: {err}")
        return dict(DEFAULT_PREFS)

    if not content:
        return dict(DEFAULT_PREFS)

    try:
        prefs = json.loads(content)
    except json.JSONDecodeError as err:
        print(f"BGM Tools 偏好设置 JSON 解析失败: {err}")
        return dict(DEFAULT_PREFS)

    return normalize_prefs(prefs)


def save_prefs(prefs):
    normalized = normalize_prefs(prefs)
    ensure_parent_dir(BGMTOOLS_PREFS_PATH)
    temp_path = BGMTOOLS_PREFS_PATH + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(normalized, file, ensure_ascii=False, indent=4)
    os.replace(temp_path, BGMTOOLS_PREFS_PATH)
    return normalized


def normalize_server_base_url(server_base_url):
    normalized = str(server_base_url or "").strip().rstrip("/")
    if not normalized:
        raise ValueError("BlogHelper 服务地址不能为空。")
    if not normalized.startswith("http://") and not normalized.startswith("https://"):
        raise ValueError("BlogHelper 服务地址必须以 http:// 或 https:// 开头。")
    return normalized


def bootstrap_resolve():
    try:
        import DaVinciResolveScript as dvr_script
        from python_get_resolve import GetResolve
        print("DaVinciResolveScript from Python")
    except ImportError:
        if platform.system() == "Darwin":
            path1 = "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Examples"
            path2 = "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Modules"
        elif platform.system() == "Windows":
            program_data = os.environ.get("PROGRAMDATA", r"C:\ProgramData")
            path1 = os.path.join(
                program_data, "Blackmagic Design", "DaVinci Resolve",
                "Support", "Developer", "Scripting", "Examples"
            )
            path2 = os.path.join(
                program_data, "Blackmagic Design", "DaVinci Resolve",
                "Support", "Developer", "Scripting", "Modules"
            )
        else:
            raise EnvironmentError("Unsupported operating system")

        sys.path += [path1, path2]
        import DaVinciResolveScript as dvr_script
        from python_get_resolve import GetResolve
        print("DaVinciResolveScript from DaVinci")

    resolve = None
    try:
        resolve = GetResolve()
    except Exception as err:
        print(f"GetResolve 获取 Resolve 失败: {err}")

    if resolve is None:
        resolve = dvr_script.scriptapp("Resolve")

    if resolve is None:
        raise RuntimeError("未能连接到 DaVinci Resolve。")

    fusion_app = globals().get("fusion")
    if fusion_app is None:
        fusion_app = dvr_script.scriptapp("Fusion")

    if fusion_app is None:
        raise RuntimeError("未能获取 Fusion UI 上下文。")

    bmd_module = globals().get("bmd")
    if bmd_module is None:
        raise RuntimeError("未能获取 bmd 模块，请在 DaVinci Resolve 中运行该脚本。")

    return resolve, fusion_app, bmd_module


def get_current_context(resolve):
    project_manager = resolve.GetProjectManager() if resolve else None
    project = project_manager.GetCurrentProject() if project_manager else None
    timeline = project.GetCurrentTimeline() if project else None
    return project_manager, project, timeline


def safe_call(callable_obj, default=None):
    try:
        return callable_obj()
    except Exception:
        return default


def get_audio_track_count(timeline):
    if timeline is None:
        return 0
    return int(timeline.GetTrackCount("audio") or 0)


def get_timeline_name(timeline):
    if timeline is None:
        return ""
    return str(safe_call(lambda: timeline.GetName(), "") or "").strip()


def get_project_name(project):
    if project is None:
        return ""
    return str(safe_call(lambda: project.GetName(), "") or "").strip()


def get_track_items_in_order(timeline, track_index):
    items = timeline.GetItemListInTrack("audio", track_index) or []
    valid_items = [item for item in items if item is not None]
    valid_items.sort(key=lambda item: safe_call(lambda: int(item.GetStart()), 0))
    return valid_items


def extract_clip_name(timeline_item):
    media_pool_item = safe_call(lambda: timeline_item.GetMediaPoolItem(), None)
    if media_pool_item is not None:
        media_pool_name = str(safe_call(lambda: media_pool_item.GetName(), "") or "").strip()
        if media_pool_name:
            return media_pool_name

        file_name = str(safe_call(lambda: media_pool_item.GetClipProperty("File Name"), "") or "").strip()
        if file_name:
            return file_name

    timeline_name = str(safe_call(lambda: timeline_item.GetName(), "") or "").strip()
    if timeline_name:
        return timeline_name

    return "UnknownClip"


def deduplicate_in_order(values):
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def build_payload(project, timeline, track_index):
    track_items = get_track_items_in_order(timeline, track_index)
    clip_names = [extract_clip_name(item) for item in track_items]
    unique_clip_names = deduplicate_in_order(clip_names)

    payload = {
        "schema_version": 1,
        "action": "lookup_bgm_info",
        "source": SCRIPT_NAME,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "project_name": get_project_name(project),
        "timeline_name": get_timeline_name(timeline),
        "audio_track_index": track_index,
        "track_label": f"A{track_index}",
        "clip_count": len(clip_names),
        "clip_names": clip_names,
        "unique_clip_count": len(unique_clip_names),
        "unique_clip_names": unique_clip_names,
    }
    return payload


def format_query_text(payload):
    lines = [
        "BLOGHELPER_BGM_LOOKUP",
        f"工程: {payload['project_name']}",
        f"时间线: {payload['timeline_name']}",
        f"音轨: {payload['track_label']}",
        f"素材总数: {payload['clip_count']}",
        "素材名称（按时间线顺序）:",
    ]

    if payload["clip_names"]:
        for index, clip_name in enumerate(payload["clip_names"], start=1):
            lines.append(f"{index}. {clip_name}")
    else:
        lines.append("(空)")

    lines.extend([
        "",
        f"去重后素材数: {payload['unique_clip_count']}",
        "去重名称列表:",
    ])

    if payload["unique_clip_names"]:
        for index, clip_name in enumerate(payload["unique_clip_names"], start=1):
            lines.append(f"{index}. {clip_name}")
    else:
        lines.append("(空)")

    return "\n".join(lines)


def build_bloghelper_request_body(payload):
    items = []
    for clip_name in payload.get("clip_names", []):
        file_name = str(clip_name or "").strip()
        if not file_name:
            continue
        items.append({"fileName": file_name})

    if not items:
        raise ValueError("当前音轨没有可发送的音频素材名称。")

    return {"items": items}


def extract_response_message(response_data, fallback_text=""):
    if isinstance(response_data, dict):
        message = str(response_data.get("message", "") or "").strip()
        if message:
            return message
        last_error = str(response_data.get("lastError", "") or "").strip()
        if last_error:
            return last_error
    return str(fallback_text or "").strip()


def post_json(url, body, timeout=10):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib_request.Request(
        url=url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
        },
    )

    response_text = ""
    status_code = 0

    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
            status_code = int(response.getcode() or 200)
            response_text = response.read().decode("utf-8", errors="replace")
    except urllib_error.HTTPError as err:
        status_code = int(err.code or 0)
        response_text = err.read().decode("utf-8", errors="replace")
    except urllib_error.URLError as err:
        raise RuntimeError(f"连接 BlogHelper 失败：{err.reason}")

    try:
        response_data = json.loads(response_text) if response_text else {}
    except json.JSONDecodeError:
        response_data = {}

    if status_code < 200 or status_code >= 300:
        message = extract_response_message(response_data, response_text)
        raise RuntimeError(f"BlogHelper HTTP {status_code}：{message or '请求失败'}")

    if isinstance(response_data, dict) and response_data.get("success") is False:
        message = extract_response_message(response_data)
        raise RuntimeError(f"BlogHelper 返回失败：{message or '未知错误'}")

    if not isinstance(response_data, dict):
        raise RuntimeError("BlogHelper 返回的数据格式不是 JSON 对象。")

    return response_data


def deliver_payload(payload, query_text, server_base_url, request_body):
    ensure_parent_dir(PAYLOAD_JSON_PATH)
    ensure_parent_dir(PAYLOAD_TEXT_PATH)

    with open(PAYLOAD_JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    with open(PAYLOAD_TEXT_PATH, "w", encoding="utf-8") as file:
        file.write(query_text)

    request_url = server_base_url + "/api/bgm-copyright-export-requests"
    response_data = post_json(request_url, request_body)

    return {
        "mode": "http",
        "json_path": PAYLOAD_JSON_PATH,
        "text_path": PAYLOAD_TEXT_PATH,
        "request_url": request_url,
        "request_body": request_body,
        "response_data": response_data,
    }


saved_prefs = load_prefs()
resolve, fusion_app, bmd = bootstrap_resolve()
ui = fusion_app.UIManager
dispatcher = bmd.UIDispatcher(ui)


win = dispatcher.AddWindow(
    {
        "ID": "MainWin",
        "WindowTitle": SCRIPT_NAME + SCRIPT_VERSION,
        "Geometry": [X_CENTER, Y_CENTER, WINDOW_WIDTH, WINDOW_HEIGHT],
        "Spacing": 10,
        "StyleSheet": """
        * {
            font-size: 14px;
        }
        """
    },
    [
        ui.VGroup(
            {"Spacing": 10},
            [
                ui.Label({
                    "ID": "ContextLabel",
                    "Text": "",
                    "WordWrap": True,
                    "Alignment": {"AlignLeft": True, "AlignVCenter": True}
                }),
                ui.HGroup(
                    [
                        ui.Label({"Text": "BlogHelper", "Weight": 0.15}),
                        ui.LineEdit({
                            "ID": "ServerBaseUrlEdit",
                            "Text": "",
                            "PlaceholderText": DEFAULT_SERVER_BASE_URL,
                            "Weight": 0.6
                        }),
                        ui.Button({"ID": "SavePrefsButton", "Text": "保存地址", "Weight": 0.25}),
                    ]
                ),
                ui.HGroup(
                    [
                        ui.Label({"Text": "音轨号", "Weight": 0.15}),
                        ui.SpinBox({
                            "ID": "TrackIndexSpinBox",
                            "Minimum": 1,
                            "Maximum": 64,
                            "Value": 1,
                            "Weight": 0.15
                        }),
                        ui.Button({"ID": "RefreshButton", "Text": "刷新信息", "Weight": 0.25}),
                        ui.Button({"ID": "CollectButton", "Text": "采集并发送", "Weight": 0.45}),
                    ]
                ),
                ui.Label({
                    "ID": "DeliveryModeLabel",
                    "Text": "当前发送方式：HTTP POST 到 BlogHelper，同时写入 prefs/bgmtools_last_payload.json 与 .txt",
                    "WordWrap": True,
                    "Alignment": {"AlignLeft": True, "AlignVCenter": True}
                }),
                ui.TextEdit({
                    "ID": "PayloadText",
                    "Text": "",
                    "ReadOnly": True,
                    "Font": ui.Font({"PixelSize": 13}),
                    "Weight": 1
                }),
                ui.Label({
                    "ID": "StatusLabel",
                    "Text": "",
                    "WordWrap": True,
                    "Alignment": {"AlignLeft": True, "AlignVCenter": True}
                }),
                ui.HGroup(
                    [
                        ui.Button({"ID": "CloseButton", "Text": "关闭", "Weight": 1}),
                    ]
                ),
            ]
        )
    ]
)

items = win.GetItems()


def update_status(text):
    items["StatusLabel"].Text = str(text or "").strip()


def build_preview_text(payload, query_text, request_url=None, request_body=None, response_data=None, error_text=""):
    preview_sections = [
        query_text,
        "",
        "----- JSON -----",
        json.dumps(payload, ensure_ascii=False, indent=2),
    ]

    if request_url or request_body is not None:
        preview_sections.extend([
            "",
            "----- BlogHelper HTTP 请求 -----",
        ])
        if request_url:
            preview_sections.append(f"POST {request_url}")
        if request_body is not None:
            preview_sections.append(json.dumps(request_body, ensure_ascii=False, indent=2))

    if response_data is not None:
        preview_sections.extend([
            "",
            "----- BlogHelper 返回 -----",
            json.dumps(response_data, ensure_ascii=False, indent=2),
        ])

    if error_text:
        preview_sections.extend([
            "",
            "----- 错误信息 -----",
            str(error_text),
        ])

    return "\n".join(preview_sections)


def save_current_prefs_from_ui():
    server_base_url = normalize_server_base_url(items["ServerBaseUrlEdit"].Text)
    normalized = save_prefs({
        "schema_version": DEFAULT_PREFS["schema_version"],
        "server_base_url": server_base_url,
    })
    items["ServerBaseUrlEdit"].Text = normalized["server_base_url"]
    return normalized


def refresh_context_ui():
    _, project, timeline = get_current_context(resolve)
    if project is None:
        items["ContextLabel"].Text = "当前未打开达芬奇工程。"
        update_status("请先打开工程。")
        return project, timeline

    if timeline is None:
        project_name = get_project_name(project) or "(未命名工程)"
        items["ContextLabel"].Text = f"工程：{project_name}\n当前未打开时间线。"
        update_status("请先打开时间线。")
        return project, timeline

    project_name = get_project_name(project) or "(未命名工程)"
    timeline_name = get_timeline_name(timeline) or "(未命名时间线)"
    track_count = get_audio_track_count(timeline)
    track_spin = items["TrackIndexSpinBox"]
    track_spin.Maximum = max(track_count, 1)
    if track_spin.Value > track_spin.Maximum:
        track_spin.Value = track_spin.Maximum

    items["ContextLabel"].Text = (
        f"工程：{project_name}\n"
        f"时间线：{timeline_name}\n"
        f"当前音频轨数量：{track_count}"
    )
    update_status("已刷新当前工程与时间线信息。")
    return project, timeline


def on_refresh_clicked(ev):
    refresh_context_ui()


def on_save_prefs_clicked(ev):
    try:
        prefs = save_current_prefs_from_ui()
    except Exception as err:
        update_status(f"保存地址失败：{err}")
        return

    update_status(f"已保存 BlogHelper 服务地址：{prefs['server_base_url']}")


def on_collect_clicked(ev):
    project, timeline = refresh_context_ui()
    if project is None or timeline is None:
        return

    track_index = int(items["TrackIndexSpinBox"].Value)
    track_count = get_audio_track_count(timeline)
    if track_count <= 0:
        update_status("当前时间线没有音频轨，无法采集。")
        return

    if track_index < 1 or track_index > track_count:
        update_status(f"音轨号超出范围，请输入 1 到 {track_count} 之间的数值。")
        return

    payload = build_payload(project, timeline, track_index)
    query_text = format_query_text(payload)
    request_body = None
    request_url = ""

    try:
        prefs = save_current_prefs_from_ui()
        request_body = build_bloghelper_request_body(payload)
        request_url = prefs["server_base_url"] + "/api/bgm-copyright-export-requests"
        delivery_result = deliver_payload(payload, query_text, prefs["server_base_url"], request_body)
        items["PayloadText"].PlainText = build_preview_text(
            payload,
            query_text,
            request_url=delivery_result["request_url"],
            request_body=delivery_result["request_body"],
            response_data=delivery_result["response_data"],
        )
        response_data = delivery_result["response_data"]
        update_status(
            "已发送到 BlogHelper。"
            f" requestId: {response_data.get('requestId', '(无)')}"
            f" | 状态: {response_data.get('status', '(未知)')}"
        )
    except Exception as err:
        items["PayloadText"].PlainText = build_preview_text(
            payload,
            query_text,
            request_url=request_url,
            request_body=request_body,
            error_text=str(err),
        )
        update_status(f"发送失败：{err}")


def on_close(ev):
    dispatcher.ExitLoop()


items["ServerBaseUrlEdit"].Text = saved_prefs["server_base_url"]
win.On.RefreshButton.Clicked = on_refresh_clicked
win.On.SavePrefsButton.Clicked = on_save_prefs_clicked
win.On.CollectButton.Clicked = on_collect_clicked
win.On.CloseButton.Clicked = on_close
win.On.MainWin.Close = on_close

refresh_context_ui()
win.Show()
dispatcher.RunLoop()
win.Hide()
