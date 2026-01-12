from flask import Flask, request, jsonify, send_file,  send_from_directory
from flask_cors import CORS
import os
import uuid
import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import logging
from datetime import datetime
import shutil
import random
import io
import base64

# 初始化模型
from model.openllama import OpenLLAMAPEFTModel

args = {
    'model': 'openllama_peft',
    'imagebind_ckpt_path': '../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth',
    'vicuna_ckpt_path': '../pretrained_ckpt/vicuna_ckpt/7b_v0',
    'anomalygpt_ckpt_path': './ckpt/train_mvtec/pytorch_model.pt',
    'delta_ckpt_path': '../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt',
    'stage': 2,
    'max_tgt_len': 128,
    'lora_r': 32,
    'lora_alpha': 32,
    'lora_dropout': 0.1
}

print("正在初始化模型...")
model = OpenLLAMAPEFTModel(**args)
delta_ckpt = torch.load(args['delta_ckpt_path'], map_location='cpu')
model.load_state_dict(delta_ckpt, strict=False)
delta_ckpt = torch.load(args['anomalygpt_ckpt_path'], map_location='cpu')
model.load_state_dict(delta_ckpt, strict=False)
model = model.eval().half().cuda()
print("模型初始化完成")

# 日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = Flask(__name__)
CORS(app)

# MVTec AD数据集路径 (需要根据实际路径修改)
MVTEC_AD_PATH = "../mvtec_anomaly_detection"  # 请修改为您的MVTec AD数据集路径

# 映射前端图像类型到MVTec AD类别
IMAGE_TYPE_MAPPING = {
    'mechanical': ['metal_nut', 'screw'],
    'electronic': ['cable', 'transistor',],
    'medicine': ['pill', 'capsule'],
    'textile': ['carpet', 'leather', 'tile', 'wood'],
    'metal': ['metal_nut', 'screw'],
}

# 创建目录
for d in [ "results", "generated", "sessions"]:
    os.makedirs(d, exist_ok=True)

SESSION_FOLDER = 'sessions'
STATIC_FOLDER = 'static'
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

OUTPUT_FOLDER = os.path.join(app.root_path, 'static', 'outputs')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

sessions = {}

# ---------- 上传 ----------
@app.route("/api/upload", methods=["POST"])
def upload_image():
    try:
        file = request.files["file"]
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.jpg")
        file.save(file_path)
        return jsonify({"success": True, "file_id": file_id, "file_path": f"/static/uploads/{file_id}.jpg"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# ---------- 分析 ----------
@app.route("/api/analyze", methods=["POST"])
def analyze_image():
    try:
        query           = request.form["query"]
        image_id        = request.form["image_id"]
        normal_image_id = request.form.get("normal_image_id") or None
        session_id      = request.form.get("session_id", "default")
        max_length      = int(request.form.get("max_length", 512))
        top_p           = float(request.form.get("top_p", 0.01))
        temperature     = float(request.form.get("temperature", 1.0))

        image_path = os.path.join("static/uploads", f"{image_id}.jpg")
        normal_path = None
        if normal_image_id:
            normal_path = os.path.join("static/uploads", f"{normal_image_id}.jpg")
        if not os.path.exists(image_path):
            return jsonify({"success": False, "error": "图像文件不存在"})

        if session_id not in sessions:
            sessions[session_id] = {"history": [], "modality_cache": []}
        session = sessions[session_id]

        prompt_text = ""
        for idx, (q, a) in enumerate(session["history"]):
            if idx == 0:
                prompt_text += f"{q}\n### Assistant: {a}\n###"
            else:
                prompt_text += f" Human: {q}\n### Assistant: {a}\n###"
        if session["history"]:
            prompt_text += f" Human: {query}"
        else:
            prompt_text += query

        response, pixel_output = model.generate({
            'prompt': prompt_text,
            'image_paths': [image_path],
            'normal_img_paths': [normal_path] if normal_path else [],
            'audio_paths': [],
            'video_paths': [],
            'thermal_paths': [],
            'top_p': top_p,
            'temperature': temperature,
            'max_tgt_len': max_length,
            'modality_embeds': session["modality_cache"]
        }, web_demo=True)

        session["history"].append((query, response))

        # ========== 热力图处理 ==========
        # 获取numpy热力值 (0-1)
        heatmap_data = pixel_output.reshape(224, 224).detach().cpu().numpy()
        heatmap_norm = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min() + 1e-8)

        # 生成彩色热力图 (COLORMAP_JET)
        heatmap_color = cv2.applyColorMap((heatmap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # 读取原始图像
        original_img = cv2.imread(image_path)
        h, w = original_img.shape[:2]
        heatmap_resized = cv2.resize(heatmap_color, (w, h))

        # 叠加：0.6 原图 + 0.4 热力图
        overlay = cv2.addWeighted(original_img, 0.6, heatmap_resized, 0.4, 0)
        heatmap_overlay_path = os.path.join("static/outputs", f"overlay_{uuid.uuid4()}.png")
        cv2.imwrite(heatmap_overlay_path, overlay)

        # 生成二值掩码 (阈值0.5)
        _, binary_mask = cv2.threshold((heatmap_norm * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
        mask_resized = cv2.resize(binary_mask, (w, h))
        mask_path = os.path.join("static/outputs", f"mask_{uuid.uuid4()}.png")
        cv2.imwrite(mask_path, mask_resized)

        return jsonify({
            "success": True,
            "response": response,
            "original_image": f"/static/uploads/{session_id}.jpg",
            "heatmap_image": f"/{heatmap_overlay_path}",  # 彩色叠加热力图
            "mask_image": f"/{mask_path}",                # 二值掩码

            "session_id": session_id
        })

    except Exception as e:
        logger.exception("analyze_image error")
        return jsonify({"success": False, "error": str(e)})

# ---------- 对话 ----------
@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        query       = request.form["query"]
        session_id  = request.form.get("session_id", "default")
        max_length  = int(request.form.get("max_length", 512))
        top_p       = float(request.form.get("top_p", 0.01))
        temperature = float(request.form.get("temperature", 1.0))

        if session_id not in sessions:
            return jsonify({"success": False, "error": "会话不存在"})
        session = sessions[session_id]

        prompt_text = ""
        for idx, (q, a) in enumerate(session["history"]):
            if idx == 0:
                prompt_text += f"{q}\n### Assistant: {a}\n###"
            else:
                prompt_text += f" Human: {q}\n### Assistant: {a}\n###"
        prompt_text += f" Human: {query}"

        response, _ = model.generate({
            'prompt': prompt_text,
            'image_paths': [],
            'normal_img_paths': [],
            'audio_paths': [],
            'video_paths': [],
            'thermal_paths': [],
            'top_p': top_p,
            'temperature': temperature,
            'max_tgt_len': max_length,
            'modality_embeds': session["modality_cache"]
        }, web_demo=True)

        session["history"].append((query, response))
        return jsonify({"success": True, "response": response, "session_id": session_id})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# ---------- 保存检测记录 ----------
@app.route('/api/save-session', methods=['POST'])
def save_session():
    try:
        original_path = request.form.get('original_path')  # 例：uploads/xxx.jpg
        heatmap_path = request.form.get('heatmap_path')   # 例：outputs/yyy.jpg
        mask_path = request.form.get('mask_path')  # 例：outputs/yyy.jpg
        chat_log = request.form.get('chat_log', '')

        if not original_path or not heatmap_path or not mask_path:
            return jsonify(success=False, detail="缺少图像路径"), 400

        # 以时间+随机串命名会话文件夹
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        sid = str(uuid.uuid4())[:6]
        session_dir = os.path.join(SESSION_FOLDER, f"{ts}_{sid}")
        os.makedirs(session_dir, exist_ok=True)

        # 复制图像
        safe_copy(original_path, session_dir, 'original.jpg')
        safe_copy(heatmap_path,  session_dir, 'heatmap.jpg')
        safe_copy(mask_path, session_dir, 'mask.jpg')

        # 保存对话
        with open(os.path.join(session_dir, 'chat.txt'), 'w', encoding='utf-8') as f:
            f.write(chat_log)

        return jsonify(success=True, folder=session_dir)
    except Exception as e:
        return jsonify(success=False, detail=str(e)), 500

def safe_copy(src_relative, dest_dir, new_name):
    """
    把 static/ 下的文件复制到 dest_dir，并重命名为 new_name
    """
    src = os.path.join(STATIC_FOLDER, src_relative)
    dst = os.path.join(dest_dir, new_name)
    if os.path.exists(src):
        shutil.copy2(src, dst)
    else:
        # 如果前端传的是绝对路径或者已经带 static，再尝试一次
        if os.path.exists(src_relative):
            shutil.copy2(src_relative, dst)
        else:
            raise FileNotFoundError(src)

# ---------- 重置 ----------
@app.route("/api/reset", methods=["POST"])
def reset_session():
    session_id = request.form.get("session_id", "default")
    if session_id in sessions:
        sessions[session_id] = {"history": [], "modality_cache": []}
    return jsonify({"success": True})

def get_random_image_from_category(category, is_normal=True):
    """从指定类别中随机获取一张图像"""
    category_path = os.path.join(MVTEC_AD_PATH, category)

    if is_normal:
        # 从正常图像中获取
        normal_path = os.path.join(category_path, "train", "good")
        if os.path.exists(normal_path):
            images = [f for f in os.listdir(normal_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                img_path = os.path.join(normal_path, random.choice(images))
                return Image.open(img_path)
    else:
        # 从异常图像中获取
        test_path = os.path.join(category_path, "test")
        if os.path.exists(test_path):
            # 获取所有异常类型
            defect_types = [d for d in os.listdir(test_path) if
                            d != "good" and os.path.isdir(os.path.join(test_path, d))]
            if defect_types:
                defect_type = random.choice(defect_types)
                defect_path = os.path.join(test_path, defect_type)
                images = [f for f in os.listdir(defect_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
                if images:
                    img_path = os.path.join(defect_path, random.choice(images))
                    return Image.open(img_path)

    return None


def image_to_base64(img):
    """将PIL图像转换为Base64字符串"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

# 在app.py中添加以下内容

# 用于存储生成记录（实际应用中应使用数据库）
generation_records = []

# 添加新的路由用于获取生成记录
@app.route('/get_records', methods=['GET'])
def get_records():
    """获取生成记录"""
    return jsonify({
        "records": generation_records
    })

# 修改generate_image路由，添加记录保存功能
@app.route('/generate_image', methods=['POST'])
def generate_image():
    """处理图像生成请求"""
    try:
        data = request.get_json()
        model_type = data.get('model_type', 'gan')
        image_type = data.get('image_type', 'mechanical')
        complexity = int(data.get('complexity', 5))

        # 根据复杂度决定是否使用异常图像
        # 复杂度越高，使用异常图像的概率越大
        use_normal = random.random() > (complexity / 10)

        # 获取对应的MVTec AD类别
        categories = IMAGE_TYPE_MAPPING.get(image_type, ["metal_nut"])
        selected_category = random.choice(categories)

        # 获取随机图像
        img = get_random_image_from_category(selected_category, use_normal)

        if img is None:
            return jsonify({"error": "无法获取图像，请检查数据集路径"}), 500

        # 转换为Base64
        img_base64 = image_to_base64(img)

        # 记录生成信息（实际应用中应使用数据库）
        import datetime
        record = {
            "id": len(generation_records) + 1,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": model_type,
            "image_type": image_type,
            "complexity": complexity,
            "category": selected_category,
            "is_normal": use_normal,
            "image_data": img_base64  # 实际应用中可能只存储路径而非完整base64
        }
        generation_records.append(record)
        # 限制记录数量，避免内存溢出
        if len(generation_records) > 100:
            generation_records.pop(0)

        return jsonify({
            "image_data": img_base64,
            "category": selected_category,
            "is_normal": use_normal
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# === 新增API：获取所有会话列表 ===
@app.route("/api/get-sessions", methods=["GET"])
def get_sessions():
    try:
        sessions = []
        for session_dir in os.listdir(SESSION_FOLDER):
            session_path = os.path.join(SESSION_FOLDER, session_dir)
            if os.path.isdir(session_path):
                # 检查是否包含必要文件
                if all(os.path.exists(os.path.join(session_path, f)) for f in ['original.jpg', 'heatmap.jpg', 'mask.jpg']):
                    sessions.append({
                        "id": session_dir,
                        "timestamp": session_dir.split("_")[0],
                        "original_image": f"/static/sessions/{session_dir}/original.jpg",
                        "heatmap_image": f"/static/sessions/{session_dir}/heatmap.jpg",
                        "mask_image": f"/static/sessions/{session_dir}/mask.jpg"
                    })
        # 按时间倒序排序
        sessions.sort(key=lambda x: x["timestamp"], reverse=True)
        return jsonify({"success": True, "sessions": sessions})
    except Exception as e:
        logger.exception("获取会话列表失败")
        return jsonify({"success": False, "error": "服务器错误"})

# === 新增API：获取单个会话详情 ===
@app.route("/api/get-session-detail/<session_id>", methods=["GET"])
def get_session_detail(session_id):
    try:
        session_path = os.path.join(SESSION_FOLDER, session_id)
        if not os.path.isdir(session_path):
            return jsonify({"success": False, "error": "会话不存在"})

        # 读取对话日志
        chat_log = ""
        chat_log_path = os.path.join(session_path, "chat.txt")
        if os.path.exists(chat_log_path):
            with open(chat_log_path, "r", encoding="utf-8") as f:
                chat_log = f.read()

        # return jsonify({
        #     "success": True,
        #     "data": {
        #         "id": session_id,
        #         "original_image": f"/static/sessions/{session_id}/original.jpg",
        #         "heatmap_image": f"/static/sessions/{session_id}/heatmap.jpg",
        #         "chat_log": chat_log,
        #         "timestamp": session_id.split("_")[0]
        #     }
        # })

        return jsonify({
            "success": True,
            "data": {
                "id": session_id,
                "original_image": f"/static/sessions/{session_id}/original.jpg",
                "heatmap_image": f"/static/sessions/{session_id}/heatmap.jpg",
                "mask_image": f"/static/sessions/{session_id}/mask.jpg",  # 新增
                "chat_log": chat_log,
                "timestamp": session_id.split("_")[0]
            }
        })

    except Exception as e:
        logger.exception(f"获取会话详情失败: {session_id}")
        return jsonify({"success": False, "error": "服务器错误"})

# === 新增API：删除会话 ===
@app.route("/api/delete-session/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    try:
        session_path = os.path.join(SESSION_FOLDER, session_id)
        if not os.path.exists(session_path):
            return jsonify({"success": False, "error": "会话不存在"})

        shutil.rmtree(session_path)
        return jsonify({"success": True})
    except Exception as e:
        logger.exception(f"删除会话失败: {session_id}")
        return jsonify({"success": False, "error": "删除失败"})

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({"status": "healthy"})

# ---------- 主页 ----------
@app.route("/")
def root():
    return send_file("index.html")

# ---------- 静态文件 ----------
@app.route('/static/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# === 新增静态文件路由 ===
@app.route('/static/sessions/<path:filename>')
def session_static(filename):
    return send_from_directory(SESSION_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=False, port=5000, threaded=False)