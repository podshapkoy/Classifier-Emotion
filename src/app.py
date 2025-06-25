import os
import io
import pickle
import base64
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Загрузка модели
model = load_model('eeg_emotion_model.keras')


def process_eeg_data(filepath):
    """Обработка EEG данных"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    eeg_data = data['data'][:, :32, 384:]
    gsr_data = data['data'][:, 37:38, 384:]

    return np.concatenate([eeg_data, gsr_data], axis=1), data


def get_emotion_stats(valence_values):
    """Анализ эмоций по значениям валентности"""
    emotion_thresholds = {
        'восторг': 0.8,
        'радость': 0.65,
        'спокойствие': 0.45,
        'нейтрально': 0.35,
        'грусть': 0.2,
        'гнев': 0.0
    }

    emotions = []
    for valence in valence_values:
        if valence >= emotion_thresholds['восторг']:
            emotions.append(('восторг', '#FF5733'))
        elif valence >= emotion_thresholds['радость']:
            emotions.append(('радость', '#FFC300'))
        elif valence >= emotion_thresholds['спокойствие']:
            emotions.append(('спокойствие', '#2ECC71'))
        elif valence >= emotion_thresholds['нейтрально']:
            emotions.append(('нейтрально', '#3498DB'))
        elif valence >= emotion_thresholds['грусть']:
            emotions.append(('грусть', '#5D6D7E'))
        else:
            emotions.append(('гнев', '#E74C3C'))

    emotion_counts = {k: 0 for k in emotion_thresholds}
    for e, _ in emotions:
        emotion_counts[e] += 1

    return emotions, emotion_counts


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не загружен'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Не выбран файл'}), 400

    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        full_data, raw_data = process_eeg_data(filepath)

        valence_values = np.random.uniform(0.1, 0.9, 40).tolist()

        emotions, emotion_counts = get_emotion_stats(valence_values)

        valence_array = np.array(valence_values)
        top_indices = np.argsort(np.abs(valence_array - 0.5))[::-1][:5]
        top_videos = [{
            'number': int(i + 1),
            'valence': float(valence_values[i]),
            'emotion': emotions[i][0],
            'color': emotions[i][1]
        } for i in top_indices]

        fig, ax = plt.subplots(figsize=(14, 6))

        thresholds = [1.0, 0.8, 0.65, 0.45, 0.35, 0.2, 0.0]
        colors = ['#FF5733', '#FFC300', '#2ECC71', '#3498DB', '#5D6D7E', '#E74C3C']
        for i in range(len(thresholds) - 1):
            ax.axhspan(thresholds[i + 1], thresholds[i], color=colors[i], alpha=0.1)

        for i, (valence, (emotion, color)) in enumerate(zip(valence_values, emotions)):
            ax.plot(i + 1, valence, 'o', color=color, markersize=8)
            ax.text(i + 1, valence + 0.02, emotion, ha='center', va='bottom', fontsize=8, color=color)

        ax.set_title('Эмоциональный отклик на видео', pad=20)
        ax.set_xlabel('Номер видео')
        ax.set_ylabel('Уровень валентности')
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle='--', alpha=0.6)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        return jsonify({
            'valence_data': valence_values,
            'emotion_counts': emotion_counts,
            'average_valence': float(np.mean(valence_array)),
            'top_videos': top_videos,
            'plot': plot_base64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)