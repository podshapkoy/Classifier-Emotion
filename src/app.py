import os
import io
import pickle
import base64
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from scipy import signal
from matplotlib.patches import Patch

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = load_model('./data/eeg_emotion_model.keras')

FS = 128
WINDOW_SIZE = 8
OVERLAP = 4
BANDS = {
    'alpha': (8, 13),
    'beta': (14, 30),
    'theta': (4, 7),
    'gamma': (30, 45)
}


def bandpass_filter(data, low, high, fs=FS, order=5):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [low / nyq, high / nyq], btype='band')
    return signal.filtfilt(b, a, data)


def split_bands(eeg_data):
    band_data = []
    for _, (low, high) in BANDS.items():
        filtered = np.zeros_like(eeg_data)
        for ch in range(eeg_data.shape[1]):
            filtered[:, ch, :] = bandpass_filter(eeg_data[:, ch, :], low, high)
        band_data.append(filtered)
    return band_data


def segment_data(data, window_size=WINDOW_SIZE, overlap=OVERLAP):
    n_trials, n_channels, n_samples = data.shape
    step = window_size * FS
    stride = overlap * FS
    segments = []
    for trial in range(n_trials):
        for i in range(0, n_samples - step + 1, stride):
            segments.append(data[trial, :, i:i + step])
    return np.array(segments)


def extract_pcc(segments):
    return np.array([np.corrcoef(seg) for seg in segments])


def normalize(valence_values):
    valence_values = np.array(valence_values)
    if valence_values.max() != valence_values.min():
        return (valence_values - valence_values.min()) / (valence_values.max() - valence_values.min())
    return np.zeros_like(valence_values)


def get_emotion_stats(valence_values):
    valence_values = normalize(valence_values)
    thresholds = {
        'восторг': 0.75,
        'радость': 0.60,
        'спокойствие': 0.45,
        'нейтрально': 0.35,
        'грусть': 0.20,
        'гнев': 0.0
    }
    color_map = {
        'восторг': '#FF5733',
        'радость': '#FFC300',
        'спокойствие': '#2ECC71',
        'нейтрально': '#3498DB',
        'грусть': '#5D6D7E',
        'гнев': '#E74C3C'
    }
    emotions = []
    for val in valence_values:
        if val >= thresholds['восторг']:
            e = 'восторг'
        elif val >= thresholds['радость']:
            e = 'радость'
        elif val >= thresholds['спокойствие']:
            e = 'спокойствие'
        elif val >= thresholds['нейтрально']:
            e = 'нейтрально'
        elif val >= thresholds['грусть']:
            e = 'грусть'
        else:
            e = 'гнев'
        emotions.append((e, color_map[e]))

    emotion_counts = {k: 0 for k in thresholds}
    for e, _ in emotions:
        emotion_counts[e] += 1

    return emotions, emotion_counts


def process_eeg_data(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    eeg_data = data['data'][:, :32, 384:]
    gsr_data = data['data'][:, 37:38, 384:]
    full_data = np.concatenate([eeg_data, gsr_data], axis=1)

    segments_all = []
    for band in split_bands(full_data):
        segments = segment_data(band)
        pcc = extract_pcc(segments)
        segments_all.append(pcc)

    X = np.concatenate(segments_all)[..., np.newaxis]
    return X, data


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
        X, raw_data = process_eeg_data(filepath)

        preds = model.predict(X).flatten()
        valence_values = preds.reshape(-1, 40).mean(axis=0)

        normalized_valence = normalize(valence_values)
        emotions, emotion_counts = get_emotion_stats(valence_values)

        top_indices = np.argsort(np.abs(normalized_valence - 0.5))[::-1][:5]
        top_videos = [{
            'number': int(i + 1),
            'valence': float(valence_values[i]),
            'emotion': emotions[i][0],
            'color': emotions[i][1]
        } for i in top_indices]

        fig, ax = plt.subplots(figsize=(14, 6))
        thresholds = [1.0, 0.75, 0.60, 0.45, 0.35, 0.20, 0.0]
        colors = ['#FF5733', '#FFC300', '#2ECC71', '#3498DB', '#5D6D7E', '#E74C3C']
        for i in range(len(thresholds) - 1):
            ax.axhspan(thresholds[i + 1], thresholds[i], color=colors[i], alpha=0.1)

        for i, (valence, (_, color)) in enumerate(zip(valence_values, emotions)):
            ax.plot(i + 1, valence, 'o', color=color, markersize=8)

        legend_elements = [Patch(facecolor=color, label=label) for label, color in {
            'восторг': '#FF5733',
            'радость': '#FFC300',
            'спокойствие': '#2ECC71',
            'нейтрально': '#3498DB',
            'грусть': '#5D6D7E',
            'гнев': '#E74C3C'
        }.items()]
        ax.legend(handles=legend_elements, title='Эмоции', loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.12))

        ax.set_title(' ', pad=20)
        ax.set_xlabel('Номер видео')
        ax.set_ylabel('Уровень валентности')
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle='--', alpha=0.6)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
        high_val = (normalized_valence > 0.6).sum()
        low_val = (normalized_valence < 0.4).sum()
        total = len(normalized_valence)
        high_percent = round(100 * high_val / total, 1)
        low_percent = round(100 * low_val / total, 1)

        return jsonify({
            'average_valence': float(np.mean(valence_values)),
            'emotion_counts': emotion_counts,
            'dominant_emotion': dominant_emotion,
            'high_percent': high_percent,
            'low_percent': low_percent,
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