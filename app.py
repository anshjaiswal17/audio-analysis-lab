import joblib
from flask import Flask, render_template_string, request, jsonify
import librosa
import librosa.display
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import os

# Load trained model
model = joblib.load("audio_model.pkl")


app = Flask(__name__)
os.makedirs('uploads', exist_ok=True)


class AudioAnalyzer:
    def __init__(self, y, sr):
        self.y = y
        self.sr = sr

    def _fig_to_base64(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        img = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return "data:image/png;base64," + img

    def get_waveform(self):
        fig, ax = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(self.y, sr=self.sr, ax=ax)
        ax.set_title("Waveform")
        return self._fig_to_base64(fig)

    def get_spectrogram(self):
        fig, ax = plt.subplots(figsize=(10, 4))
        D = librosa.stft(self.y)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img = librosa.display.specshow(S_db, sr=self.sr, x_axis='time', y_axis='hz', ax=ax)
        fig.colorbar(img, ax=ax)
        ax.set_title("Spectrogram")
        return self._fig_to_base64(fig)

    def get_mel_spectrogram(self):
        fig, ax = plt.subplots(figsize=(10, 4))
        M = librosa.feature.melspectrogram(y=self.y, sr=self.sr)
        M_db = librosa.power_to_db(M, ref=np.max)
        img = librosa.display.specshow(M_db, sr=self.sr, x_axis='time', y_axis='mel', ax=ax)
        fig.colorbar(img, ax=ax)
        ax.set_title("Mel Spectrogram")
        return self._fig_to_base64(fig)

    def get_spectrum(self):
        fig, ax = plt.subplots(figsize=(10, 3))
        fft = np.fft.fft(self.y[:8192])
        freqs = np.fft.fftfreq(len(fft), 1 / self.sr)
        ax.plot(freqs[:len(freqs)//2], np.abs(fft[:len(fft)//2]))
        ax.set_title("Frequency Spectrum")
        return self._fig_to_base64(fig)

    def extract_features(self):
        mfcc = np.mean(librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=13), axis=1)
        zcr = np.mean(librosa.feature.zero_crossing_rate(self.y))
        centroid = np.mean(librosa.feature.spectral_centroid(y=self.y, sr=self.sr))
        rms = np.mean(librosa.feature.rms(y=self.y))

        return np.hstack([mfcc, zcr, centroid, rms])

    def classify(self):
        features = self.extract_features()

        prediction = model.predict([features])[0]
        probabilities = model.predict_proba([features])[0]

        classes = model.classes_

        return {
            "prediction": prediction,
            "probabilities": dict(zip(classes, probabilities * 100))
        }

HTML = """<!DOCTYPE html>
<!DOCTYPE html>
<html>
<head>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Lobster+Two:ital,wght@0,400;0,700;1,400;1,700&display=swap" rel="stylesheet">
<title>Audio Analysis Lab</title>

<style>
body {
    font-family: Arial;
    background: #0a0e17;
    color: #e8f4f8;
    padding: 20px;
}

.container {
    max-width: 900px;
    margin: auto;
    text-align: center;
}

h1 {
    font-family: 'Lobster Two', cursive;
    font-size: 3.5rem;
    background: linear-gradient(135deg,#00ffcc,#ff006e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 20px;
}

.btn {
    background: linear-gradient(135deg,#00ffcc,#ff006e);
    color: #0a0e17;
    border: none;
    border-radius: 10 px;
    font-weight: bold;
    cursor: pointer;
    margin-top: 15px;
    }

.btn:disabled {
    opacity: 0.5;
}

.card {
    background: #141b2d;
    border-radius: 10px;
    padding: 20px;
    margin-top: 20px;
}

.prediction {
    font-size: 1.8rem;
    color: #ff006e;
    margin-bottom: 10px;
}

.bar {
    height: 10px;
    background: #00ffcc;
    margin-top: 5px;
    border-radius: 5px;
}

.status {
    margin-top: 15px;
    font-weight: bold;
    color: #00ffcc;
}
</style>

</head>

<body>

<div class="container">

<h1><img src="/static/speaker.gif" width="50" height="50" style="vertical-align: middle;"> Audio Analysis Lab</h1>

<input type="file" id="fileInput">
<br>
<button class="btn" id="analyzeBtn" onclick="analyze()">Analyze</button>

<div id="status" class="status"></div>

<div id="result"></div>
<div id="visuals"></div>

</div>

<script>
let file;

document.getElementById('fileInput').onchange = e => {
    file = e.target.files[0];
};

async function analyze(){

    if(!file){
        alert("Please select a file first!");
        return;
    }

    let status = document.getElementById('status');
    let btn = document.getElementById('analyzeBtn');

    // Show loading
    status.innerText = "⏳ Analyzing...";
    btn.disabled = true;

    let fd = new FormData();
    fd.append('audio', file);

    try{
        let r = await fetch('/analyze',{method:'POST',body:fd});
        let d = await r.json();

        if(d.error){
            status.innerText = "❌ Error: " + d.error;
            btn.disabled = false;
            return;
        }

        // ===== ML RESULT =====
        let c = d.ml_results.probabilities;
        let prediction = d.ml_results.prediction;

        let html = `
            <div class="card">
                <div class="prediction">Prediction: ${prediction}</div>
        `;

        Object.entries(c).forEach(([label, value]) => {
            html += `
                <p>${label}: ${value.toFixed(1)}%</p>
                <div class="bar" style="width:${value}%"></div>
            `;
        });

        html += `</div>`;
        document.getElementById('result').innerHTML = html;

        // ===== VISUALIZATIONS =====
        let v = d.visualizations;

        document.getElementById('visuals').innerHTML = `
            <div class="card">
                <h3>Waveform</h3>
                <img src="${v.waveform}" width="100%">
            </div>

            <div class="card">
                <h3>Spectrogram</h3>
                <img src="${v.spectrogram}" width="100%">
            </div>

            <div class="card">
                <h3>Mel Spectrogram</h3>
                <img src="${v.mel_spectrogram}" width="100%">
            </div>

            <div class="card">
                <h3>Frequency Spectrum</h3>
                <img src="${v.spectrum}" width="100%">
            </div>
        `;

        // Done
        status.innerText = "✅ Analysis Complete";

    }catch(e){
        status.innerText = "❌ Error: " + e.message;
    }

    btn.disabled = false;
}
</script>

</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        file = request.files['audio']
        path = 'uploads/temp.wav'
        file.save(path)

        y, sr = librosa.load(path, sr=22050)
        analyzer = AudioAnalyzer(y, sr)

        return jsonify({
            'visualizations': {
                'waveform': analyzer.get_waveform(),
                'spectrogram': analyzer.get_spectrogram(),
                'mel_spectrogram': analyzer.get_mel_spectrogram(),
                'spectrum': analyzer.get_spectrum()
            },
            'ml_results': analyzer.classify()
        })

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)