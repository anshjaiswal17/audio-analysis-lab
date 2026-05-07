import os
os.environ["NUMBA_DISABLE_JIT"] = "1"
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
import tempfile
import traceback

# Load trained model
try:
    model = joblib.load("audio_model.pkl")
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"⚠ Warning: Could not load model - {e}")
    model = None

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
        try:
            fig, ax = plt.subplots(figsize=(10, 3))
            librosa.display.waveshow(self.y, sr=self.sr, ax=ax)
            ax.set_title("Waveform")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            plt.tight_layout()
            return self._fig_to_base64(fig)
        except Exception as e:
            print(f"Error in get_waveform: {e}")
            raise

    def get_spectrogram(self):
        try:
            fig, ax = plt.subplots(figsize=(10, 4))
            D = librosa.stft(self.y)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            img = librosa.display.specshow(S_db, sr=self.sr, x_axis='time', y_axis='hz', ax=ax)
            fig.colorbar(img, ax=ax)
            ax.set_title("Spectrogram")
            plt.tight_layout()
            return self._fig_to_base64(fig)
        except Exception as e:
            print(f"Error in get_spectrogram: {e}")
            raise

    def get_mel_spectrogram(self):
        try:
            fig, ax = plt.subplots(figsize=(10, 4))
            M = librosa.feature.melspectrogram(y=self.y, sr=self.sr)
            M_db = librosa.power_to_db(M, ref=np.max)
            img = librosa.display.specshow(M_db, sr=self.sr, x_axis='time', y_axis='mel', ax=ax)
            fig.colorbar(img, ax=ax)
            ax.set_title("Mel Spectrogram")
            plt.tight_layout()
            return self._fig_to_base64(fig)
        except Exception as e:
            print(f"Error in get_mel_spectrogram: {e}")
            raise

    def get_spectrum(self):
        try:
            fig, ax = plt.subplots(figsize=(10, 3))
            fft_size = min(8192, len(self.y))
            fft = np.fft.fft(self.y[:fft_size])
            freqs = np.fft.fftfreq(fft_size, 1 / self.sr)
            ax.plot(freqs[:len(freqs)//2], np.abs(fft[:len(fft)//2]))
            ax.set_title("Frequency Spectrum")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Magnitude")
            plt.tight_layout()
            return self._fig_to_base64(fig)
        except Exception as e:
            print(f"Error in get_spectrum: {e}")
            raise

    def extract_features(self):
        try:
            mfcc = np.mean(librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=13), axis=1)
            zcr = np.mean(librosa.feature.zero_crossing_rate(self.y))
            centroid = np.mean(librosa.feature.spectral_centroid(y=self.y, sr=self.sr))
            rms = np.mean(librosa.feature.rms(y=self.y))
            
            features = np.hstack([mfcc, zcr, centroid, rms])
            print(f"✓ Features extracted: shape {features.shape}")
            return features
        except Exception as e:
            print(f"Error in extract_features: {e}")
            raise

    def classify(self):
        try:
            features = self.extract_features()

            if model is None:
                # Fallback to simple rule-based classification if model not loaded
                print("⚠ Using fallback classification (no model loaded)")
                return self._fallback_classify(features)

            prediction = model.predict([features])[0]
            probabilities = model.predict_proba([features])[0]
            classes = model.classes_

            result = {
                "prediction": str(prediction),
                "probabilities": {str(cls): float(prob * 100) for cls, prob in zip(classes, probabilities)}
            }
            print(f"✓ Classification: {prediction}")
            return result
            
        except Exception as e:
            print(f"Error in classify: {e}")
            traceback.print_exc()
            # Return fallback classification
            return self._fallback_classify(self.extract_features())
    
    def _fallback_classify(self, features):
        """Simple rule-based classification as fallback"""
        # Use RMS energy (last feature) and ZCR for basic classification
        rms = features[-1]
        zcr = features[-2]
        
        scores = {
            'Speech': 0.0,
            'Music': 0.0,
            'Noise': 0.0,
            'Silence': 0.0
        }
        
        if rms < 0.001:
            scores['Silence'] = 80.0
            scores['Noise'] = 20.0
        elif rms > 0.01:
            scores['Music'] = 40.0
            scores['Speech'] = 30.0
            scores['Noise'] = 30.0
        else:
            scores['Speech'] = 40.0
            scores['Music'] = 30.0
            scores['Noise'] = 30.0
        
        if zcr > 0.05:
            scores['Noise'] += 10.0
        elif zcr > 0.02:
            scores['Speech'] += 10.0
        
        # Normalize
        total = sum(scores.values())
        scores = {k: (v/total)*100 for k, v in scores.items()}
        
        prediction = max(scores.items(), key=lambda x: x[1])[0]
        
        return {
            "prediction": prediction,
            "probabilities": scores
        }


HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Lobster+Two:ital,wght@0,400;0,700;1,400;1,700&display=swap" rel="stylesheet">
<title>Audio Analysis Lab</title>

<style>
body {
    font-family: Arial, sans-serif;
    background: #0a0e17;
    color: #e8f4f8;
    padding: 20px;
    margin: 0;
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
    background-clip: text;
    margin-bottom: 20px;
}

.btn {
    background: linear-gradient(135deg,#00ffcc,#ff006e);
    color: #0a0e17;
    border: none;
    border-radius: 10px;
    padding: 15px 30px;
    font-size: 1.1rem;
    font-weight: bold;
    cursor: pointer;
    margin-top: 15px;
    transition: all 0.3s ease;
}

.btn:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 5px 20px rgba(0, 255, 204, 0.4);
}

.btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

.card {
    background: #141b2d;
    border-radius: 10px;
    padding: 20px;
    margin-top: 20px;
    border: 2px solid rgba(0, 255, 204, 0.3);
}

.prediction {
    font-size: 1.8rem;
    color: #ff006e;
    margin-bottom: 15px;
    font-weight: bold;
}

.prob-item {
    margin: 10px 0;
    text-align: left;
}

.prob-label {
    display: inline-block;
    width: 100px;
    font-weight: bold;
    color: #00ffcc;
}

.bar {
    height: 10px;
    background: linear-gradient(90deg, #00ffcc, #ff006e);
    margin-top: 5px;
    border-radius: 5px;
    transition: width 0.5s ease;
}

.status {
    margin-top: 15px;
    font-weight: bold;
    color: #00ffcc;
    padding: 10px;
    border-radius: 5px;
}

.status.error {
    color: #ff006e;
    background: rgba(255, 0, 110, 0.1);
}

.status.loading {
    color: #ffd600;
}

input[type="file"] {
    margin: 20px 0;
    padding: 10px;
    background: #141b2d;
    border: 2px solid #00ffcc;
    border-radius: 8px;
    color: #e8f4f8;
}

img {
    max-width: 100%;
    height: auto;
    border-radius: 8px;
}

h3 {
    color: #00ffcc;
    margin-bottom: 15px;
}
</style>

</head>

<body>

<div class="container">

<h1><img src="/static/speaker.gif"
         width="60"
         height="60"
         style="vertical-align: middle; margin-right: 10px;"> Audio Analysis Lab</h1>


<input type="file" id="fileInput" accept=".wav,.mp3">
<br>
<button class="btn" id="analyzeBtn" onclick="analyze()">🔬 Analyze Audio</button>

<div id="status" class="status"></div>

<div id="result"></div>
<div id="visuals"></div>

</div>

<script>
let file;

document.getElementById('fileInput').onchange = e => {
    file = e.target.files[0];
    if (file) {
        document.getElementById('status').innerText = "✓ File selected: " + file.name;
        document.getElementById('status').className = "status";
    }
};

async function analyze(){

    if(!file){
        alert("Please select an audio file first!");
        return;
    }

    let status = document.getElementById('status');
    let btn = document.getElementById('analyzeBtn');

    // Show loading
    status.innerText = "⏳ Analyzing audio...";
    status.className = "status loading";
    btn.disabled = true;

    // Clear previous results
    document.getElementById('result').innerHTML = '';
    document.getElementById('visuals').innerHTML = '';

    let fd = new FormData();
    fd.append('audio', file);

    try{
        console.log("Sending request to /analyze...");
        
        let r = await fetch('/analyze', {
            method: 'POST',
            body: fd
        });

        console.log("Response status:", r.status);
        console.log("Response headers:", r.headers.get('content-type'));

        // Check if response is JSON
        const contentType = r.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            const text = await r.text();
            console.error("Server returned non-JSON response:", text.substring(0, 500));
            throw new Error("Server error: Expected JSON response but got HTML. Check server logs.");
        }

        let d = await r.json();
        console.log("Received data:", d);

        if(d.error){
            status.innerText = "❌ Error: " + d.error;
            status.className = "status error";
            btn.disabled = false;
            return;
        }

        // ===== ML RESULT =====
        let c = d.ml_results.probabilities;
        let prediction = d.ml_results.prediction;

        let html = `
            <div class="card">
                <div class="prediction">Predicted: ${prediction}</div>
        `;

        // Sort by probability descending
        Object.entries(c)
            .sort((a, b) => b[1] - a[1])
            .forEach(([label, value]) => {
                html += `
                    <div class="prob-item">
                        <span class="prob-label">${label}:</span>
                        <span>${value.toFixed(1)}%</span>
                        <div class="bar" style="width:${value}%"></div>
                    </div>
                `;
            });

        html += `</div>`;
        document.getElementById('result').innerHTML = html;

        // ===== VISUALIZATIONS =====
        let v = d.visualizations;

        document.getElementById('visuals').innerHTML = `
            <div class="card">
                <h3>Time-Domain Waveform</h3>
                <img src="${v.waveform}" alt="Waveform">
            </div>

            <div class="card">
                <h3>Spectrogram (STFT)</h3>
                <img src="${v.spectrogram}" alt="Spectrogram">
            </div>

            <div class="card">
                <h3>Mel-Spectrogram</h3>
                <img src="${v.mel_spectrogram}" alt="Mel Spectrogram">
            </div>

            <div class="card">
                <h3>Frequency Spectrum</h3>
                <img src="${v.spectrum}" alt="Frequency Spectrum">
            </div>
        `;

        // Scroll to results
        document.getElementById('result').scrollIntoView({ behavior: 'smooth' });

        // Done
        status.innerText = "✅ Analysis Complete!";
        status.className = "status";

    }catch(e){
        console.error("Error during analysis:", e);
        status.innerText = "❌ Error: " + e.message;
        status.className = "status error";
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
    """Main analysis endpoint with proper error handling"""
    temp_file = None
    try:
        print("\n" + "="*60)
        print("Received analyze request")
        
        if 'audio' not in request.files:
            print("ERROR: No audio file in request")
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        print(f"File received: {file.filename}")
        
        if file.filename == '':
            print("ERROR: Empty filename")
            return jsonify({'error': 'No file selected'}), 400

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
            temp_file = temp.name
            file.save(temp_file)
            print(f"Saved to temp file: {temp_file}")

        # Load audio with librosa
        print("Loading audio with librosa...")
        y, sr = librosa.load(temp_file, sr=22050)
        print(f"✓ Loaded: {len(y)} samples at {sr}Hz ({len(y)/sr:.2f}s)")

        # Analyze
        print("Creating analyzer...")
        analyzer = AudioAnalyzer(y, sr)
        
        print("Generating visualizations...")
        visualizations = {
            'waveform': analyzer.get_waveform(),
            'spectrogram': analyzer.get_spectrogram(),
            'mel_spectrogram': analyzer.get_mel_spectrogram(),
            'spectrum': analyzer.get_spectrum()
        }
        print("✓ Visualizations generated")
        
        print("Performing classification...")
        ml_results = analyzer.classify()
        print("✓ Classification complete")

        response_data = {
            'visualizations': visualizations,
            'ml_results': ml_results
        }
        
        print("✓ Sending response")
        print("="*60 + "\n")
        
        return jsonify(response_data)

    except Exception as e:
        print(f"\n{'='*60}")
        print("ERROR during analysis:")
        print(traceback.format_exc())
        print("="*60 + "\n")
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up temp file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
                print(f"✓ Cleaned up temp file: {temp_file}")
            except:
                pass


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔬 Audio Analysis Lab Starting...")
    print("="*60)
    print(f"Model loaded: {model is not None}")
    print(f"Librosa version: {librosa.__version__}")
    print(f"Numpy version: {np.__version__}")
    print("="*60 + "\n")
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)