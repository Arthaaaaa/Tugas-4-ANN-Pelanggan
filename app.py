# app.py

from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
from flask import send_file


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Pastikan folder uploads ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model dan scaler
model = tf.keras.models.load_model('model_pelanggan.h5')

df = pd.read_excel('Dataset.xlsx')
scaler = MinMaxScaler()
X = df[['Tanggal']].values
scaler.fit(X)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    predictions_from_file = None
    uploaded_df = None

    if request.method == 'POST':
        if 'tanggal' in request.form:  # Input manual
            tanggal_input = int(request.form['tanggal'])
            tanggal_scaled = scaler.transform(np.array([[tanggal_input]]))
            hasil_prediksi = model.predict(tanggal_scaled)
            prediction = int(hasil_prediksi[0][0])

        if 'file' in request.files:  # Upload file
            file = request.files['file']
            if file.filename != '':
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)

                # Baca file
                uploaded_df = pd.read_excel(filepath)
                if 'Tanggal' in uploaded_df.columns:
                    tanggal_uploaded = uploaded_df[['Tanggal']].values
                    tanggal_scaled_uploaded = scaler.transform(tanggal_uploaded)
                    hasil_prediksi_uploaded = model.predict(tanggal_scaled_uploaded)
                    uploaded_df['Prediksi Pelanggan'] = hasil_prediksi_uploaded.astype(int)
                    predictions_from_file = uploaded_df.to_dict(orient='records')
                    uploaded_df.to_excel(os.path.join(app.config['UPLOAD_FOLDER'], 'hasil_prediksi.xlsx'), index=False)


    return render_template('index.html', prediction=prediction, predictions_from_file=predictions_from_file)

# Tambah route untuk download
@app.route('/download', methods=['POST'])
def download():
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'hasil_prediksi.xlsx')
    uploaded_df = pd.read_excel(file_path)
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
