import streamlit as st
from audiorecorder import audiorecorder
from datetime import datetime
import json
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from PIL import Image
import pytz
import unicodedata
import re
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
import pandas as pd
import parselmouth
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import time
import joblib

service_account_info = json.loads(st.secrets["SERVICE_ACCOUNT_JSON"])
creds = service_account.Credentials.from_service_account_info(service_account_info, scopes=['https://www.googleapis.com/auth/drive.file'])
drive_folder_id = st.secrets["DRIVE_FOLDER_ID"]  # Get from Streamlit secrets
service = build('drive', 'v3', credentials=creds)
vietnam_timezone = pytz.timezone('Asia/Ho_Chi_Minh')

def extract_info(input_string):
  parts = input_string.split("_")

  name = None
  status = None
  gender = None
  age = None
  yod = None

  if parts[0] != '0':
      status = 1
      name = parts[0]
      gender = parts[1]
      age = datetime.now().year - int(parts[2])
      yod = int(parts[3])
  else:
      status = 0
      name = parts[1]
      gender = parts[2]
      age = datetime.now().year - int(parts[3])

  return {
      "name": name,
      "gender": 1 if gender == 'Nam' else 0,
      "age": age,
      "yod": yod,
      "status": status
  }

def sort_dataframe_by_columns(df, columns=None, ascending=True):
    if columns is None:
        columns = df.columns.tolist()  # Sort by all columns

    sorted_df = df.sort_values(by=columns, ascending=ascending)
    return sorted_df
    
def extract_features2(audio_file):
    try:
        file_name = os.path.basename(audio_file)
        info = extract_info(file_name)

        # Load audio files into a waveform (y) and the sample rate (sr)
        # y_parkinson and y_normal are 1D NumPy arrays representing the waveform amplitudes
        # sr_parkinson and sr_normal are the respective sampling rates (e.g., 22050 Hz by default)
        y_au, sr_au = librosa.load(audio_file,sr=48000,mono=True)

        # Feature Extraction using librosa (Example: MFCCs)
        # mfccs_au = librosa.feature.mfcc(y=y_au, sr=sr_au, n_mfcc=13)

        # Basic Comparison of Average MFCCs
        # avg_mfccs_au = np.mean(mfccs_au, axis=1)

        # Load audio (resample + convert to mono)
        # y, sr = librosa.load(file_path, sr=48000, mono=True)

        # Feature parameters
        n_mfcc = 20
        win_length = 1200
        hop_length = 480
        n_fft = 2048
        n_mels = 40
        # window = 'hamming'
        window = 'hann'

        # ===== 🎵 MFCC =====
        mfcc = librosa.feature.mfcc(
            y=y_au,
            sr=sr_au,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window
        )
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        # ===== 🔁 IMFCC =====
        S = librosa.feature.melspectrogram(
            y=y_au,
            sr=sr_au,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            window=window
        )

        # Avoid divide-by-zero
        S_inv = 1 / (S + np.finfo(float).eps)
        log_S_inv = np.log(S_inv)

        imfcc = librosa.feature.mfcc(S=log_S_inv, n_mfcc=n_mfcc)
        imfcc_delta = librosa.feature.delta(imfcc)
        imfcc_delta2 = librosa.feature.delta(imfcc, order=2)

        # ===== 🧠 Combine all features =====
        combined = np.vstack([
            mfcc, mfcc_delta, mfcc_delta2,
            imfcc, imfcc_delta, imfcc_delta2
        ])

        # ===== 📊 Aggregate (mean across time) =====
        feature_vector = np.mean(combined, axis=1)

        # Feature Extraction using parselmouth (Pitch, Jitter, Shimmer, Hnr)
        # Convert audio to Parselmouth Sound objects
        sound_au = parselmouth.Sound(audio_file)

        # Extract pitch (f0)
        pitch_au = sound_au.to_pitch()

        # Extract jitter and shimmer (requires pulse detection)
        point_process_au = parselmouth.praat.call([sound_au, pitch_au], "To PointProcess (cc)")

        jitter_au = parselmouth.praat.call(point_process_au, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.32) * 100
        # Corrected: Extract shimmer from the Sound object directly
        shimmer_au = parselmouth.praat.call([sound_au, point_process_au], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.32, 1.6) * 100
        # print(f"Jitter={jitter_au:.2f}%, Shimmer={shimmer_au:.2f}%")

        # Harmonicity (HNR - Harmonics-to-Noise Ratio)
        hnr_au = parselmouth.praat.call(sound_au, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr_db_au = parselmouth.praat.call(hnr_au, "Get mean", 0, 0)
        # print(f"\nHarmonicity (HNR): {hnr_db_au:.2f} dB")

        # Zero Crossing Rate (tần số âm thanh thay đổi cực tính)
        zcr_au = np.mean(librosa.feature.zero_crossing_rate(y_au))

        # Spectral Centroid (trung tâm phổ)
        centroid_au = np.mean(librosa.feature.spectral_centroid(y=y_au, sr=sr_au))

        # Spectral Bandwidth
        bw_au = np.mean(librosa.feature.spectral_bandwidth(y=y_au, sr=sr_au))

        features = {
            "file": file_name,
            "name": info["name"],
            "gender": info["gender"],
            "age": info["age"],
            "yod": info["yod"],
            "jitter": jitter_au,
            "shimmer": shimmer_au,
            "hnr": hnr_db_au,
            "zcr": zcr_au,
            "centroid": centroid_au,
            "bandwidth": bw_au,
            **{f"mfcc_{i}": feature_vector[i] for i in range(len(feature_vector))},
            "status": info["status"]
        }
        return features
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None
    
def predict_pd(audio, _name, _gender, _year_of_birth, _phone):
    st.audio(audio.export().read())
    
    utc_now = datetime.now().replace(tzinfo=pytz.utc)
    vietnam_now = utc_now.astimezone(vietnam_timezone)
    timestamp = vietnam_now.strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
    __gender = _gender
    if unicodedata.normalize("NFC", _gender) == "Nữ":
        __gender = "Nu"
    filename = f"{_name}_{__gender}_{_year_of_birth}_{_phone}_{timestamp}_a.wav"

    audio.export(filename, format="wav")
    print(filename)
    st.write(f"Frame rate: {audio.frame_rate}, Frame width: {audio.frame_width}, Duration: {audio.duration_seconds} seconds")

    all_features = []
    features = extract_features2(filename)
    if features:
        all_features.append(features)
        print(f"Extracted features for {filename}")
        print(features)
    else:
        print(f"Skipping {filename} due to errors.")
    df = pd.DataFrame(all_features)
    print(df)
    
    # clean data
    df.drop(['file','name','yod','status'], axis=1, inplace=True)
    df = df.fillna(df.mean(numeric_only=True))

    # Load the model and scaler
    loaded_model = joblib.load("rf_cuckoo_model.pkl")
    loaded_mask = joblib.load("selected_features_mask.pkl")
    loaded_scaler = joblib.load("scaler.pkl")

    # # Apply loaded mask and scaler to test data
    # X_test_loaded = loaded_scaler.transform(X_test)
    # X_test_loaded_selected = X_test_loaded[:, loaded_mask == 1]

    # y_pred_loaded = loaded_model.predict(X_test_loaded_selected)

    # Predict
    npy_arr = df.to_numpy()
    print('npy_arr:')
    print(npy_arr)
    # index = pd.Index(['gender','age','jitter','shimmer','hnr','zcr','centroid','bandwidth','mfcc_0','mfcc_1','mfcc_2','mfcc_3','mfcc_4','mfcc_5','mfcc_6','mfcc_7','mfcc_8','mfcc_9','mfcc_10','mfcc_11','mfcc_12'])
    # gender,age,jitter,shimmer,hnr,zcr,centroid,bandwidth,mfcc_0,mfcc_1,mfcc_2,mfcc_3,mfcc_4,mfcc_5,mfcc_6,mfcc_7,mfcc_8,mfcc_9,mfcc_10,mfcc_11,mfcc_12,mfcc_13,mfcc_14,mfcc_15,mfcc_16,mfcc_17,mfcc_18,mfcc_19,mfcc_20,mfcc_21,mfcc_22,mfcc_23,mfcc_24,mfcc_25,mfcc_26,mfcc_27,mfcc_28,mfcc_29,mfcc_30,mfcc_31,mfcc_32,mfcc_33,mfcc_34,mfcc_35,mfcc_36,mfcc_37,mfcc_38,mfcc_39,mfcc_40,mfcc_41,mfcc_42,mfcc_43,mfcc_44,mfcc_45,mfcc_46,mfcc_47,mfcc_48,mfcc_49,mfcc_50,mfcc_51,mfcc_52,mfcc_53,mfcc_54,mfcc_55,mfcc_56,mfcc_57,mfcc_58,mfcc_59,mfcc_60,mfcc_61,mfcc_62,mfcc_63,mfcc_64,mfcc_65,mfcc_66,mfcc_67,mfcc_68,mfcc_69,mfcc_70,mfcc_71,mfcc_72,mfcc_73,mfcc_74,mfcc_75,mfcc_76,mfcc_77,mfcc_78,mfcc_79,mfcc_80,mfcc_81,mfcc_82,mfcc_83,mfcc_84,mfcc_85,mfcc_86,mfcc_87,mfcc_88,mfcc_89,mfcc_90,mfcc_91,mfcc_92,mfcc_93,mfcc_94,mfcc_95,mfcc_96,mfcc_97,mfcc_98,mfcc_99,mfcc_100,mfcc_101,mfcc_102,mfcc_103,mfcc_104,mfcc_105,mfcc_106,mfcc_107,mfcc_108,mfcc_109,mfcc_110,mfcc_111,mfcc_112,mfcc_113,mfcc_114,mfcc_115,mfcc_116,mfcc_117,mfcc_118,mfcc_119,status
    # index = ['gender','age','jitter','shimmer','hnr','zcr','centroid','bandwidth','mfcc_0','mfcc_1','mfcc_2','mfcc_3','mfcc_4','mfcc_5','mfcc_6','mfcc_7','mfcc_8','mfcc_9','mfcc_10','mfcc_11','mfcc_12']
    
    # Base feature names
    base_features = ['gender', 'age', 'jitter', 'shimmer', 'hnr', 'zcr', 'centroid', 'bandwidth']
    # Generate MFCC feature names: mfcc_0 to mfcc_119
    mfcc_features = [f'mfcc_{i}' for i in range(120)]
    # Add the target/status column
    all_features = base_features + mfcc_features
    # Create the pandas Index
    index = pd.Index(all_features)

    new_data = pd.DataFrame(npy_arr, columns=index)
    # new_data = pd.DataFrame(npy_arr, columns=loaded_scaler.feature_names_in_)
    new_data_scaled = loaded_scaler.transform(new_data)
    new_data_scaled_selected = new_data_scaled[:, loaded_mask == 1]
    predictions = loaded_model.predict(new_data_scaled_selected)
    print("\nPredictions using loaded model:\n", predictions)

    file_metadata = {
        'name': filename,
        'parents': [drive_folder_id]
    }

    media = MediaFileUpload(filename, mimetype='audio/wav')
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()

    print(f"Ghi âm '{filename}' đã được lưu vào Google Drive")
    print(f"File ID: {file.get('id')}")

    # Clean up the local file after upload
    os.remove(filename)
    return predictions

st.markdown(
    """
    <link rel="icon" href="favicon.ico" type="image/x-icon">
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .st-emotion-cache-1v0mbdj.e115fcil1 {display: none;}  /* Streamlit Cloud profile footer */
        h1 {
            font-size: 24px;  /* Adjust the size here */
        }
        [data-testid="stColumn"] {
            padding: 0px !important;
        }
        label {
            font-size: 14px; /* Reduced label font size */
            margin-bottom: 2px; /* Reduced margin */
        }
        input, [data-baseweb="input"], [data-baseweb="input-container"] {
            font-size: 14px; /* Reduced input font size */
            padding: 4px; /* Reduced padding */
            margin-bottom: 4px; /* Reduced margin */
        }
    </style>
    """,
    unsafe_allow_html=True,
)

logo = Image.open("logo_new.png")
col1a, col2a = st.columns([2, 4])  # Điều chỉnh tỷ lệ cột tùy ý
with col1a:
    st.image(logo, width=300)
with col2a:
    st.subheader("CHẨN ĐOÁN BỆNH PARKINSON QUA GIỌNG NÓI")
st.write("""
         Giới thiệu: đây là 1 đồ án nghiên cứu, giọng nói ông/bà cô/chú anh/chị sẽ được lưu lại nhằm mục đích nâng cao kết quả nghiên cứu.
         """)
st.markdown("THÔNG TIN CÁ NHÂN:")

col1, col2 = st.columns([1, 2])
with col1:
    st.write("Họ tên:")
with col2:
    name = st.text_input("name_input", key="name_input", label_visibility="collapsed")

col7, col8 = st.columns([1, 2])
with col7:
    st.write("Giới tính:")
with col8:
    gender = st.radio("gender_input", ['Nam', 'Nữ'], key="gender_input", label_visibility="collapsed")

col3, col4 = st.columns([1, 2])
with col3:
    st.write("Năm sinh:")
with col4:
    year_of_birth = st.number_input("yob_input", value=1960, min_value=1900, max_value=2025, step=1, key="yob_input", label_visibility="collapsed")

phone = '0908123456'

# Khởi tạo trạng thái
if "recording" not in st.session_state:
    st.session_state.recording = False
if "start_time" not in st.session_state:
    st.session_state.start_time = None

st.markdown("---")
st.markdown("NỘI DUNG CHẨN ĐOÁN:")

st.write("Hít nhẹ và phát âm nguyên âm “A” thật to, đều, dài và lâu nhất có thể, vd Aaaa..., chú ý không thêm dấu vào như Áááá...")
audio1 = audiorecorder("Ghi âm", "Ngừng ghi âm", custom_style={"backgroundColor": "lightblue"}, key="ghiam1")

if len(audio1) > 0:
    with st.spinner("Đang phân tích..."):
        predict = predict_pd(audio1, name, gender, year_of_birth, phone)
        print(f"Predict: {predict}")
        if predict[0] == 0:
            st.success("Kết quả chẩn đoán: Xác suất bị bệnh thấp")
        else:
            st.success("Kết quả chẩn đoán: Xác suất bị bệnh cao")

st.markdown("---")
st.write("Lời cảm ơn: Xin cảm ơn ông/bà cô/chú anh/chị Cộng Đồng PARKINTON VIỆT NAM, đặc biệt là anh admin Tung Mix vì đã hỗ trợ em thực hiện đồ án này!")
logo2 = Image.open("logo2.png")
st.image(logo2)
