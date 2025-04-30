import pickle
import numpy as np
import csv
import os
import pandas as pd

def convert_pkl_to_csv_universal(pkl_file_path, csv_output_path):
    """
    Mengkonversi data dalam format PKL ke format CSV secara universal,
    menangani berbagai jenis data yang mungkin tersimpan dalam file PKL.
    
    Args:
        pkl_file_path (str): Path ke file PKL
        csv_output_path (str): Path untuk menyimpan file CSV hasil konversi
    
    Returns:
        bool: True jika konversi berhasil, False jika tidak
    """
    try:
        # Cek apakah file PKL ada
        if not os.path.exists(pkl_file_path):
            print(f"File PKL tidak ditemukan: {pkl_file_path}")
            return False
        
        # Load data dari file PKL
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Data berhasil dimuat dari {pkl_file_path}")
        print(f"Tipe data: {type(data)}")
        
        # Sesuaikan penanganan berdasarkan tipe data
        if isinstance(data, pd.DataFrame):
            # Jika data adalah DataFrame, simpan langsung ke CSV
            data.to_csv(csv_output_path, index=False)
            print(f"DataFrame berhasil ditulis ke {csv_output_path}")
            return True
            
        elif isinstance(data, dict):
            # Jika data adalah dictionary
            print("Data dalam format dictionary")
            
            # Cek apakah dictionary berisi data training dan label
            if 'data' in data and 'labels' in data:
                # Struktur umum: {'data': [...], 'labels': [...]}
                features = data['data']
                labels = data['labels']
                
                # Pastikan data dan label memiliki ukuran yang sama
                if len(features) != len(labels):
                    print("Ukuran data dan label tidak sama.")
                    return False
                
                # Buat DataFrame
                df = pd.DataFrame(features)
                df['gesture_class'] = labels
                
                # Rename kolom fitur
                num_features = df.shape[1] - 1
                feature_names = [f'feature_{i+1}' for i in range(num_features)]
                column_names = feature_names + ['gesture_class']
                df.columns = column_names
                
                # Simpan ke CSV
                df.to_csv(csv_output_path, index=False)
                print(f"Data dictionary berhasil ditulis ke {csv_output_path} ({len(df)} sampel)")
                return True
            
            # Coba tangani struktur dictionary lainnya
            try:
                # Konversi dictionary menjadi DataFrame
                df = pd.DataFrame.from_dict(data, orient='index').reset_index()
                df.to_csv(csv_output_path, index=False)
                print(f"Dictionary berhasil dikonversi ke {csv_output_path}")
                return True
            except Exception as e:
                print(f"Gagal mengkonversi dictionary: {str(e)}")
                
                # Fallback: simpan key-value pairs sebagai CSV
                with open(csv_output_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['key', 'value'])
                    for key, value in data.items():
                        writer.writerow([key, str(value)])
                print(f"Key-value pairs dari dictionary berhasil ditulis ke {csv_output_path}")
                return True
                
        elif isinstance(data, list) or isinstance(data, np.ndarray):
            # Jika data adalah list atau array
            print(f"Data dalam format {'list' if isinstance(data, list) else 'numpy array'}")
            
            # Cek struktur data
            if len(data) > 0:
                if isinstance(data[0], (list, np.ndarray)) and all(len(item) == len(data[0]) for item in data):
                    # Data adalah array 2D dengan ukuran konsisten
                    # Asumsikan kolom terakhir adalah label kelas
                    df = pd.DataFrame(data)
                    
                    # Rename kolom
                    num_cols = df.shape[1]
                    if num_cols > 1:  # Jika ada lebih dari 1 kolom, asumsikan kolom terakhir adalah label
                        feature_names = [f'feature_{i+1}' for i in range(num_cols-1)]
                        column_names = feature_names + ['gesture_class']
                    else:
                        column_names = ['value']
                    
                    df.columns = column_names
                    
                    # Simpan ke CSV
                    df.to_csv(csv_output_path, index=False)
                    print(f"Data array berhasil ditulis ke {csv_output_path} ({len(df)} sampel)")
                    return True
                else:
                    # Data adalah array 1D atau tidak konsisten
                    with open(csv_output_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['value'])
                        for item in data:
                            writer.writerow([str(item)])
                    print(f"Data 1D berhasil ditulis ke {csv_output_path}")
                    return True
            else:
                print("Data array kosong")
                return False
                
        elif hasattr(data, 'to_csv'):
            # Coba panggil metode to_csv jika tersedia
            data.to_csv(csv_output_path)
            print(f"Data berhasil ditulis ke {csv_output_path} menggunakan metode to_csv")
            return True
            
        else:
            # Coba penanganan khusus untuk model sklearn
            if hasattr(data, 'feature_importances_') or hasattr(data, 'coef_') or hasattr(data, 'theta_'):
                print("Data tampaknya adalah model machine learning")
                
                # Ekstrak informasi yang relevan dari model
                model_info = {}
                for attr in dir(data):
                    if not attr.startswith('_') and not callable(getattr(data, attr)):
                        try:
                            value = getattr(data, attr)
                            if isinstance(value, (np.ndarray, list, int, float, str)):
                                model_info[attr] = value
                        except:
                            pass
                
                # Tulis informasi model ke CSV
                with open(csv_output_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['attribute', 'value'])
                    for attr, value in model_info.items():
                        writer.writerow([attr, str(value)])
                
                print(f"Atribut model berhasil ditulis ke {csv_output_path}")
                return True
            
            else:
                print("Format data tidak dikenali")
                
                # Percobaan terakhir: simpan sebagai dump string
                with open(csv_output_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['data_dump'])
                    writer.writerow([str(data)])
                
                print(f"Data dump berhasil ditulis ke {csv_output_path}")
                return True
        
    except Exception as e:
        print(f"Error selama konversi: {str(e)}")
        return False

# Tambahan: fungsi untuk menganalisis dan menampilkan informasi tentang file PKL
def analyze_pkl_file(pkl_file_path):
    """
    Menganalisis dan menampilkan informasi tentang isi file PKL
    untuk membantu memahami strukturnya.
    
    Args:
        pkl_file_path (str): Path ke file PKL
        
    Returns:
        dict: Informasi tentang isi file PKL
    """
    try:
        # Cek apakah file PKL ada
        if not os.path.exists(pkl_file_path):
            print(f"File PKL tidak ditemukan: {pkl_file_path}")
            return None
        
        # Load data dari file PKL
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"\n===== ANALISIS FILE PKL =====")
        print(f"Path: {pkl_file_path}")
        print(f"Tipe data: {type(data)}")
        
        info = {'type': str(type(data))}
        
        # Analisis berdasarkan tipe data
        if isinstance(data, dict):
            print("\nData adalah dictionary:")
            print(f"Jumlah keys: {len(data)}")
            print(f"Keys: {', '.join(str(k) for k in data.keys())}")
            
            info['keys'] = list(data.keys())
            
            # Tampilkan tipe nilai untuk setiap key
            for key in data:
                val_type = type(data[key])
                print(f"  - {key}: {val_type}")
                
                # Jika nilai adalah array/list, tampilkan ukurannya
                if isinstance(data[key], (list, np.ndarray)):
                    shape = np.array(data[key]).shape
                    print(f"    Shape: {shape}")
                    if len(shape) > 0 and shape[0] > 0:
                        print(f"    Contoh elemen pertama: {data[key][0]}")
                
        elif isinstance(data, (list, np.ndarray)):
            shape = np.array(data).shape
            print("\nData adalah array/list:")
            print(f"Shape: {shape}")
            info['shape'] = shape
            
            if len(shape) > 0 and shape[0] > 0:
                print("\nContoh data pertama:")
                print(data[0])
                info['sample'] = str(data[0])
                
                if len(shape) > 1 and shape[1] > 0:
                    print(f"\nJumlah fitur: {shape[1]}")
                    if isinstance(data[0], (list, np.ndarray)):
                        print(f"Contoh fitur: {data[0]}")
        
        elif hasattr(data, '__dict__'):
            print("\nData adalah objek dengan atribut:")
            attrs = [attr for attr in dir(data) if not attr.startswith('_') and not callable(getattr(data, attr))]
            print(f"Atribut: {', '.join(attrs)}")
            info['attributes'] = attrs
            
            # Tampilkan detail untuk beberapa atribut umum dalam model ML
            for attr in ['classes_', 'feature_names_', 'n_features_in_', 'theta_', 'sigma_', 'class_count_']:
                if hasattr(data, attr):
                    value = getattr(data, attr)
                    print(f"\n{attr}:")
                    
                    if isinstance(value, (np.ndarray, list)):
                        shape = np.array(value).shape
                        print(f"  Shape: {shape}")
                        if len(shape) > 0 and shape[0] > 0:
                            print(f"  Contoh: {value[0]}")
                    else:
                        print(f"  Nilai: {value}")
        
        return info
        
    except Exception as e:
        print(f"Error selama analisis: {str(e)}")
        return None

# Contoh penggunaan
if __name__ == "__main__":
    pkl_file = 'hand_gesture_model.pkl'
    csv_file = 'hand_gesture_data_converted.csv'
    
    # Analisis file PKL
    print("\n=== LANGKAH 1: ANALISIS FILE PKL ===")
    info = analyze_pkl_file(pkl_file)
    
    # Konversi file PKL ke CSV
    print("\n=== LANGKAH 2: KONVERSI PKL KE CSV ===")
    success = convert_pkl_to_csv_universal(pkl_file, csv_file)
    
    if success:
        print("\nKonversi berhasil! File CSV telah dibuat.")
    else:
        print("\nKonversi gagal. Silakan periksa pesan error di atas.")