import os
import shutil
from tqdm import tqdm

def copy_wav_files(source_dir, target_dir):
    """Wav dosyalarını hedef dizine kopyala ve yeniden adlandır"""
    os.makedirs(target_dir, exist_ok=True)
    
    # Kaynak dizindeki wav dosyalarını bul
    wav_files = [f for f in os.listdir(source_dir) if f.endswith('.wav')]
    print(f"Kaynak dizinde {len(wav_files)} wav dosyası bulundu")
    
    # Dosyaları kopyala ve yeniden adlandır
    copied_count = 0
    for idx, wav_file in enumerate(tqdm(wav_files, desc="Wav dosyaları kopyalanıyor")):
        try:
            # Orijinal dosya: mel_audio_X.wav
            # Hedef format: mel_XXXX.wav
            new_name = f"mel_{idx:04d}.wav"  # 0000, 0001, 0002, ...
                
            source_path = os.path.join(source_dir, wav_file)
            target_path = os.path.join(target_dir, new_name)
            
            shutil.copy2(source_path, target_path)
            copied_count += 1
            
            if idx < 5:  # İlk 5 dönüşümü göster
                print(f"Dönüşüm: {wav_file} -> {new_name}")
                
        except Exception as e:
            print(f"Hata: {wav_file} kopyalanırken hata oluştu - {str(e)}")
    
    print(f"\nKopyalama tamamlandı!")
    print(f"Toplam {copied_count} dosya kopyalandı")
    print(f"Hedef dizin: {target_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', required=True, help='Orijinal wav dosyalarının dizini')
    parser.add_argument('--target_dir', default='datasets/turkish_tts/wavs', help='Hedef dizin')
    
    args = parser.parse_args()
    copy_wav_files(args.source_dir, args.target_dir)