import os
import shutil
import time
from tqdm import tqdm
import random
import psutil  # Kilit kontrolü için

def is_file_locked(filepath):
    """Dosyanın kilitli olup olmadığını kontrol et."""
    for proc in psutil.process_iter(attrs=['open_files']):
        try:
            files = proc.info['open_files'] or []
            for file in files:
                if file.path == filepath:
                    return True
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return False

def prepare_dataset(wav_dir, mel_dir, output_dir, val_ratio=0.1):
    """Veri setini HifiGAN için hazırla"""
    os.makedirs(output_dir, exist_ok=True)
    wav_output_dir = os.path.join(output_dir, 'wavs')
    os.makedirs(wav_output_dir, exist_ok=True)
    
    # Mel dosyalarını listele
    mel_files = [f for f in os.listdir(mel_dir) if f.endswith('.npy')]
    print(f"Toplam {len(mel_files)} mel spektrogram bulundu")
    
    wav_files = set(f for f in os.listdir(wav_dir) if f.endswith('.wav'))
    print(f"\nÖrnek mel dosyaları:")
    for mel_file in mel_files[:5]:
        print(f"- {mel_file}")
    
    print("\nÖrnek wav dosyaları:")
    for wav_file in list(wav_files)[:5]:
        print(f"- {wav_file}")
    
    # Dosya listelerini oluştur
    all_pairs = []
    for mel_file in tqdm(mel_files, desc="Dosyalar eşleştiriliyor"):
        base_name = mel_file.replace('.npy', '')
        wav_name = f"{base_name}.wav"
        wav_path = os.path.join(wav_dir, wav_name)
        
        if os.path.exists(wav_path):
            print(f"\nEşleşme bulundu: {mel_file} -> {wav_name}")
            try:
                os.replace(wav_path, os.path.join(wav_output_dir, wav_name))
                all_pairs.append(f'wavs/{wav_name}|mels/{mel_file}')
            except Exception as e:
                print(f"Hata: {wav_name} kopyalanırken hata oluştu - {str(e)}")
                continue
    
    # Train/val split
    if all_pairs:
        random.shuffle(all_pairs)
        val_size = int(len(all_pairs) * val_ratio)
        val_list = all_pairs[:val_size]
        train_list = all_pairs[val_size:]
        
        print(f"\nEğitim seti: {len(train_list)} örnek")
        print(f"Validasyon seti: {len(val_list)} örnek")
        
        with open(os.path.join(output_dir, 'train.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(train_list))
        
        with open(os.path.join(output_dir, 'val.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(val_list))
        
        print("\nVeri seti hazırlama tamamlandı!")
    else:
        print("\nHATA: Hiç eşleşme bulunamadı!")
        print("Lütfen wav ve mel dosya isimlerini kontrol edin.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_dir', required=True, help='Orijinal wav dosyalarının dizini')
    parser.add_argument('--mel_dir', required=True, help='Mel spektrogramların dizini')
    parser.add_argument('--output_dir', default='datasets/turkish_tts', help='Çıktı dizini')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validasyon seti oranı')
    
    args = parser.parse_args()
    prepare_dataset(args.wav_dir, args.mel_dir, args.output_dir, args.val_ratio)
