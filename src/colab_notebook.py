# =============================================================================
# Bu dosyanın İÇERİĞİNİ Google Colab'da yeni bir notebook'a yapıştırın.
# Her "# %%"  işareti yeni bir hücre (cell) anlamına gelir.
# Colab'da: Runtime > Change runtime type > GPU > A100 seçin.
# =============================================================================

# %% [markdown]
# # LayoutTransformer ile UI Layout Üretimi
# **Adımlar:**
# 1. Drive'ı bağla ve ZIP'i aç
# 2. Bağımlılıkları kur
# 3. RICO veri setini indir (sadece ilk sefer)
# 4. Veri setini kontrol et
# 5. Modeli eğit
# 6. Örnekleme ve değerlendirme
# 7. Sonuçları görüntüle

# %% --- HÜCRE 1: Drive Bağlantısı ve Kurulum ---
from google.colab import drive
drive.mount('/content/drive')

import os, shutil

ZIP_PATH = "/content/drive/MyDrive/colab_projesi_v4.zip"
WORK_DIR = "/content/ui_layout_transformer"

# Temiz başlangıç
if os.path.exists(WORK_DIR):
    shutil.rmtree(WORK_DIR)
os.makedirs(WORK_DIR, exist_ok=True)

# ZIP'i aç
import zipfile
with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    z.extractall(WORK_DIR)
    print(f"ZIP açıldı: {WORK_DIR}")
    for f in z.namelist():
        print(f"  {f}")

os.chdir(WORK_DIR)
print(f"\nÇalışma dizini: {os.getcwd()}")

# Drive'daki RICO cache'i otomatik bağla
DRIVE_DATA_DIR = "/content/drive/MyDrive/rico_data"
DRIVE_CACHE_FILE = os.path.join(DRIVE_DATA_DIR, "rico_dataset_cache.pt")
LOCAL_CACHE_DIR = os.path.join(WORK_DIR, "data")
LOCAL_CACHE_FILE = os.path.join(LOCAL_CACHE_DIR, "rico_dataset_cache_v2.pt")

os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)

if os.path.exists(DRIVE_CACHE_FILE) and not os.path.exists(LOCAL_CACHE_FILE):
    print("Drive'dan RICO cache kopyalanıyor...")
    shutil.copy2(DRIVE_CACHE_FILE, LOCAL_CACHE_FILE)
    size_mb = os.path.getsize(LOCAL_CACHE_FILE) / (1024 * 1024)
    print(f"Cache kopyalandı! ({size_mb:.1f} MB)")
    print("HÜCRE 3'ü atlayıp direkt HÜCRE 4'e geçebilirsiniz.")
elif os.path.exists(LOCAL_CACHE_FILE):
    size_mb = os.path.getsize(LOCAL_CACHE_FILE) / (1024 * 1024)
    print(f"Yerel RICO cache zaten mevcut ({size_mb:.1f} MB)")
else:
    print("RICO cache bulunamadı - HÜCRE 3'ü çalıştırarak veriyi indirin.")

# Drive checkpoint'ları varsa yerel diske kopyala (resume için)
DRIVE_CKPT_DIR = "/content/drive/MyDrive/layout_transformer_checkpoints"
LOCAL_CKPT_DIR = os.path.join(WORK_DIR, "outputs", "checkpoints")
os.makedirs(LOCAL_CKPT_DIR, exist_ok=True)

if os.path.exists(DRIVE_CKPT_DIR):
    for f in os.listdir(DRIVE_CKPT_DIR):
        if f.endswith('.pt'):
            src = os.path.join(DRIVE_CKPT_DIR, f)
            dst = os.path.join(LOCAL_CKPT_DIR, f)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                print(f"Checkpoint geri yuklendi: {f}")

# %% --- HÜCRE 2: Bağımlılıkları Kur ---
!pip install -q torch torchvision tqdm matplotlib numpy datasets Pillow

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Bellek: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

# %% --- HÜCRE 3: RICO Veri Setini İndir (Sadece ilk sefer) ---
# Cache varsa bu hücreyi ATLAYIN.

import sys
sys.path.insert(0, WORK_DIR)

# ========== HUGGING FACE TOKEN ==========
from google.colab import userdata
try:
    HF_TOKEN = userdata.get('HF_TOKEN')
    print("HF Token Secrets'tan alindi.")
except Exception:
    HF_TOKEN = ""
    if HF_TOKEN:
        print("HF Token manuel olarak ayarlandi.")
    else:
        print("UYARI: HF Token yok, indirme yavas olabilir!")

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
# =========================================

DRIVE_DATA_DIR = "/content/drive/MyDrive/rico_data"
DRIVE_CACHE_FILE = os.path.join(DRIVE_DATA_DIR, "rico_dataset_cache.pt")
LOCAL_CACHE_FILE = os.path.join(WORK_DIR, "data", "rico_dataset_cache_v2.pt")
MAX_SAMPLES = 50000

if os.path.exists(LOCAL_CACHE_FILE):
    print(f"RICO cache zaten mevcut, indirmeye gerek yok.")
    print("Direkt HÜCRE 4'e geçebilirsiniz.")
else:
    print(f"RICO veri seti YEREL DISKE indiriliyor...")
    !pip install -q datasets huggingface_hub

    LOCAL_JSON_DIR = "/content/rico_temp/rico_jsons"
    LOCAL_IMG_DIR = "/content/rico_temp/rico_images"
    os.makedirs(LOCAL_JSON_DIR, exist_ok=True)
    os.makedirs(LOCAL_IMG_DIR, exist_ok=True)

    from ricodownloader import download_and_save_rico
    download_and_save_rico(
        output_folder=LOCAL_JSON_DIR,
        image_folder=LOCAL_IMG_DIR,
        max_samples=MAX_SAMPLES,
        skip_existing=True,
        token=HF_TOKEN or None,
    )

    # Symlink kur (parse için)
    data_dir = os.path.join(WORK_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    for name, local_dir in [("rico_jsons", LOCAL_JSON_DIR), ("rico_images", LOCAL_IMG_DIR)]:
        dst = os.path.join(data_dir, name)
        if not os.path.exists(dst):
            os.symlink(local_dir, dst)

    # Parse et ve cache oluştur
    print("\nVeri seti parse ediliyor ve cache oluşturuluyor...")
    from data_preprocessing import load_rico_dataset
    dataset = load_rico_dataset()

    if dataset and os.path.exists(LOCAL_CACHE_FILE):
        os.makedirs(DRIVE_DATA_DIR, exist_ok=True)
        shutil.copy2(LOCAL_CACHE_FILE, DRIVE_CACHE_FILE)
        size_mb = os.path.getsize(DRIVE_CACHE_FILE) / (1024 * 1024)
        print(f"\nCache Drive'a kaydedildi: {DRIVE_CACHE_FILE} ({size_mb:.1f} MB)")

# %% --- HÜCRE 4: Veri Setini Kontrol Et ---
os.chdir(WORK_DIR)
import sys
sys.path.insert(0, WORK_DIR)
from data_preprocessing import load_rico_dataset, layout_to_tokens, tokens_to_layout
from config import VOCAB_SIZE, MAX_SEQ_LEN, COMPONENT_CLASSES

data = load_rico_dataset()
if data:
    print(f"\nToplam layout: {len(data)}")
    ex = data[0]
    print(f"Örnek eleman sayısı: {len(ex['elements'])}")
    print(f"Örnek elemanlar:")
    for e in ex['elements'][:3]:
        cls_name = COMPONENT_CLASSES[int(e[0])]
        print(f"  {cls_name}: [{e[1]:.3f}, {e[2]:.3f}, {e[3]:.3f}, {e[4]:.3f}]")

    # Token dönüşüm testi
    tokens = layout_to_tokens(ex['elements'])
    recovered = tokens_to_layout(tokens)
    print(f"\nToken sayısı: {len(tokens)}")
    print(f"Geri dönüştürülen eleman: {len(recovered)}")
    print(f"Vocab boyutu: {VOCAB_SIZE}, Max seq len: {MAX_SEQ_LEN}")
else:
    print("HATA: Veri yüklenemedi!")

# %% --- HÜCRE 5: Model Eğitimi ---
# A100'de ~30-60 dakika sürer (200 epoch).
# Bağlantı koparsa kaldığı yerden devam eder (resume=True).

os.chdir(WORK_DIR)
import torch
torch.cuda.empty_cache()

EPOCHS = 200
BATCH_SIZE = 128

!python main.py \
    --mode train \
    --epochs {EPOCHS} \
    --batch-size {BATCH_SIZE}

print("\nModel eğitimi tamamlandı!")

# %% --- HÜCRE 6: Örnekleme + Değerlendirme ---
os.chdir(WORK_DIR)
torch.cuda.empty_cache()

!python main.py \
    --mode all \
    --epochs 0 \
    --num-samples 100 \
    --batch-size {BATCH_SIZE} \
    --temperature 0.9 \
    --top-p 0.9

print("\nÖrnekleme ve değerlendirme tamamlandı!")

# %% --- HÜCRE 7: Sınıf-Koşullu Üretim (Orijinal Katkı) ---
# Kullanıcı hangi UI elemanlarını istediğini belirtir,
# model her eleman için koordinatları otomatik üretir.

import os, torch
os.chdir(WORK_DIR)
torch.cuda.empty_cache()

# === İstediğiniz UI elemanlarını buradan değiştirin ===
SINIFLAR = "Panel,Button,Button,TextInput,Label,Image"
VARYASYON = 6

print(f"Kullanılabilir sınıflar: Panel, Button, TextInput, Label, Icon,")
print(f"  CheckBox, Dropdown, Slider, Toggle, Image")
print(f"\nSeçilen: {SINIFLAR}")
print(f"Varyasyon: {VARYASYON}\n")

!python main.py \
    --mode conditioned \
    --classes "{SINIFLAR}" \
    --num-samples {VARYASYON} \
    --temperature 0.9 \
    --top-p 0.9

from IPython.display import display, Image as IPImage
import glob

print("\n=== Sınıf-Koşullu Layout'lar ===")
cond_img = "outputs/figures/conditioned_layouts.png"
if os.path.exists(cond_img):
    display(IPImage(filename=cond_img, width=900))

print("\n=== Sınıf-Koşullu UI Mockup'lar ===")
for img_path in sorted(glob.glob("outputs/samples/conditioned_mockup_*.png"))[:6]:
    print(f"\n{os.path.basename(img_path)}")
    display(IPImage(filename=img_path, width=360))

print("\nSınıf-koşullu üretim tamamlandı!")

# %% --- HÜCRE 8: Sonuçları Görüntüle ---
import matplotlib.pyplot as plt
from IPython.display import display, Image as IPImage
import glob

print("=== Üretilen Layout'lar ===")
for img_path in sorted(glob.glob("outputs/figures/*.png")):
    print(f"\n{os.path.basename(img_path)}")
    display(IPImage(filename=img_path, width=800))

# %% --- HÜCRE 9: Sonuçları Drive'a Kaydet ---
DRIVE_SAMPLES_DIR = "/content/drive/MyDrive/layout_transformer_samples"
os.makedirs(DRIVE_SAMPLES_DIR, exist_ok=True)

import shutil
for src_dir in ["outputs/samples", "outputs/figures"]:
    if os.path.exists(src_dir):
        for f in os.listdir(src_dir):
            shutil.copy2(os.path.join(src_dir, f), os.path.join(DRIVE_SAMPLES_DIR, f))

print(f"Sonuçlar Drive'a kaydedildi: {DRIVE_SAMPLES_DIR}")
