import os
import json
try:
    from datasets import load_dataset
except ImportError:
    print("Lütfen önce 'datasets' kütüphanesini kurun: pip install datasets")
    exit()


def _count_existing(folder):
    """Klasordeki mevcut JSON dosyalarini say ve indeks setini dondur."""
    existing = set()
    if not os.path.exists(folder):
        return existing
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.endswith('.json'):
                existing.add(os.path.join(root, f))
    return existing


def download_and_save_rico(output_folder="data/rico_jsons", image_folder="data/rico_images",
                           max_samples=2000, skip_existing=True, token=None):
    """
    Hugging Face üzerinden RICO veri setini indirir ve JSON + PNG olarak kaydeder.
    - skip_existing: True ise mevcut dosyalari atlar (resume destegi).
    - token: Hugging Face API token (indirme hizini arttirir).
    """
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(image_folder, exist_ok=True)
    print(f"Klasörler hazır: '{output_folder}', '{image_folder}'")

    # Mevcut dosyalari say (resume icin)
    existing_files = _count_existing(output_folder)
    existing_count = len(existing_files)
    if existing_count > 0:
        print(f"Mevcut dosya sayisi: {existing_count} (skip_existing={skip_existing})")
    if skip_existing and existing_count >= max_samples:
        print(f"Zaten {existing_count} dosya var, indirmeye gerek yok.")
        return

    dataset_candidates = [
        "rootsautomation/RICO-SCA",
        "Fliper/rico",
        "jxu124/rico",
        "nateraw/rico",
        "dipanjyoti/rico",
    ]

    dataset = None
    for repo in dataset_candidates:
        print(f"'{repo}' veri seti deneniyor...")
        try:
            dataset = load_dataset(repo, token=token)
            print(f"Basarili! '{repo}' veri seti indirildi.")
            break
        except Exception:
            print(f"'{repo}' bulunamadi, bir sonrakine geciliyor...")

    if dataset is None:
        print("Listedeki repolara ulasilamadi.")
        return

    splits = dataset.keys()
    print(f"Bulunan alt kümeler: {list(splits)}")

    total_saved = existing_count if skip_existing else 0
    skipped = 0

    for split in splits:
        split_folder = os.path.join(output_folder, split)
        os.makedirs(split_folder, exist_ok=True)
        img_split_folder = os.path.join(image_folder, split)
        os.makedirs(img_split_folder, exist_ok=True)

        print(f"'{split}' verileri kaydediliyor...")

        for idx, item in enumerate(dataset[split]):
            if total_saved >= max_samples:
                print(f"Maksimum limit ({max_samples}) asildi, durduruluyor.")
                print(f"Toplam: {total_saved} kayitli, {skipped} atlanmis.")
                return

            # Dosya yollarini belirle
            filename = f"ui_layout_{idx}.json"
            filepath = os.path.join(split_folder, filename)
            img_filename = f"ui_layout_{idx}.png"
            img_filepath = os.path.join(img_split_folder, img_filename)

            # Zaten varsa atla (resume)
            if skip_existing and os.path.exists(filepath):
                skipped += 1
                continue

            # Resmi kaydet (varsa)
            img_obj = item.get("image", None)
            img_saved = False
            if img_obj is not None:
                try:
                    from PIL import Image as PILImage
                    if isinstance(img_obj, PILImage.Image):
                        img_obj.save(img_filepath)
                        img_saved = True
                except Exception as e:
                    print(f"[Uyari] {idx}. resim kaydedilemedi: {e}")

            # Image objesini JSON'dan cikar
            keys_to_remove = [
                key for key, val in item.items()
                if key == 'image' or "Image" in str(type(val))
            ]
            for key in keys_to_remove:
                del item[key]

            # JSON olarak kaydet
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(item, f, ensure_ascii=False, indent=4)

            total_saved += 1
            if total_saved % 500 == 0:
                saved_with_img = " (resimlerle)" if img_saved else ""
                print(f"   -> {total_saved} dosya kaydedildi{saved_with_img}...")

    print(f"Islem tamamlandi! Toplam {total_saved} adet dosya kaydedildi, {skipped} atlanmis.")

if __name__ == "__main__":
    download_and_save_rico(max_samples=2000)
