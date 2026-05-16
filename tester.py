import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

SRC_CANDIDATES = [
    os.path.join(HERE, "src"),
    os.path.join(HERE, "Code", "src"),
    os.path.join(HERE, "code", "src"),
]
SRC_DIR = None
for candidate in SRC_CANDIDATES:
    if os.path.isdir(candidate):
        SRC_DIR = candidate
        break

if SRC_DIR is None:
    print("[HATA] Kaynak kod klasoru (src) bulunamadi.")
    print(f"       Aranan yerler: {SRC_CANDIDATES}")
    sys.exit(1)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import DEVICE, COMPONENT_CLASSES, CHECKPOINT_DIR
from model import create_model
from sampling import sample_layouts, sample_conditioned, layouts_to_json
from visualization import save_layout_samples, save_ui_mockups
from metrics import evaluate_layouts, print_metrics


def load_model():
    model = create_model()

    ckpt_path = os.path.join(CHECKPOINT_DIR, "layout_transformer_best.pt")
    if not os.path.exists(ckpt_path):
        print(f"[HATA] Checkpoint bulunamadi: {ckpt_path}")
        print("       Lutfen egitilmis modeli bu yola yerlestirin.")
        sys.exit(1)

    print(f"[Tester] Checkpoint yukleniyor: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        epoch = ckpt.get("epoch", "?")
        print(f"[Tester] Model yuklendi (epoch {epoch})")
    except (RuntimeError, KeyError) as e:
        print(f"[HATA] Checkpoint uyumsuz: {e}")
        sys.exit(1)

    model.eval()
    return model


def menu():
    print()
    print("=" * 50)
    print("  LayoutTransformer Test Aracı")
    print("=" * 50)
    print("  1) Serbest UI uretimi")
    print("  2) Sinif-kosullu UI uretimi")
    print("  3) Cikis")
    print("=" * 50)
    return input("  Seciminiz [1-3]: ").strip()


def run_free(model):
    raw = input("  Kac adet layout uretelim? [varsayilan 9]: ").strip()
    try:
        n = int(raw) if raw else 9
    except ValueError:
        n = 9

    if n < 1:
        n = 1

    print(f"\n[Tester] {n} layout uretiliyor...")
    layouts = sample_layouts(model, num_samples=n, temperature=0.9, top_p=0.9)

    layouts_to_json(layouts, "tester_free_layouts.json")

    show = min(n, 9)
    save_layout_samples(layouts, num_show=show, filename="tester_free_layouts.png")
    save_ui_mockups(layouts, num_show=show, filename_prefix="tester_free_mockup")

    results = evaluate_layouts(layouts)
    print_metrics(results, "Tester (Serbest)")


def run_conditioned(model):
    print(f"\n  Kullanilabilir siniflar: {', '.join(COMPONENT_CLASSES)}")
    raw = input("  Sinif listesini virgulle ayirarak yazin\n  (or: Panel,Button,Button,TextInput,Label,Image): ").strip()

    if not raw:
        print("[Tester] Sinif listesi bos, iptal edildi.")
        return

    class_list = []
    for c in raw.split(","):
        cleaned = c.strip()
        if cleaned:
            class_list.append(cleaned)

    if len(class_list) == 0:
        print("[Tester] Gecerli sinif bulunamadi.")
        return

    raw_n = input("  Kac varyasyon uretelim? [varsayilan 6]: ").strip()
    try:
        n = int(raw_n) if raw_n else 6
    except ValueError:
        n = 6

    if n < 1:
        n = 1

    print(f"\n[Tester] {n} adet kosullu varyasyon uretiliyor...")
    layouts = sample_conditioned(
        model, class_list, num_samples=n,
        temperature=0.9, top_p=0.9,
    )

    layouts_to_json(layouts, "tester_conditioned_layouts.json")

    show = min(n, 9)
    save_layout_samples(layouts, num_show=show, filename="tester_conditioned_layouts.png")
    save_ui_mockups(layouts, num_show=show, filename_prefix="tester_conditioned_mockup")

    results = evaluate_layouts(layouts)
    print_metrics(results, "Tester (Kosullu)")


def main():
    print(f"\n[Tester] Cihaz: {DEVICE}")
    model = load_model()

    while True:
        choice = menu()

        if choice == "1":
            run_free(model)
        elif choice == "2":
            run_conditioned(model)
        elif choice == "3":
            print("\n[Tester] Cikiliyor...")
            break
        else:
            print("[Tester] Gecersiz secim, tekrar deneyin.")


if __name__ == "__main__":
    main()
