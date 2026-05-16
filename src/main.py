import argparse
import torch
import os

from config import DEVICE, NUM_EPOCHS, NUM_SAMPLES, CHECKPOINT_DIR, COMPONENT_CLASSES
from data_preprocessing import get_dataloaders
from model import create_model
from train import train_model, DRIVE_CKPT_DIR
from sampling import sample_layouts, sample_conditioned, layouts_to_json
from metrics import evaluate_layouts, print_metrics
from visualization import save_layout_samples, save_ui_mockups, plot_metrics_bar


def parse_args():
    parser = argparse.ArgumentParser(
        description="LayoutTransformer ile UI Layout Uretimi"
    )
    parser.add_argument("--mode", type=str, default="all",
                        choices=["train", "sample", "evaluate", "conditioned", "all"],
                        help="Calistirma modu")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help="Egitim epoch sayisi")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES,
                        help="Uretilecek ornek sayisi")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch boyutu")
    parser.add_argument("--use-mock-data", action="store_true",
                        help="Sentetik veri kullan")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Ornekleme sicakligi")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Nucleus sampling threshold")
    parser.add_argument("--classes", type=str, default="",
                        help="Kosullu uretim icin sinif listesi (virgul ile ayir)")
    return parser.parse_args()


def _load_best_checkpoint(model):
    ckpt_path = os.path.join(CHECKPOINT_DIR, "layout_transformer_best.pt")

    if not os.path.exists(ckpt_path):
        drive_ckpt = os.path.join(DRIVE_CKPT_DIR, "layout_transformer_best.pt")
        if os.path.exists(drive_ckpt):
            import shutil
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            shutil.copy2(drive_ckpt, ckpt_path)

    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"  [Model] Best checkpoint yuklendi (epoch {ckpt['epoch']})")
            return True
        except (RuntimeError, KeyError) as e:
            print(f"  [UYARI] Checkpoint uyumsuz (model degismis): {e}")
            print(f"  [UYARI] Modeli yeniden egitmeniz gerekiyor!")
            return False
    else:
        print("  [UYARI] Checkpoint bulunamadi, egitimsiz model kullaniliyor.")
        return False


def main():
    args = parse_args()

    print("=" * 60)
    print("  LayoutTransformer — UI Layout Uretimi")
    print("=" * 60)
    print(f"  Cihaz:        {DEVICE}")
    print(f"  Mod:          {args.mode}")
    print(f"  Epoch:        {args.epochs}")
    print(f"  Batch:        {args.batch_size}")
    print(f"  Ornek sayisi: {args.num_samples}")
    print(f"  Temperature:  {args.temperature}")
    print(f"  Top-p:        {args.top_p}")
    if args.classes:
        print(f"  Siniflar:     {args.classes}")
    print("=" * 60)

    model = create_model()

    if args.mode in ["train", "all"]:
        print("\n[1/3] Veri yukleniyor...")
        loaders = get_dataloaders(use_mock=args.use_mock_data, batch_size=args.batch_size)

        print("\n[2/3] Egitim basliyor...")
        model, train_losses, val_losses = train_model(
            model, loaders["train"], loaders["val"],
            num_epochs=args.epochs, resume=True
        )

    if args.mode == "conditioned":
        _load_best_checkpoint(model)

        if not args.classes:
            print("\n[HATA] --classes parametresi gerekli!")
            print("  Ornek: --classes \"Panel,Button,Button,TextInput,Label,Image\"")
            print(f"  Kullanilabilir siniflar: {', '.join(COMPONENT_CLASSES)}")
            return

        class_list = []
        for c in args.classes.split(","):
            class_list.append(c.strip())

        print(f"\n  Kullanilabilir siniflar: {', '.join(COMPONENT_CLASSES)}")

        cond_layouts = sample_conditioned(
            model, class_list, num_samples=args.num_samples,
            temperature=args.temperature, top_p=args.top_p,
        )
        layouts_to_json(cond_layouts, "conditioned_layouts.json")

        metrics = evaluate_layouts(cond_layouts)
        print_metrics(metrics, "Sinif-Kosullu LayoutTransformer")

        num_show = min(9, len(cond_layouts))
        save_layout_samples(cond_layouts, num_show=num_show,
                            filename="conditioned_layouts.png")
        save_ui_mockups(cond_layouts, num_show=num_show,
                        filename_prefix="conditioned_mockup")
        return

    layouts = None

    if args.mode in ["sample", "evaluate", "all"]:
        if args.mode != "all":
            _load_best_checkpoint(model)

        print("\n[3/3] Ornekleme ve degerlendirme...")
        layouts = sample_layouts(
            model, num_samples=args.num_samples,
            temperature=args.temperature, top_p=args.top_p,
        )
        layouts_to_json(layouts)

    if args.mode in ["evaluate", "all"] and layouts is not None:
        metrics = evaluate_layouts(layouts)
        print_metrics(metrics)
        save_layout_samples(layouts, num_show=9, filename="generated_layouts.png")
        save_ui_mockups(layouts, num_show=9)
        plot_metrics_bar(metrics)

    print("\n" + "=" * 60)
    print("  TAMAMLANDI!")
    print("=" * 60)


if __name__ == "__main__":
    main()
