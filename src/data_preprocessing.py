import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader

from config import (
    DATA_DIR, RICO_JSON_DIR, RICO_IMAGE_DIR, NUM_CLASSES,
    MAX_PANELS, MAX_COMPONENTS, MAX_ELEMENTS, MAX_SEQ_LEN,
    TOKENS_PER_ELEMENT, BATCH_SIZE, DEVICE,
    PAD_TOKEN, BOS_TOKEN, EOS_TOKEN,
    NUM_STYLES, STYLE_OFFSET,
    coord_to_token, class_to_token, style_to_token,
    COMPONENT_CLASSES, MOCK_DATASET_SIZE,
)


RICO_CLASS_MAP = {
    "Toolbar": 0, "List Item": 0, "Card": 0, "Drawer": 0, "Modal": 0,
    "Bottom Navigation": 0, "Tab Bar": 0,
    "Button": 1, "Text Button": 1, "Icon Button": 1, "FAB": 1,
    "Input": 2, "Text": 3, "Text View": 3, "Icon": 4,
    "Checkbox": 5, "Check Box": 5, "Spinner": 6, "Slider": 7,
    "Switch": 8, "Toggle Button": 8, "Image": 9, "Image View": 9, "Web View": 9,
    "LinearLayout": 0, "RelativeLayout": 0, "FrameLayout": 0,
    "ConstraintLayout": 0, "CoordinatorLayout": 0, "RecyclerView": 0,
    "ListView": 0, "ScrollView": 0, "ViewGroup": 0, "CardView": 0,
    "NavigationView": 0, "TabLayout": 0, "AppBarLayout": 0, "ActionBar": 0,
    "ImageButton": 1, "FloatingActionButton": 1,
    "EditText": 2, "AutoCompleteTextView": 2, "SearchView": 2,
    "TextInputLayout": 2, "TextInputEditText": 2,
    "TextView": 3, "ImageView": 9, "CheckBox": 5, "RadioButton": 5,
    "SeekBar": 7, "ProgressBar": 7, "ToggleButton": 8, "CompoundButton": 8,
}

CHILD_CLASSES = {1, 2, 3, 4, 5, 6, 7, 8, 9}
PANEL_CLASS = 0


def _parse_rico_hierarchy(node, screen_w, screen_h, panels, components, depth=0):
    if not isinstance(node, dict):
        return

    bounds = node.get("bounds", None)
    class_name = node.get("componentLabel", node.get("className", node.get("class", "")))
    if class_name and "." in class_name:
        class_name = class_name.rsplit(".", 1)[-1]

    mapped_class = None
    for key, val in RICO_CLASS_MAP.items():
        if key.lower() in class_name.lower():
            mapped_class = val
            break

    if bounds and mapped_class is not None and len(bounds) == 4:
        x_min = bounds[0] / screen_w
        y_min = bounds[1] / screen_h
        x_max = bounds[2] / screen_w
        y_max = bounds[3] / screen_h

        is_in_unit_x = (0 <= x_min < x_max <= 1)
        is_in_unit_y = (0 <= y_min < y_max <= 1)

        if is_in_unit_x and is_in_unit_y:
            w = x_max - x_min
            h = y_max - y_min
            if w > 0.02 and h > 0.02:
                if mapped_class == PANEL_CLASS and depth < 3:
                    panels.append([mapped_class, x_min, y_min, x_max, y_max])
                elif mapped_class in CHILD_CLASSES:
                    components.append([mapped_class, x_min, y_min, x_max, y_max])

    children = node.get("children", [])
    for child in children:
        _parse_rico_hierarchy(child, screen_w, screen_h, panels, components, depth + 1)


def load_rico_dataset():
    cache_path = os.path.join(DATA_DIR, "rico_dataset_cache_v2.pt")

    if os.path.exists(cache_path):
        print(f"[BILGI] Onbellege alinmis RICO veri seti: {cache_path}")
        try:
            dataset = torch.load(cache_path, weights_only=False)
            print(f"[BILGI] {len(dataset)} adet hazir layout yuklendi.")
            return dataset
        except Exception as e:
            print(f"[UYARI] Onbellek yuklenemedi: {e}. Yeniden parse edilecek...")

    dataset = []

    if not os.path.exists(RICO_JSON_DIR):
        print(f"[HATA] RICO JSON dizini bulunamadi: {RICO_JSON_DIR}")
        return None

    json_files = []
    for root_dir, dirs, files in os.walk(RICO_JSON_DIR):
        for f in files:
            if f.endswith(".json"):
                json_files.append(os.path.join(root_dir, f))

    if len(json_files) == 0:
        print("[HATA] Klasorde hic RICO JSON dosyasi bulunamadi.")
        return None

    print(f"[BILGI] {len(json_files)} adet RICO JSON dosyasi bulundu.")

    skipped = 0
    for jf_path in json_files:
        try:
            with open(jf_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            root = None
            if "view_hierarchy" in data:
                vh = data["view_hierarchy"]
                if isinstance(vh, str):
                    vh = json.loads(vh)
                activity = vh.get("activity", {})
                root = activity.get("root", vh)
            elif "activity" in data:
                root = data.get("activity", {}).get("root", data)
            else:
                root = data

            if root is None:
                skipped += 1
                continue

            panels = []
            components = []
            _parse_rico_hierarchy(root, 1440, 2560, panels, components)

            if len(panels) == 0:
                skipped += 1
                continue

            panels = panels[:MAX_PANELS]
            components = components[:MAX_COMPONENTS]

            elements = panels + components
            elements = elements[:MAX_ELEMENTS]

            if len(elements) >= 3:
                dataset.append({"elements": elements})
            else:
                skipped += 1
        except Exception:
            skipped += 1

    print(f"[BILGI] {len(dataset)} adet gecerli layout yuklendi. ({skipped} atlandi)")

    if len(dataset) > 0:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(dataset, cache_path)
            print(f"[BILGI] Veri seti onbellege kaydedildi: {cache_path}")
        except Exception as e:
            print(f"[UYARI] Onbellek kaydedilemedi: {e}")

    if len(dataset) > 0:
        return dataset
    return None


def _assign_style(cls_id, x_min, y_min, x_max, y_max, elem_idx, total_elems):
    width = x_max - x_min
    height = y_max - y_min
    area = width * height
    cy = (y_min + y_max) / 2.0

    if area > 0.12:
        size_cat = 0
    elif area > 0.04:
        size_cat = 1
    else:
        size_cat = 2

    if cy < 0.33:
        pos_cat = 0
    elif cy < 0.66:
        pos_cat = 1
    else:
        pos_cat = 2

    raw_style = size_cat * 3 + pos_cat + elem_idx
    style = raw_style % NUM_STYLES
    return style


def layout_to_tokens(elements):
    tokens = [BOS_TOKEN]
    total = len(elements)

    for idx, elem in enumerate(elements):
        cls_id = int(elem[0])
        x_min = elem[1]
        y_min = elem[2]
        x_max = elem[3]
        y_max = elem[4]

        if len(elem) >= 6:
            style_id = int(elem[5])
        else:
            style_id = _assign_style(cls_id, x_min, y_min, x_max, y_max, idx, total)

        tokens.append(class_to_token(cls_id))
        tokens.append(coord_to_token(x_min))
        tokens.append(coord_to_token(y_min))
        tokens.append(coord_to_token(x_max))
        tokens.append(coord_to_token(y_max))
        tokens.append(style_to_token(style_id))

    tokens.append(EOS_TOKEN)
    return tokens


def tokens_to_layout(tokens):
    from config import (token_to_class, token_to_coord, token_to_style,
                        CLASS_OFFSET, COORD_OFFSET)

    elements = []
    i = 0

    while i < len(tokens) and tokens[i] == BOS_TOKEN:
        i += 1

    while i + TOKENS_PER_ELEMENT - 1 < len(tokens):
        tok = tokens[i]
        if tok == EOS_TOKEN or tok == PAD_TOKEN:
            break

        if tok >= CLASS_OFFSET and tok < STYLE_OFFSET:
            cls_id = token_to_class(tok)
            if cls_id < 0 or cls_id >= NUM_CLASSES:
                i += TOKENS_PER_ELEMENT
                continue

            coords = []
            valid = True
            for j in range(1, 5):
                ct = tokens[i + j]
                if ct < COORD_OFFSET or ct >= CLASS_OFFSET:
                    valid = False
                    break
                coords.append(token_to_coord(ct))

            style_id = 0
            if valid and i + 5 < len(tokens):
                st = tokens[i + 5]
                if st >= STYLE_OFFSET and st < STYLE_OFFSET + NUM_STYLES:
                    style_id = token_to_style(st)
                else:
                    valid = False

            if valid and len(coords) == 4:
                x_min, y_min, x_max, y_max = coords
                if x_max > x_min and y_max > y_min:
                    elements.append([cls_id, x_min, y_min, x_max, y_max, style_id])

            i += TOKENS_PER_ELEMENT
        else:
            i += 1

    return elements


def generate_mock_layout():
    elements = []
    num_panels = random.randint(1, MAX_PANELS)

    candidate_positions = [i / 10.0 for i in range(1, 10)]
    sample_count = min(num_panels - 1, 8)
    chosen = random.sample(candidate_positions, sample_count)
    y_positions = sorted([0.0] + sorted(chosen) + [1.0])

    panel_count = min(num_panels, len(y_positions) - 1)
    for i in range(panel_count):
        elements.append([PANEL_CLASS, 0.0, y_positions[i], 1.0, y_positions[i + 1]])

    for panel in list(elements):
        _, px1, py1, px2, py2 = panel
        pw = px2 - px1
        ph = py2 - py1

        remaining_slots = MAX_ELEMENTS - len(elements)
        num_comps = random.randint(1, min(4, remaining_slots))
        if num_comps <= 0:
            break

        step = ph / (num_comps + 1)
        for ci in range(num_comps):
            cls = random.choice([1, 2, 3, 4, 5])
            margin_x = pw * random.uniform(0.05, 0.15)
            h = step * random.uniform(0.3, 0.7)
            cy1 = py1 + step * (ci + 0.5) - h / 2
            cy2 = cy1 + h

            elem_x1 = px1 + margin_x
            elem_y1 = max(cy1, py1 + 0.01)
            elem_x2 = px2 - margin_x
            elem_y2 = min(cy2, py2 - 0.01)
            elements.append([cls, elem_x1, elem_y1, elem_x2, elem_y2])

    return {"elements": elements[:MAX_ELEMENTS]}


def generate_mock_dataset(size=MOCK_DATASET_SIZE):
    print(f"[BILGI] {size} adet sentetik layout uretiliyor...")
    result = []
    for _ in range(size):
        result.append(generate_mock_layout())
    return result


class LayoutTokenDataset(Dataset):

    def __init__(self, raw_data):
        self.data = []
        for item in raw_data:
            tokens = layout_to_tokens(item["elements"])
            if len(tokens) > MAX_SEQ_LEN:
                tokens = tokens[:MAX_SEQ_LEN - 1] + [EOS_TOKEN]

            pad_len = MAX_SEQ_LEN - len(tokens)
            tokens = tokens + [PAD_TOKEN] * pad_len
            self.data.append(torch.tensor(tokens, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        padding_mask = (input_ids == PAD_TOKEN)
        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "padding_mask": padding_mask,
        }


def get_dataloaders(use_mock=False, batch_size=BATCH_SIZE, val_split=0.1):
    raw_data = None
    if not use_mock:
        raw_data = load_rico_dataset()

    if raw_data is None:
        if use_mock:
            raw_data = generate_mock_dataset()
        else:
            raise FileNotFoundError(
                "\n[KRITIK HATA] RICO veri seti bulunamadi! "
                "Lutfen once veri setini indirin."
            )

    random.shuffle(raw_data)
    val_size = max(1, int(len(raw_data) * val_split))
    train_data = raw_data[val_size:]
    val_data = raw_data[:val_size]

    print(f"[BILGI] Egitim seti: {len(train_data)}, Dogrulama seti: {len(val_data)}")

    train_ds = LayoutTokenDataset(train_data)
    val_ds = LayoutTokenDataset(val_data)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "raw_data": raw_data,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Veri On Isleme Testi")
    print("=" * 60)

    mock = generate_mock_layout()
    tokens = layout_to_tokens(mock["elements"])
    recovered = tokens_to_layout(tokens)
    print(f"Orijinal eleman: {len(mock['elements'])}")
    print(f"Token sayisi: {len(tokens)}")
    print(f"Geri donusturulen eleman: {len(recovered)}")
    print(f"Ornek token dizisi: {tokens[:15]}...")

    loaders = get_dataloaders(use_mock=True, batch_size=4)
    for batch in loaders["train"]:
        print(f"\nInput shape: {batch['input_ids'].shape}")
        print(f"Target shape: {batch['target_ids'].shape}")
        print(f"Padding mask shape: {batch['padding_mask'].shape}")
        print(f"Ornek input: {batch['input_ids'][0, :20]}")
        break

    print("\n[BASARILI] Veri on isleme testi tamamlandi!")
