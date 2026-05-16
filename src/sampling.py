import torch
import json
import os
from tqdm import tqdm

from config import (
    DEVICE, NUM_SAMPLES, MAX_SEQ_LEN, TOKENS_PER_ELEMENT,
    VOCAB_SIZE, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN,
    COORD_OFFSET, CLASS_OFFSET, STYLE_OFFSET,
    NUM_COORD_BINS, NUM_CLASSES, NUM_STYLES,
    COMPONENT_CLASSES, SAMPLES_DIR,
    TEMPERATURE, TOP_K, TOP_P,
    token_to_class, token_to_coord, class_to_token, token_to_style,
)
from data_preprocessing import tokens_to_layout


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, temperature=1.0):
    safe_temp = max(temperature, 1e-8)
    logits = logits / safe_temp

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        topk_vals = torch.topk(logits, top_k)[0]
        threshold = topk_vals[..., -1, None]
        indices_to_remove = logits < threshold
        logits[indices_to_remove] = float('-inf')

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')

    return logits


def _get_valid_token_mask(position_in_element, device):
    mask = torch.zeros(VOCAB_SIZE, device=device)

    if position_in_element == 0:
        for c in range(NUM_CLASSES):
            mask[CLASS_OFFSET + c] = 1.0
        mask[EOS_TOKEN] = 1.0
    elif position_in_element <= 4:
        for b in range(NUM_COORD_BINS):
            mask[COORD_OFFSET + b] = 1.0
    else:
        for s in range(NUM_STYLES):
            mask[STYLE_OFFSET + s] = 1.0

    return mask


@torch.no_grad()
def sample_layouts(model, num_samples=NUM_SAMPLES, temperature=TEMPERATURE,
                   top_k=TOP_K, top_p=TOP_P):
    model.eval()
    print(f"\n[Ornekleme] {num_samples} adet layout uretiliyor...")

    generated = torch.full((num_samples, 1), BOS_TOKEN, dtype=torch.long, device=DEVICE)
    finished = torch.zeros(num_samples, dtype=torch.bool, device=DEVICE)

    for step in tqdm(range(MAX_SEQ_LEN - 1), desc="Token uretimi", leave=False):
        if finished.all():
            break

        logits = model(generated)
        next_logits = logits[:, -1, :]

        tokens_generated = generated.shape[1] - 1
        pos_in_elem = tokens_generated % TOKENS_PER_ELEMENT

        valid_mask = _get_valid_token_mask(pos_in_elem, DEVICE)
        valid_mask[PAD_TOKEN] = 0.0
        valid_mask[BOS_TOKEN] = 0.0

        invalid_penalty = (1 - valid_mask).unsqueeze(0) * (-1e10)
        next_logits = next_logits + invalid_penalty

        filtered_logits = top_k_top_p_filtering(
            next_logits.clone(), top_k=top_k, top_p=top_p, temperature=temperature
        )

        probs = torch.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        next_token[finished] = PAD_TOKEN
        finished = finished | (next_token.squeeze(-1) == EOS_TOKEN)

        generated = torch.cat([generated, next_token], dim=1)

    layouts = []
    for i in range(num_samples):
        tokens = generated[i].cpu().tolist()
        elements = tokens_to_layout(tokens)
        layouts.append(elements)

    valid_count = 0
    for l in layouts:
        if len(l) >= 1:
            valid_count += 1
    print(f"[Ornekleme] Tamamlandi. Gecerli layout: {valid_count}/{num_samples}")

    return layouts


def _build_conditioned_prompt(class_list):
    prompt_tokens = [BOS_TOKEN]
    gen_positions = []

    for cls in class_list:
        if isinstance(cls, str):
            cls_name = cls.strip()
            cls_id = None
            for i, name in enumerate(COMPONENT_CLASSES):
                if name.lower() == cls_name.lower():
                    cls_id = i
                    break
            if cls_id is None:
                print(f"  [UYARI] Bilinmeyen sinif: '{cls_name}', atlaniyor.")
                continue
        else:
            cls_id = int(cls)
            if cls_id < 0 or cls_id >= NUM_CLASSES:
                continue

        cls_token = class_to_token(cls_id)
        prompt_tokens.append(cls_token)

        for _ in range(5):
            pos = len(prompt_tokens)
            gen_positions.append(pos)
            prompt_tokens.append(None)

    return prompt_tokens, gen_positions


@torch.no_grad()
def sample_conditioned(model, class_list, num_samples=1, temperature=TEMPERATURE,
                       top_k=TOP_K, top_p=TOP_P):
    model.eval()

    class_names = []
    for cls in class_list:
        if isinstance(cls, str):
            class_names.append(cls)
        elif 0 <= int(cls) < NUM_CLASSES:
            class_names.append(COMPONENT_CLASSES[int(cls)])

    print(f"\n[Kosullu Ornekleme] Siniflar: {class_names}")
    print(f"  {num_samples} adet varyasyon uretiliyor...")

    prompt_template, coord_positions = _build_conditioned_prompt(class_list)
    total_len = len(prompt_template)

    all_layouts = []

    for sample_idx in range(num_samples):
        current_seq = []
        gen_counter = 0

        for tok in prompt_template:
            if tok is not None:
                current_seq.append(tok)
            else:
                input_tensor = torch.tensor([current_seq], dtype=torch.long, device=DEVICE)
                logits = model(input_tensor)
                next_logits = logits[0, -1, :]

                pos_in_gen = gen_counter % 5

                valid_mask = torch.zeros(VOCAB_SIZE, device=DEVICE)
                if pos_in_gen < 4:
                    for b in range(NUM_COORD_BINS):
                        valid_mask[COORD_OFFSET + b] = 1.0
                else:
                    for s in range(NUM_STYLES):
                        valid_mask[STYLE_OFFSET + s] = 1.0

                next_logits = next_logits + (1 - valid_mask) * (-1e10)

                filtered = top_k_top_p_filtering(
                    next_logits.unsqueeze(0).clone(),
                    top_k=top_k, top_p=top_p, temperature=temperature
                )
                probs = torch.softmax(filtered[0], dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

                current_seq.append(next_token)
                gen_counter += 1

        current_seq.append(EOS_TOKEN)

        elements = tokens_to_layout(current_seq)
        all_layouts.append(elements)

    valid_count = 0
    for l in all_layouts:
        if len(l) >= 1:
            valid_count += 1
    print(f"[Kosullu Ornekleme] Tamamlandi. Gecerli: {valid_count}/{num_samples}")

    return all_layouts


def layouts_to_json(layouts, filename="generated_layouts.json"):
    results = []
    for i, elements in enumerate(layouts):
        layout = {"id": i, "elements": []}
        for elem in elements:
            cls_id = int(elem[0])
            if 0 <= cls_id < NUM_CLASSES:
                cls_name = COMPONENT_CLASSES[cls_id]
            else:
                cls_name = "Unknown"

            style_id = int(elem[5]) if len(elem) > 5 else 0

            layout["elements"].append({
                "class": cls_name,
                "class_id": cls_id,
                "x_min": round(elem[1], 4),
                "y_min": round(elem[2], 4),
                "x_max": round(elem[3], 4),
                "y_max": round(elem[4], 4),
                "style_id": style_id,
            })
        results.append(layout)

    path = os.path.join(SAMPLES_DIR, filename)
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[JSON] {len(results)} layout kaydedildi: {path}")
    return results
