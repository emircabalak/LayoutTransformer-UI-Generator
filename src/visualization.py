import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from config import (
    NUM_CLASSES, COMPONENT_CLASSES, CLASS_COLORS_VIZ, CLASS_COLORS_RGB,
    FIGURES_DIR, SAMPLES_DIR, STYLE_PALETTES, NUM_STYLES,
)


def draw_layout(ax, elements, title="Layout"):
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect('equal')
    ax.set_facecolor('#1a1a2e')
    ax.set_title(title, fontsize=10, fontweight='bold', color='white', pad=8)
    ax.tick_params(colors='gray', labelsize=7)

    for elem in elements:
        cls_id = int(elem[0])
        x1, y1, x2, y2 = elem[1], elem[2], elem[3], elem[4]
        style_id = int(elem[5]) if len(elem) > 5 else 0

        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue

        if 0 <= cls_id < NUM_CLASSES:
            cls_name = COMPONENT_CLASSES[cls_id]
            color = CLASS_COLORS_VIZ.get(cls_id, (0.5, 0.5, 0.5, 0.5))
        else:
            continue

        if cls_id == 0:
            rect = patches.FancyBboxPatch(
                (x1, y1), w, h, boxstyle="round,pad=0.008",
                linewidth=2.5, edgecolor=(0.2, 0.5, 1.0),
                facecolor=(0.2, 0.4, 0.8, 0.15)
            )
            ax.add_patch(rect)
            ax.text(x1 + 0.01, y1 + 0.02, cls_name, fontsize=6,
                    color=(0.5, 0.8, 1.0), fontweight='bold')
        else:
            rect = patches.FancyBboxPatch(
                (x1, y1), w, h, boxstyle="round,pad=0.003",
                linewidth=1.2, edgecolor=color[:3],
                facecolor=(*color[:3], 0.4)
            )
            ax.add_patch(rect)
            display_label = f"{cls_name}:s{style_id}"
            ax.text(x1 + w/2, y1 + h/2, display_label, ha='center', va='center',
                    fontsize=5, color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor=color[:3], alpha=0.7))


def save_layout_samples(layouts, num_show=6, filename="generated_layouts.png"):
    valid_layouts = []
    for l in layouts:
        if len(l) >= 1:
            valid_layouts.append(l)

    if not valid_layouts:
        print("[Gorsel] Gosterilecek gecerli layout yok.")
        return

    num_show = min(num_show, len(valid_layouts))
    cols = min(3, num_show)
    rows = (num_show + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows), facecolor='#0d1117')
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(num_show):
        draw_layout(axes[i], valid_layouts[i], f"Layout #{i+1}")
    for i in range(num_show, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150, facecolor='#0d1117')
    plt.close()
    print(f"[Gorsel] Kaydedildi: {path}")


def _get_style(cls_id, style_id):
    if cls_id in STYLE_PALETTES and style_id in STYLE_PALETTES[cls_id]:
        return STYLE_PALETTES[cls_id][style_id]

    if cls_id in STYLE_PALETTES:
        return STYLE_PALETTES[cls_id][0]

    return ((200, 200, 200), (100, 100, 100), "?")


def render_ui_mockup(elements, width=360, height=640):
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new('RGB', (width, height), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)

    font_small = None
    font_medium = None
    candidate_fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "arial.ttf",
    ]
    for font_path in candidate_fonts:
        try:
            font_small = ImageFont.truetype(font_path, 10)
            font_medium = ImageFont.truetype(font_path, 13)
            break
        except (OSError, IOError):
            continue

    def panel_first_key(e):
        if int(e[0]) == 0:
            return 0
        return 1

    sorted_elems = sorted(elements, key=panel_first_key)

    for elem in sorted_elems:
        cls_id = int(elem[0])
        x1 = int(elem[1] * width)
        y1 = int(elem[2] * height)
        x2 = int(elem[3] * width)
        y2 = int(elem[4] * height)
        style_id = int(elem[5]) if len(elem) > 5 else 0

        if x2 <= x1 or y2 <= y1:
            continue
        if cls_id < 0 or cls_id >= NUM_CLASSES:
            continue

        bw = x2 - x1
        bh = y2 - y1
        fill_color, accent_color, label = _get_style(cls_id, style_id)

        if cls_id == 0:
            draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=accent_color)

        elif cls_id == 1:
            r = min(8, bh // 2)
            if style_id == 5:
                draw.rounded_rectangle([x1, y1, x2, y2], radius=r,
                                       fill=(255, 255, 255), outline=accent_color, width=2)
                text_color = accent_color
            else:
                draw.rounded_rectangle([x1, y1, x2, y2], radius=r, fill=fill_color)
                text_color = accent_color
            if font_medium:
                draw.text((x1 + bw//2, y1 + bh//2), label,
                          fill=text_color, font=font_medium, anchor="mm")

        elif cls_id == 2:
            draw.rounded_rectangle([x1, y1, x2, y2], radius=4,
                                   fill=fill_color, outline=(180, 180, 180), width=2)
            if font_small:
                draw.text((x1 + 8, y1 + bh//2), label,
                          fill=accent_color, font=font_small, anchor="lm")

        elif cls_id == 3:
            draw.rectangle([x1, y1, x2, y2], fill=fill_color)
            if bw > 14 and bh > 6:
                line_y = y1 + 4
                while line_y + 3 < y2 - 2:
                    rand_factor = np.random.uniform(0.5, 0.95)
                    line_w = max(4, min(bw - 12, int(bw * rand_factor)))
                    draw.rectangle([x1 + 6, line_y, x1 + 6 + line_w, line_y + 3],
                                   fill=accent_color)
                    line_y += 7

        elif cls_id == 4:
            cx = x1 + bw//2
            cy = y1 + bh//2
            r = min(bw, bh) // 2 - 2
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill_color)
            ir = max(2, r // 3)
            draw.ellipse([cx - ir, cy - ir, cx + ir, cy + ir], fill=accent_color)

        elif cls_id == 5:
            box_size = min(16, bh - 4, bw - 4)
            bx = x1 + 4
            by = y1 + (bh - box_size) // 2
            is_checked = style_id not in (2, 5)

            draw.rectangle([bx, by, bx + box_size, by + box_size],
                           fill=fill_color, outline=accent_color)

            if is_checked:
                draw.line([bx + 3, by + box_size//2, bx + box_size//3, by + box_size - 4],
                          fill=(255, 255, 255), width=2)
                draw.line([bx + box_size//3, by + box_size - 4, bx + box_size - 3, by + 3],
                          fill=(255, 255, 255), width=2)

            if bx + box_size + 8 < x2:
                tw = min(x2 - bx - box_size - 12, 60)
                draw.rectangle([bx + box_size + 8, by + 4, bx + box_size + 8 + tw, by + 7],
                               fill=(80, 80, 80))

        elif cls_id == 6:
            draw.rounded_rectangle([x1, y1, x2, y2], radius=4,
                                   fill=fill_color, outline=(180, 180, 180), width=2)
            ax_center = x2 - 14
            ay_center = y1 + bh // 2
            draw.polygon([(ax_center - 5, ay_center - 3),
                          (ax_center + 5, ay_center - 3),
                          (ax_center, ay_center + 3)], fill=(100, 100, 100))
            if font_small:
                draw.text((x1 + 8, y1 + bh//2), label,
                          fill=accent_color, font=font_small, anchor="lm")

        elif cls_id == 7:
            track_y = y1 + bh // 2
            draw.rectangle([x1 + 4, track_y - 2, x2 - 4, track_y + 2],
                           fill=accent_color)
            fill_w = int((x2 - x1 - 8) * 0.6)
            draw.rectangle([x1 + 4, track_y - 2, x1 + 4 + fill_w, track_y + 2],
                           fill=fill_color)
            tx = x1 + 4 + fill_w
            draw.ellipse([tx - 7, track_y - 7, tx + 7, track_y + 7],
                         fill=fill_color, outline=(255, 255, 255), width=2)

        elif cls_id == 8:
            pill_w = min(44, bw - 4)
            pill_h = min(24, bh - 4)
            px = x1 + (bw - pill_w) // 2
            py = y1 + (bh - pill_h) // 2
            is_on = style_id not in (2, 5)

            draw.rounded_rectangle([px, py, px + pill_w, py + pill_h],
                                   radius=pill_h // 2, fill=fill_color)

            tr = pill_h // 2 - 3
            if is_on:
                tcx = px + pill_w - pill_h // 2
            else:
                tcx = px + pill_h // 2
            tcy = py + pill_h // 2
            draw.ellipse([tcx - tr, tcy - tr, tcx + tr, tcy + tr], fill=accent_color)

        elif cls_id == 9:
            draw.rectangle([x1, y1, x2, y2], fill=fill_color)
            cx = x1 + bw//2
            cy = y1 + bh//2
            s = min(bw, bh) // 4

            if style_id in (0, 2):
                draw.polygon([(cx - s, cy + s//2), (cx, cy - s//2), (cx + s, cy + s//2)],
                             fill=accent_color)
                sr = max(3, s // 3)
                draw.ellipse([cx + s//2 - sr, cy - s//2 - sr,
                              cx + s//2 + sr, cy - s//2 + sr], fill=(255, 193, 7))
            elif style_id in (1, 7):
                hr = max(3, s // 3)
                draw.ellipse([cx - hr, cy - s//2 - hr, cx + hr, cy - s//2 + hr],
                             fill=accent_color)
                draw.arc([cx - s//2, cy - s//4, cx + s//2, cy + s//2],
                         start=0, end=180, fill=accent_color, width=2)
            elif style_id in (3, 5):
                qs = max(3, s // 2)
                draw.rectangle([cx - qs, cy - qs, cx, cy], fill=accent_color)
                draw.rectangle([cx + 2, cy + 2, cx + qs + 2, cy + qs + 2],
                               fill=(*accent_color[:2], min(255, accent_color[2] + 30)))
            else:
                r = max(3, s // 2)
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=accent_color)
                draw.rectangle([cx - r//2, cy + r + 2, cx + r//2, cy + r + 6],
                               fill=(*accent_color[:2], min(255, accent_color[2] + 20)))

    return img


def save_ui_mockups(layouts, num_show=9, filename_prefix="ui_mockup"):
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    valid_layouts = []
    for l in layouts:
        if len(l) >= 1:
            valid_layouts.append(l)

    if not valid_layouts:
        print("[Mockup] Gosterilecek gecerli layout yok.")
        return

    num_show = min(num_show, len(valid_layouts))
    saved = []

    for i in range(num_show):
        img = render_ui_mockup(valid_layouts[i])
        path = os.path.join(SAMPLES_DIR, f"{filename_prefix}_{i}.png")
        img.save(path)
        saved.append(path)

    print(f"[Mockup] {len(saved)} adet UI mockup kaydedildi: {SAMPLES_DIR}")

    if num_show >= 2:
        cols = min(3, num_show)
        rows = (num_show + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 7*rows), facecolor='white')

        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i in range(num_show):
            img = render_ui_mockup(valid_layouts[i])
            axes[i].imshow(np.array(img))
            axes[i].set_title(f"UI #{i+1}", fontsize=10, fontweight='bold')
            axes[i].axis('off')
        for i in range(num_show, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        grid_path = os.path.join(FIGURES_DIR, "ui_mockups_grid.png")
        plt.savefig(grid_path, dpi=150, facecolor='white', bbox_inches='tight')
        plt.close()
        print(f"[Mockup Grid] Kaydedildi: {grid_path}")


def plot_metrics_bar(metrics, filename="metrics.png"):
    labels = ['Overlap\nRate', 'Alignment\nScore', 'Validity']
    values = [
        metrics['avg_overlap_rate'],
        metrics['avg_alignment_score'],
        metrics.get('avg_validity', 0),
    ]
    colors_list = ['#ff6b6b', '#4ecdc4', '#45b7d1']

    fig, ax = plt.subplots(figsize=(8, 5), facecolor='#0d1117')
    ax.set_facecolor('#1a1a2e')
    bars = ax.bar(labels, values, color=colors_list, alpha=0.85,
                  edgecolor='white', linewidth=0.5)

    ax.set_ylabel('Deger', fontsize=12, color='white')
    ax.set_title('LayoutTransformer — Degerlendirme Metrikleri',
                 fontsize=14, fontweight='bold', color='white')
    ax.tick_params(colors='gray')
    ax.grid(axis='y', alpha=0.2)
    ax.set_ylim(0, 1.1)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}', xy=(bar.get_x()+bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                    fontsize=10, color='white', fontweight='bold')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150, facecolor='#0d1117')
    plt.close()
    print(f"[Gorsel] Kaydedildi: {path}")
