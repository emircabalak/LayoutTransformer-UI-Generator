# LayoutTransformer ile UI Layout Uretimi

GPT-tarzi autoregressive transformer kullanarak mobil arayuz (UI) layout'lari ureten bir uretken yapay zeka projesi.

## Proje Hakkinda

Bu proje, RICO veri setindeki gercek mobil uygulama ekranlarindan ogrenilen kaliplari kullanarak yeni UI layout'lari uretir. Model, her UI elemanini **sinif + koordinat + stil** tokenlerinden olusan bir dizi olarak ogrenir ve autoregressive sekilde yeni layout'lar uretir.

### Temel Ozellikler

- **10 UI sinifi**: Panel, Button, TextInput, Label, Icon, CheckBox, Dropdown, Slider, Toggle, Image
- **Stil tokenleri**: Her sinif icin 8 farkli gorsel stil — model hem pozisyon hem gorunum uretiyor
- **Sinif-kosullu uretim**: Kullanicinin belirledigi elemanlara gore layout olusturma
- **Token maskeleme**: Gecersiz token uretimini engelleyen yapisal kisitlar
- **UI mockup renderer**: Bounding box'lardan gercekci gorunumlu arayuz gorselleri

## Proje Yapisi

```
├── config.py                 # Hiperparametreler ve konfigürasyon
├── data_preprocessing.py     # RICO veri seti isleme ve tokenizasyon
├── model.py                  # LayoutTransformer modeli (GPT-tabanli)
├── train.py                  # Egitim dongusu
├── sampling.py               # Layout uretimi ve sinif-kosullu ornekleme
├── visualization.py          # UI mockup renderer (stil destekli)
├── metrics.py                # Degerlendirme metrikleri (validity, alignment, overlap)
├── main.py                   # Ana calistirma scripti
├── requirements.txt          # Bagimlilklar
├── LayoutTransformer_UI.ipynb # Google Colab notebook'u
├── colab_projesi_v4.zip      # Colab icin hazir paket
└── layout_transformer_samples/ # Ornek model ciktilari
```

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanim

### Google Colab (Onerilen)

1. `colab_projesi_v4.zip` dosyasini Google Drive'iniza yukleyin
2. `LayoutTransformer_UI.ipynb` notebook'unu Colab'da acin
3. Hucreleri sirayla calistirin

### Yerel Calistirma

```bash
python tester.py
```

## Model Mimarisi

| Parametre | Deger |
|-----------|-------|
| Embedding boyutu | 256 |
| Attention head | 8 |
| Transformer katmani | 6 |
| Vocab boyutu | 53 (10 sinif + 35 koordinat + 8 stil) |
| Token/eleman | 6 (sinif, x1, y1, x2, y2, stil) |
| Maks dizi uzunlugu | 152 |
| Toplam parametre | ~6.4M |

## Token Yapisi

Her UI elemani 6 token ile temsil edilir:

```
[class, x_min, y_min, x_max, y_max, style]
```

- `class` (0-9): UI eleman sinifi
- `x_min, y_min, x_max, y_max` (10-44): 7x5 grid uzerinde ayriklestirilmis koordinatlar
- `style` (45-52): 8 farkli gorsel stil

## Degerlendirme Sonuclari

| Metrik | Deger |
|--------|-------|
| Validity | %100 |
| Alignment | 0.662 |
| Overlap | 0.179 |

## Ornek Ciktilar

Model tarafindan uretilen layout'lar ve UI mockup'lar `layout_transformer_samples/` klasorunde bulunabilir.

## Veri Seti

Proje, [RICO](https://interactionmining.org/rico) veri setini kullanmaktadir. ~47,830 mobil uygulama ekrani icermektedir.

## Teknolojiler

- Python, PyTorch
- Google Colab (GPU egitimi)
- RICO Dataset
