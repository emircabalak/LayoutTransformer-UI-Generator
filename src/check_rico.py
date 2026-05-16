from data_preprocessing import load_rico_dataset
data = load_rico_dataset()
if data:
    print("Yuklenen layout sayisi:", len(data))
    ex = data[0]
    print("Ornek panels:", len(ex["panels"]))
    print("Ornek components:", len(ex["components"]))
    print("Panel[0]:", ex["panels"][0])
    print("Comp[0]:", ex["components"][0])
else:
    print("HATA: Veri yuklenemedi")
